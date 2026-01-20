# text encoder出力のdiskへの事前キャッシュを行う / cache text encoder outputs to disk in advance

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import math
from multiprocessing import Value
import os

from accelerate.utils import set_seed
from accelerate import cpu_offload
import torch
from tqdm import tqdm

from library import (
    config_util,
    flux_train_utils,
    flux_utils,
    sdxl_model_util,
    strategy_base,
    strategy_flux,
    strategy_sd,
    strategy_sdxl,
)
from library import train_util
from library import sdxl_train_util
from library import utils
import library.sai_model_spec as sai_model_spec
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.utils import setup_logging, add_logging_arguments
from cache_latents import set_tokenize_strategy

setup_logging()
import logging

logger = logging.getLogger(__name__)


def cache_to_disk(args: argparse.Namespace) -> None:
    setup_logging(args, reset=True)
    train_util.prepare_dataset_args(args, True)
    train_util.enable_high_vram(args)

    args.cache_text_encoder_outputs = True
    args.cache_text_encoder_outputs_to_disk = True

    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)

    is_sd = not args.sdxl and not args.flux
    is_sdxl = args.sdxl
    is_flux = args.flux
    cache_t5_only = is_flux and getattr(args, "cache_t5_only", False)

    if not is_sdxl and args.weighted_captions:
        raise ValueError("Weighted captions are only supported for SDXL models")

    set_tokenize_strategy(is_sd, is_sdxl, is_flux, args)

    # Prepare dataset first
    use_user_config = args.dataset_config is not None
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if use_user_config:
            logger.info(f"Loading dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {"datasets": [{"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir, args.reg_data_dir)}]}
            else:
                logger.info("Training with captions.")
                user_config = {"datasets": [{"subsets": [{"image_dir": args.train_data_dir, "metadata_file": args.in_json}]}]}
        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, _ = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)

    # Prepare dtypes - use bf16 for efficiency
    weight_dtype, _ = train_util.prepare_dtype(args)
    t5xxl_dtype = utils.str_to_dtype(args.t5xxl_dtype, weight_dtype) if args.t5xxl_dtype else weight_dtype

    logger.info("Loading models with device_map for low VRAM...")
    # Build a minimal accelerator for model loading and caching (no DeepSpeed)
    args.deepspeed = False
    accelerator = train_util.prepare_accelerator(args)

    if is_sdxl:
        # Load SDXL text encoders with minimal GPU usage; allow caching only TE2 if desired
        _, text_encoder1, text_encoder2, _, _, _, _ = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)
        cache_te2_only = getattr(args, "cache_te2_only", False)
        # Move only necessary encoders to GPU to avoid OOM on small VRAM GPUs
        if not cache_te2_only:
            text_encoder1.to("cuda", dtype=weight_dtype)
        text_encoder2.to("cuda", dtype=weight_dtype)
        text_encoders = [text_encoder1 if not cache_te2_only else None, text_encoder2]
    else:  # is_flux
        from transformers import CLIPTextModel, T5EncoderModel
        from safetensors.torch import load_file

        clip_l = None
        if not cache_t5_only:
            # Load CLIP-L to GPU (fits in 6GB)
            logger.info("Loading CLIP-L to GPU...")
            clip_l = flux_utils.load_clip_l(
                args.clip_l, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors
            )
            clip_l.to("cuda", dtype=weight_dtype)

        # Load T5XXL with device_map to automatically shard across CPU and GPU
        logger.info("Loading T5XXL with device_map='auto' (will shard across CPU and GPU)...")
        import json
        from transformers import T5Config

        T5_CONFIG_JSON = '''
{
  "architectures": ["T5EncoderModel"],
  "classifier_dropout": 0.0,
  "d_ff": 10240,
  "d_kv": 64,
  "d_model": 4096,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 24,
  "num_heads": 64,
  "num_layers": 24,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.41.2",
  "use_cache": true,
  "vocab_size": 32128
}
'''
        config = json.loads(T5_CONFIG_JSON)
        config = T5Config(**config)

        # Load T5XXL with device_map - this will automatically shard
        t5xxl = T5EncoderModel.from_pretrained(
            None,
            config=config,
            state_dict=load_file(args.t5xxl),
            device_map="auto",
            torch_dtype=t5xxl_dtype,
        )

        if cache_t5_only:
            text_encoders = [None, t5xxl]
        else:
            text_encoders = [clip_l, t5xxl]

    for text_encoder in text_encoders:
        if text_encoder is not None:
            text_encoder.requires_grad_(False)
            text_encoder.eval()

    # Build strategies
    if is_sdxl:
        text_encoder_outputs_caching_strategy = strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
            True, None, args.skip_cache_check, is_weighted=args.weighted_captions, cache_te2_only=getattr(args, "cache_te2_only", False)
        )
        text_encoding_strategy = strategy_sdxl.SdxlTextEncodingStrategy()
    else: # is_flux
        text_encoder_outputs_caching_strategy = strategy_flux.FluxTextEncoderOutputsCachingStrategy(True, args.text_encoder_batch_size, args.skip_cache_check, is_partial=cache_t5_only, apply_t5_attn_mask=args.apply_t5_attn_mask)
        text_encoding_strategy = strategy_flux.FluxTextEncodingStrategy(args.apply_t5_attn_mask)

    strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_outputs_caching_strategy)
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    # Create a simple accelerator that reports GPU as device
    class SimpleAccelerator:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.num_processes = 1
            self.process_index = 0
            self.is_main_process = True
            self.is_local_main_process = True

        def wait_for_everyone(self):
            pass

        def print(self, msg):
            logger.info(msg)

        def autocast(self):
            # Return a no-op context manager
            import contextlib
            return contextlib.nullcontext()

    logger.info("Starting caching process with device_map sharding...")
    train_dataset_group.new_cache_text_encoder_outputs(text_encoders, accelerator)

    logger.info(f"Finished caching text encoder outputs to disk.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    train_util.add_training_arguments(parser, True)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_masked_loss_arguments(parser)
    config_util.add_config_arguments(parser)
    train_util.add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)

    parser.add_argument("--sdxl", action="store_true", help="Use SDXL model / SDXLモデルを使用する")
    parser.add_argument("--flux", action="store_true", help="Use FLUX model / FLUXモデルを使用する")
    parser.add_argument(
        "--cache_t5_only",
        action="store_true",
        help="For FLUX, cache only T5XXL outputs",
    )
    parser.add_argument(
        "--cache_te2_only",
        action="store_true",
        help="For SDXL, cache only CLIP_G outputs",
    )
    parser.add_argument(
        "--t5xxl_dtype",
        type=str,
        default=None,
        help="T5XXL model dtype, default: None (use mixed precision dtype) / T5XXLモデルのdtype, デフォルト: None (mixed precisionのdtypeを使用)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="[Deprecated] This option does not work. Existing .npz files are always checked. Use `--skip_cache_check` to skip the check."
        " / [非推奨] このオプションは機能しません。既存の .npz は常に検証されます。`--skip_cache_check` で検証をスキップできます。",
    )
    parser.add_argument(
        "--weighted_captions",
        action="store_true",
        default=False,
        help="Enable weighted captions in the standard style (token:1.3). No commas inside parens, or shuffle/dropout may break the decoder. / 「[token]」、「(token)」「(token:1.3)」のような重み付きキャプションを有効にする。カンマを括弧内に入れるとシャッフルやdropoutで重みづけがおかしくなるので注意",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    cache_to_disk(args)
