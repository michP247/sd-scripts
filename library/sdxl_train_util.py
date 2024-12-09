import argparse
import math
import os
from typing import Optional

import torch
from library.device_utils import init_ipex, clean_memory_on_device
init_ipex()

from accelerate import init_empty_weights
from tqdm import tqdm
from transformers import CLIPTokenizer
from library import model_util, sdxl_model_util, train_util, sdxl_original_unet
from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
from .utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

TOKENIZER1_PATH = "openai/clip-vit-large-patch14"
TOKENIZER2_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

# DEFAULT_NOISE_OFFSET = 0.0357


def load_target_model(args, accelerator, model_version: str, weight_dtype):
    model_dtype = match_mixed_precision(args, weight_dtype)  # prepare fp16/bf16
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.info(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

            (
                load_stable_diffusion_format,
                text_encoder1,
                text_encoder2,
                vae,
                unet,
                logit_scale,
                ckpt_info,
            ) = _load_target_model(
                args.pretrained_model_name_or_path,
                args.vae,
                model_version,
                weight_dtype,
                accelerator.device if args.lowram else "cpu",
                model_dtype,
            )

            # work on low-ram device
            if args.lowram:
                text_encoder1.to(accelerator.device)
                text_encoder2.to(accelerator.device)
                unet.to(accelerator.device)
                vae.to(accelerator.device)

            clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info


def _load_target_model(
    name_or_path: str, vae_path: Optional[str], model_version: str, weight_dtype, device="cpu", model_dtype=None
):
    # model_dtype only work with full fp16/bf16
    name_or_path = os.readlink(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers

    if load_stable_diffusion_format:
        logger.info(f"load StableDiffusion checkpoint: {name_or_path}")
        (
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_model_util.load_models_from_sdxl_checkpoint(model_version, name_or_path, device, model_dtype)
    else:
        # Diffusers model is loaded to CPU
        from diffusers import StableDiffusionXLPipeline

        variant = "fp16" if weight_dtype == torch.float16 else None
        logger.info(f"load Diffusers pretrained models: {name_or_path}, variant={variant}")
        try:
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    name_or_path, torch_dtype=model_dtype, variant=variant, tokenizer=None
                )
            except EnvironmentError as ex:
                if variant is not None:
                    logger.info("try to load fp32 model")
                    pipe = StableDiffusionXLPipeline.from_pretrained(name_or_path, variant=None, tokenizer=None)
                else:
                    raise ex
        except EnvironmentError as ex:
            logger.error(
                f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path}"
            )
            raise ex

        text_encoder1 = pipe.text_encoder
        text_encoder2 = pipe.text_encoder_2

        # convert to fp32 for cache text_encoders outputs
        if text_encoder1.dtype != torch.float32:
            text_encoder1 = text_encoder1.to(dtype=torch.float32)
        if text_encoder2.dtype != torch.float32:
            text_encoder2 = text_encoder2.to(dtype=torch.float32)

        vae = pipe.vae
        unet = pipe.unet
        del pipe

        # Diffusers U-Net to original U-Net
        state_dict = sdxl_model_util.convert_diffusers_unet_state_dict_to_sdxl(unet.state_dict())
        with init_empty_weights():
            unet = sdxl_original_unet.SdxlUNet2DConditionModel()  # overwrite unet
        sdxl_model_util._load_state_dict_on_device(unet, state_dict, device=device, dtype=model_dtype)
        logger.info("U-Net converted to original U-Net")

        logit_scale = None
        ckpt_info = None

    # VAEを読み込む
    if vae_path is not None:
        vae = model_util.load_vae(vae_path, weight_dtype)
        logger.info("additional VAE loaded")

    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info


def load_tokenizers(args: argparse.Namespace):
    logger.info("prepare tokenizers")

    original_paths = [TOKENIZER1_PATH, TOKENIZER2_PATH]
    tokeniers = []
    for i, original_path in enumerate(original_paths):
        tokenizer: CLIPTokenizer = None
        if args.tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(args.tokenizer_cache_dir, original_path.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                logger.info(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)

        if tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained(original_path)

        if args.tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
            logger.info(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        if i == 1:
            tokenizer.pad_token_id = 0  # fix pad token id to make same as open clip tokenizer

        tokeniers.append(tokenizer)

    if hasattr(args, "max_token_length") and args.max_token_length is not None:
        logger.info(f"update token length: {args.max_token_length}")

    return tokeniers


def match_mixed_precision(args, weight_dtype):
    if args.full_fp16:
        assert (
            weight_dtype == torch.float16
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        return weight_dtype
    elif args.full_bf16:
        assert (
            weight_dtype == torch.bfloat16
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        return weight_dtype
    else:
        return None


#Modify timestep_embedding function to ensure it is on the TPU
def timestep_embedding(timesteps, dim, max_period=10000, device=None):
    half = dim // 2
    # Correct device placement, pass device to .to() only
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device) 
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(device)


def get_timestep_embedding(x, outdim, device):
    assert len(x.shape) == 2
    b, dims = x.shape[0], x.shape[1]
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim, device=device) # Pass device keyword argument
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb


def get_size_embeddings(orig_size, crop_size, target_size, device):
    emb1 = get_timestep_embedding(orig_size.to(device), 256, device)
    emb2 = get_timestep_embedding(crop_size.to(device), 256, device)
    emb3 = get_timestep_embedding(target_size.to(device), 256, device)
    vector = torch.cat([emb1, emb2, emb3], dim=1)
    return vector.to(device)


def save_sd_model_on_train_end(
    args: argparse.Namespace,  # Add args
    *other_args,              # Use *other_args to simplify argument passing
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(None, args, True, False, False, is_stable_diffusion_ckpt=True)
        sdxl_model_util.save_stable_diffusion_checkpoint(
            ckpt_file,
            text_encoder1,
            text_encoder2,
            unet,
            epoch_no,
            global_step,
            ckpt_info,
            vae,
            logit_scale,
            sai_metadata,
            save_dtype,
        )

    def diffusers_saver(out_dir):
        sdxl_model_util.save_diffusers_checkpoint(
            out_dir,
            text_encoder1,
            text_encoder2,
            unet,
            src_path,
            vae,
            use_safetensors=use_safetensors,
            save_dtype=save_dtype,
        )
    
    # TPU-specific saver
    def xla_saver(ckpt_file, epoch_no, global_step): # Update arguments
        import torch_xla.utils.serialization as xser
        state_dict = { # Create the state_dict directly here
            'model.diffusion_model.': args.unet.state_dict(), # Add args prefix for models
            'conditioner.embedders.0.transformer.': args.text_encoder1.state_dict(),
            'conditioner.embedders.1.model.': sdxl_model_util.convert_text_encoder_2_state_dict_to_sdxl(args.text_encoder2.state_dict(), args.logit_scale),
            'first_stage_model.': model_util.convert_vae_state_dict(args.vae.state_dict()), 
        }
        if getattr(args, 'use_tpu', False):  # Check if using TPU
            xser.save(state_dict, ckpt_file, master_only=True) # Use xser.save
        else: # Save as regular checkpoint if not using TPU
            torch.save(state_dict, ckpt_file)



    if getattr(args, 'use_tpu', False):
        train_util.save_sd_model_on_train_end_common(
            args, *other_args, xla_saver, xla_saver
        )
    else:
        train_util.save_sd_model_on_train_end_common(
            args, *other_args, sd_saver, diffusers_saver
        )


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_sd_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    src_path,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    text_encoder1,
    text_encoder2,
    unet,
    vae,
    logit_scale,
    ckpt_info,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(None, args, True, False, False, is_stable_diffusion_ckpt=True)
        sdxl_model_util.save_stable_diffusion_checkpoint(
            ckpt_file,
            text_encoder1,
            text_encoder2,
            unet,
            epoch_no,
            global_step,
            ckpt_info,
            vae,
            logit_scale,
            sai_metadata,
            save_dtype,
        )

    def diffusers_saver(out_dir):
        sdxl_model_util.save_diffusers_checkpoint(
            out_dir,
            text_encoder1,
            text_encoder2,
            unet,
            src_path,
            vae,
            use_safetensors=use_safetensors,
            save_dtype=save_dtype,
        )

    # TPU-specific saver
    def xla_saver(save_path):
        import torch_xla.utils.serialization as xser
        # Aggregate all model components' state_dicts into a single dictionary
        model_state = {
            'ckpt_file': ckpt_file.state_dict(),
            'text_encoder1': text_encoder1.state_dict(),
            'text_encoder2': text_encoder2.state_dict(),
            'unet': unet.state_dict(),
            'epoch_no': epoch_no.state_dict(),
            'global_step': global_step.state_dict(),
            'ckpt_info': ckpt_info.state_dict(),
            'vae': vae.state_dict(),
            'logit_scale': logit_scale.state_dict(),
            'sai_metadata': sai_metadata.state_dict(),
            'save_dtype': save_dtype.state_dict(),
        }
        
        # Save the aggregated state_dict using XLA serialization
        xser.save(model_state, save_path, master_only=True)
    

    if getattr(args, 'use_tpu', False):
        def tpu_saver(ckpt_file, epoch_no, global_step):
            xla_saver(ckpt_file)
        
        # Pass the TPU-specific saver for both SD and Diffusers formats if applicable
        train_util.save_sd_model_on_epoch_end_or_stepwise_common(
            args,
            on_epoch_end,
            accelerator,
            save_stable_diffusion_format,
            use_safetensors,
            epoch,
            num_train_epochs,
            global_step,
            tpu_saver,        # SD Saver replaced with TPU saver
            tpu_saver         # Diffusers Saver replaced with TPU saver
        )
    else:
        # Original saving method for non-TPU setups
        train_util.save_sd_model_on_epoch_end_or_stepwise_common(
            args,
            on_epoch_end,
            accelerator,
            save_stable_diffusion_format,
            use_safetensors,
            epoch,
            num_train_epochs,
            global_step,
            sd_saver,         # SD Saver
            diffusers_saver    # Diffusers Saver
        )
        


def add_sdxl_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--cache_text_encoder_outputs", action="store_true", help="cache text encoder outputs / text encoderの出力をキャッシュする"
    )
    parser.add_argument(
        "--cache_text_encoder_outputs_to_disk",
        action="store_true",
        help="cache text encoder outputs to disk / text encoderの出力をディスクにキャッシュする",
    )


def verify_sdxl_training_args(args: argparse.Namespace, supportTextEncoderCaching: bool = True):
    assert not args.v2, "v2 cannot be enabled in SDXL training / SDXL学習ではv2を有効にすることはできません"
    if args.v_parameterization:
        logger.warning("v_parameterization will be unexpected / SDXL学習ではv_parameterizationは想定外の動作になります")

    if args.clip_skip is not None:
        logger.warning("clip_skip will be unexpected / SDXL学習ではclip_skipは動作しません")

    # if args.multires_noise_iterations:
    #     logger.info(
    #         f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET}, but noise_offset is disabled due to multires_noise_iterations / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されていますが、multires_noise_iterationsが有効になっているためnoise_offsetは無効になります"
    #     )
    # else:
    #     if args.noise_offset is None:
    #         args.noise_offset = DEFAULT_NOISE_OFFSET
    #     elif args.noise_offset != DEFAULT_NOISE_OFFSET:
    #         logger.info(
    #             f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET} / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されています"
    #         )
    #     logger.info(f"noise_offset is set to {args.noise_offset} / noise_offsetが{args.noise_offset}に設定されました")

    assert (
        not hasattr(args, "weighted_captions") or not args.weighted_captions
    ), "weighted_captions cannot be enabled in SDXL training currently / SDXL学習では今のところweighted_captionsを有効にすることはできません"

    if supportTextEncoderCaching:
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            args.cache_text_encoder_outputs = True
            logger.warning(
                "cache_text_encoder_outputs is enabled because cache_text_encoder_outputs_to_disk is enabled / "
                + "cache_text_encoder_outputs_to_diskが有効になっているためcache_text_encoder_outputsが有効になりました"
            )


def sample_images(*args, **kwargs):
    return train_util.sample_images_common(SdxlStableDiffusionLongPromptWeightingPipeline, *args, **kwargs)
