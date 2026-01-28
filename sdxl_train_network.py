import argparse
import toml
from typing import List, Optional, Union
import re
from collections import defaultdict
from tqdm import tqdm

import torch
from accelerate import Accelerator
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import sdxl_model_util, sdxl_train_util, strategy_base, strategy_sd, strategy_sdxl, train_util, metrics as metrics_utils
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class SdxlNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True
        self.train_text_encoder_1 = False
        self.train_text_encoder_2 = False

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        sdxl_train_util.verify_sdxl_training_args(args)

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        if args.cache_text_encoder_outputs and not args.network_train_unet_only:
            logger.warning(
                "Text Encoder will be trained with cached text encoder outputs. This may not be what you want. / Text EncoderをキャッシュされたText Encoderの出力で学習します。意図した動作ではない可能性があります。"
            )

        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        super().post_process_network(args, accelerator, network, text_encoders, unet)
        te_loras = getattr(network, "text_encoder_loras", []) or []
        self.train_text_encoder_1 = len(te_loras) > 0 and te_loras[0] is not None
        self.train_text_encoder_2 = len(te_loras) > 1 and te_loras[1] is not None

        if args.cache_text_encoder_outputs:
            # Check if we're using selective caching (only TE2 cached)
            te2_lr = 0
            if hasattr(args, 'text_encoder_lr') and args.text_encoder_lr is not None:
                if isinstance(args.text_encoder_lr, (list, tuple)) and len(args.text_encoder_lr) >= 3:
                    te2_lr = args.text_encoder_lr[2]  # Third value is for TE2
            
            # Only raise error if TE2 learning rate is not 0 (i.e., we're trying to train TE2)
            if self.train_text_encoder_2 and te2_lr != 0:
                raise ValueError("Cannot train Text Encoder 2 with cached text encoder outputs.")

    def load_target_model(self, args, weight_dtype, accelerator):
        # Reuse the robust SDXL loader (handles ZeRO‑3 hollow load, TE fixes, etc.)
        # It returns: load_stable_diffusion_format, te1, te2, vae, unet, logit_scale, ckpt_info
        load_stable_diffusion_format, te1, te2, vae, unet, logit_scale, ckpt_info = sdxl_train_util.load_target_model(
            args, accelerator, "sdxl", weight_dtype
        )
        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.ckpt_info = ckpt_info
        # Expose text encoders for later hooks
        self.text_encoders = [te1, te2]

        # Preemptively harden CLIP-L (TE1) position embedding before ZeRO wrapping and LyCORIS scanning
        try:
            if te1 is not None:
                te = te1.module if hasattr(te1, "module") else te1
                # Ensure config max positions are valid
                if hasattr(te, "config"):
                    te.config.max_position_embeddings = 77
                if hasattr(te, "text_model") and hasattr(te.text_model, "config"):
                    te.text_model.config.max_position_embeddings = 77
                if hasattr(te, "text_model") and hasattr(te.text_model, "embeddings"):
                    emb = te.text_model.embeddings
                    # Ensure position_ids buffer exists
                    if not hasattr(emb, "position_ids") or getattr(emb.position_ids, "shape", torch.Size([1, 0]))[-1] != 77:
                        if hasattr(emb, "position_ids") and hasattr(emb.position_ids, "_is_param"):
                            delattr(emb, "position_ids")
                        emb.register_buffer(
                            "position_ids", torch.arange(77, device=emb.weight.device if hasattr(emb, 'weight') else 'cpu').expand((1, -1)), persistent=False
                        )
                    # Replace position_embedding with buffer-backed embedding to avoid ZeRO param corruption
                    if hasattr(emb, "position_embedding") and hasattr(emb.position_embedding, "weight"):
                        pos_w = emb.position_embedding.weight.detach()
                        hidden = getattr(te.text_model.config, "hidden_size", pos_w.shape[1] if pos_w.dim() == 2 else 768)
                        if pos_w.numel() == 0:
                            pos_w = torch.zeros((77, hidden), dtype=getattr(te, 'dtype', torch.float32))
                        # Match training dtype
                        pos_w = pos_w.to(dtype=getattr(te, 'dtype', torch.float32))

                        class _BufferEmbedding(torch.nn.Module):
                            def __init__(self, weight: torch.Tensor):
                                super().__init__()
                                self.register_buffer('weight', weight, persistent=False)
                            def forward(self, input_ids: torch.LongTensor):
                                return torch.nn.functional.embedding(input_ids, self.weight)

                        emb.position_embedding = _BufferEmbedding(pos_w)
                        try:
                            logger.info(f"[SDXL LoRA] Hardened TE1 position_embedding: shape={tuple(emb.position_embedding.weight.shape)}")
                        except Exception:
                            pass
        except Exception:
            pass

        # Also harden CLIP-G (TE2) positional embeddings similarly
        try:
            if te2 is not None:
                te = te2.module if hasattr(te2, "module") else te2
                if hasattr(te, "config"):
                    te.config.max_position_embeddings = 77
                if hasattr(te, "text_model") and hasattr(te.text_model, "config"):
                    te.text_model.config.max_position_embeddings = 77
                if hasattr(te, "text_model") and hasattr(te.text_model, "embeddings"):
                    emb = te.text_model.embeddings
                    # Ensure position_ids buffer exists
                    if not hasattr(emb, "position_ids") or getattr(emb.position_ids, "shape", torch.Size([1, 0]))[-1] != 77:
                        if hasattr(emb, "position_ids") and hasattr(emb.position_ids, "_is_param"):
                            delattr(emb, "position_ids")
                        emb.register_buffer(
                            "position_ids",
                            torch.arange(77, device=accelerator.device).expand((1, -1)),
                            persistent=False,
                        )
                    # Replace position_embedding with buffer-backed module
                    if hasattr(emb, "position_embedding") and hasattr(emb.position_embedding, "weight"):
                        pos_w = emb.position_embedding.weight.detach()
                        hidden = getattr(te.text_model.config, "hidden_size", pos_w.shape[1] if pos_w.dim() == 2 else 1280)
                        if pos_w.numel() == 0:
                            pos_w = torch.zeros((77, hidden), dtype=getattr(te, 'dtype', torch.float32), device=accelerator.device)
                        else:
                            pos_w = pos_w.to(device=accelerator.device, dtype=getattr(te, 'dtype', torch.float32))

                        class _BufferEmbedding2(torch.nn.Module):
                            def __init__(self, weight: torch.Tensor):
                                super().__init__()
                                self.register_buffer('weight', weight, persistent=False)
                            def forward(self, input_ids: torch.LongTensor):
                                return torch.nn.functional.embedding(input_ids, self.weight)

                        emb.position_embedding = _BufferEmbedding2(pos_w)
                        try:
                            logger.info(f"[SDXL LoRA] Hardened TE2 position_embedding: shape={tuple(emb.position_embedding.weight.shape)}")
                        except Exception:
                            pass
        except Exception:
            pass

        # Replace attention with memory‑efficient variants if requested
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        model_version = sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0
        return model_version, [te1, te2], vae, unet

    def post_accelerator_prepare_hook(self, args, accelerator, training_model):
        is_deepspeed_stage3 = (
            args.deepspeed
            and hasattr(accelerator.state, "deepspeed_plugin")
            and accelerator.state.deepspeed_plugin is not None
            and accelerator.state.deepspeed_plugin.zero_stage == 3
        )

        # Guard: ensure text_encoders is initialized; try to refresh from the prepared DS wrapper
        if not hasattr(self, "text_encoders") or self.text_encoders is None or len(self.text_encoders) == 0:
            try:
                unwrapped = accelerator.unwrap_model(training_model)
                if hasattr(unwrapped, "models") and isinstance(unwrapped.models, torch.nn.ModuleDict):
                    md = unwrapped.models
                    te1 = md.get("text_encoder1", None)
                    te2 = md.get("text_encoder2", None)
                    self.text_encoders = [te1, te2]
                else:
                    self.text_encoders = [None, None]
            except Exception:
                self.text_encoders = [None, None]

        if is_deepspeed_stage3 and self.text_encoders and self.text_encoders[0] is not None:
            clip_l = self.text_encoders[0]
            clip_l_inner = clip_l.module if hasattr(clip_l, "module") else clip_l

            logger.info("Applying CLIP-L position embedding ZeRO-3 fix (buffer-backed)")

            # Force correct config
            try:
                if hasattr(clip_l_inner, "config"):
                    clip_l_inner.config.max_position_embeddings = 77
                if hasattr(clip_l_inner, "text_model") and hasattr(clip_l_inner.text_model, "config"):
                    clip_l_inner.text_model.config.max_position_embeddings = 77
            except Exception:
                pass

            # Replace position_embedding with buffer-backed module to avoid empty ZeRO shard params
            try:
                import deepspeed
            except Exception:
                deepspeed = None

            try:
                if hasattr(clip_l_inner, "text_model") and hasattr(clip_l_inner.text_model, "embeddings"):
                    emb = clip_l_inner.text_model.embeddings

                    # Ensure position_ids buffer exists
                    try:
                        if not hasattr(emb, "position_ids") or emb.position_ids.shape[-1] != 77:
                            if hasattr(emb, "position_ids") and hasattr(emb.position_ids, "_is_param"):
                                delattr(emb, "position_ids")
                            emb.register_buffer(
                                "position_ids",
                                torch.arange(77, device=accelerator.device).expand((1, -1)),
                                persistent=False,
                            )
                    except Exception:
                        pass

                    if hasattr(emb, "position_embedding") and hasattr(emb.position_embedding, "weight"):
                        # Gather existing weight (may be empty due to ZeRO)
                        def _gather_weight():
                            try:
                                if deepspeed is not None:
                                    with deepspeed.zero.GatheredParameters([emb.position_embedding.weight], modifier_rank=None):
                                        return emb.position_embedding.weight.detach().to(accelerator.device)
                                return emb.position_embedding.weight.detach().to(accelerator.device)
                            except Exception:
                                return None

                        pos_w = _gather_weight()
                        hidden = getattr(clip_l_inner.text_model.config, "hidden_size", 768)
                        if pos_w is None or pos_w.numel() == 0:
                            pos_w = torch.zeros((77, hidden), device=accelerator.device, dtype=clip_l_inner.text_model.embeddings.position_embedding.weight.dtype if hasattr(clip_l_inner.text_model.embeddings.position_embedding, 'weight') else torch.float32)

                        class _BufferEmbedding(torch.nn.Module):
                            def __init__(self, weight: torch.Tensor):
                                super().__init__()
                                self.register_buffer("weight", weight, persistent=False)
                            def forward(self, input_ids: torch.LongTensor):
                                return torch.nn.functional.embedding(input_ids, self.weight)

                        emb.position_embedding = _BufferEmbedding(pos_w)
                        # Log final shapes for sanity
                        try:
                            logger.info(f"CLIP-L position_embedding fixed: weight shape={tuple(emb.position_embedding.weight.shape)}")
                        except Exception:
                            pass
            except Exception:
                pass

    def get_tokenize_strategy(self, args):
        return strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sdxl.SdxlTokenizeStrategy):
        return [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            False, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_sdxl.SdxlTextEncodingStrategy()

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        # Check if we're using selective caching (only TE2 cached)
        te2_lr = 0
        if hasattr(args, 'text_encoder_lr') and args.text_encoder_lr is not None:
            if isinstance(args.text_encoder_lr, (list, tuple)) and len(args.text_encoder_lr) >= 3:
                te2_lr = args.text_encoder_lr[2]  # Third value is for TE2
        
        # If TE2 learning rate is 0, we're only training TE1
        # In this case, we need to handle the models list differently
        if te2_lr == 0:
            # Return TE1 and TE2 (for caching), plus unwrapped TE2
            return text_encoders + [accelerator.unwrap_model(text_encoders[-1])]
        else:
            # Default behavior
            return text_encoders + [accelerator.unwrap_model(text_encoders[-1])]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            # Check if the second text encoder learning rate is 0
            te2_lr = 0
            if hasattr(args, 'text_encoder_lr') and args.text_encoder_lr is not None:
                if isinstance(args.text_encoder_lr, (list, tuple)) and len(args.text_encoder_lr) >= 3:
                    te2_lr = args.text_encoder_lr[2]  # Third value is for TE2
            
            # Only cache TE2 if its learning rate is 0
            cache_te2_only = (te2_lr == 0)
            
            return strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, None, args.skip_cache_check, is_weighted=args.weighted_captions, cache_te2_only=cache_te2_only
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # Check if we're using selective caching (only TE2 cached)
            te2_lr = 0
            if hasattr(args, 'text_encoder_lr') and args.text_encoder_lr is not None:
                if isinstance(args.text_encoder_lr, (list, tuple)) and len(args.text_encoder_lr) >= 3:
                    te2_lr = args.text_encoder_lr[2]  # Third value is for TE2
            
            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            # Only load text encoders that will be used
            if text_encoders[0] is not None and te2_lr != 0:  # Only load TE1 if it's not frozen
                text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            if text_encoders[1] is not None:  # Always load TE2 for caching
                text_encoders[1].to(accelerator.device, dtype=weight_dtype)

            # Get the caching strategy with learning rate information
            caching_strategy = self.get_text_encoder_outputs_caching_strategy(args)
            
            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders + [accelerator.unwrap_model(text_encoders[-1])], accelerator)
            accelerator.wait_for_everyone()

            if not self.train_text_encoder_1 and text_encoders[0] is not None:
                text_encoders[0].to("cpu", dtype=torch.float32)
            if not self.train_text_encoder_2 and text_encoders[1] is not None:
                text_encoders[1].to("cpu", dtype=torch.float32)
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            if text_encoders[0] is not None:
                text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            if text_encoders[1] is not None:
                text_encoders[1].to(accelerator.device, dtype=weight_dtype)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
            input_ids1 = batch["input_ids"]
            input_ids2 = batch["input_ids2"]
            with torch.enable_grad():
                # Get the text embedding for conditioning
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)

                encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                    args.max_token_length,
                    input_ids1,
                    input_ids2,
                    tokenizers[0],
                    tokenizers[1],
                    text_encoders[0],
                    text_encoders[1],
                    None if not args.full_fp16 else weight_dtype,
                    accelerator=accelerator,
                )
        else:
            encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
            encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
            pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

        return encoder_hidden_states1, encoder_hidden_states2, pool2

    def call_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
        indices: Optional[List[int]] = None,
    ):
        noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        if indices is not None and len(indices) > 0:
            noisy_latents = noisy_latents[indices]
            timesteps = timesteps[indices]
            text_embedding = text_embedding[indices]
            vector_embedding = vector_embedding[indices]

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        sdxl_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)

    def generate_recommended_weights(self, sorted_results, args, accelerator):
        """Generate recommended learning rate weights based on NFN scores"""
        accelerator.print("\n" + "="*60)
        accelerator.print("Recommended --network_block_lr_weights")
        accelerator.print("="*60)

        # Aggregate granular scores back to block level for block_lr_weight
        block_aggregates = defaultdict(list)
        for key, data in sorted_results.items():
            if "|" in key:
                block_name = key.split("|")[0]
            else:
                block_name = key
            
            if data.get('module_count', 0) > 0 and block_name != "other":
                block_aggregates[block_name].append(data['nfn'])

        # Compute block-level NFN (average of attn/mlp scores)
        block_level_results = {}
        valid_scores = []
        
        # Use the expected order to print debug info nicely
        expected_order = []
        for i in range(9): expected_order.append(f"down_block_{i:02d}")
        expected_order.append("mid_block")
        for i in range(9): expected_order.append(f"up_block_{i:02d}")

        for block_name in expected_order:
            # Extract attn/mlp specific scores for this block
            attn_key = f"{block_name}|attn"
            mlp_key = f"{block_name}|mlp"
            
            s_attn = sorted_results.get(attn_key, {}).get('nfn', None)
            s_mlp = sorted_results.get(mlp_key, {}).get('nfn', None)
            
            scores = []
            if s_attn is not None: scores.append(s_attn)
            if s_mlp is not None: scores.append(s_mlp)
            
            if scores:
                avg_score = sum(scores) / len(scores)
                block_level_results[block_name] = avg_score
                valid_scores.append(avg_score)
                
                # Format debug string
                parts = []
                if s_attn: parts.append(f"Attn={s_attn:.4f}")
                if s_mlp: parts.append(f"MLP={s_mlp:.4f}")
                details = ", ".join(parts)
                
                accelerator.print(f"DEBUG: Block '{block_name}': Avg={avg_score:.4f} [{details}]")

        if not valid_scores:
            accelerator.print("No valid NFN scores found. Cannot generate weights.")
            return

        # Check for inversion (PLoP strategy: adapt misaligned/low-score layers)
        strategy_name = "PLoP (Low Score -> High LR)"
        accelerator.print(f"Strategy: {strategy_name}")

        if getattr(args, 'nfn_target_mean_lr', None) is not None:
            # Proportional scaling mode
            target_mean = args.nfn_target_mean_lr
            accelerator.print(f"Mode: Proportional Scaling (Target Mean LR = {target_mean})")
            
            inverse_scores = [1.0 / (s + 1e-8) for s in valid_scores]
            mean_inverse_score = sum(inverse_scores) / len(inverse_scores)
            scaling_factor = target_mean / mean_inverse_score 
            
            accelerator.print(f"Mean (1/NFN): {mean_inverse_score:.4f}")
            accelerator.print(f"Inverted Scaling Factor (k): {scaling_factor:.4f}")

            def nfn_to_weight(nfn_score):
                return scaling_factor * (1.0 / (nfn_score + 1e-8))
            
            accelerator.print("NOTE: These weights are absolute Learning Rates. Set --learning_rate 1.0 and --unet_lr 1.0")

        else:
            # Min-Max normalization mode
            min_score, max_score = min(valid_scores), max(valid_scores)
            score_range = max_score - min_score if max_score > min_score else 1.0

            accelerator.print(f"Mode: Min-Max Normalization")
            accelerator.print(f"NFN Score Range: Min={min_score:.4f}, Max={max_score:.4f}")
            accelerator.print(f"Weighting Range: Min LR Weight={args.nfn_min_lr_weight}, Max LR Weight={args.nfn_max_lr_weight}")

            def nfn_to_weight(nfn_score):
                normalized = (nfn_score - min_score) / score_range
                # Invert: 0.0 (min score) -> 1.0, 1.0 (max score) -> 0.0
                ratio = 1.0 - normalized
                weight_range = args.nfn_max_lr_weight - args.nfn_min_lr_weight
                return (ratio * weight_range) + args.nfn_min_lr_weight

        weights = []
        def sort_key(x):
            if x.startswith('down_block_'):
                return (0, int(x.split('_')[-1]))
            if x == "mid_block":
                return (1, 0)
            if x.startswith('up_block_'):
                return (2, int(x.split('_')[-1]))
            if x == "conv_in":
                return (3, 0)
            if x == "conv_out":
                return (4, 0)
            if x == "time_embedding":
                return (5, 0)
            if x == "other":
                return (7, 0)
            return (6, 0)

        # Only include actual UNet blocks (down, mid, up) in the weights
        unet_blocks = [k for k in block_level_results.keys() if k.startswith('down_block_') or k == 'mid_block' or k.startswith('up_block_')]
        sorted_keys = sorted(unet_blocks, key=sort_key)

        for block_name in sorted_keys:
            nfn = block_level_results[block_name]
            weight = nfn_to_weight(nfn)
            weights.append(f"{weight:.4f}")

        accelerator.print("\nCopy and paste these into your training arguments:")
        
        # Format the weights as a single string (9 input, 1 middle, 9 output)
        final_weights_list = []
        expected_order = []
        for i in range(9): expected_order.append(f"down_block_{i:02d}")
        expected_order.append("mid_block")
        for i in range(9): expected_order.append(f"up_block_{i:02d}")

        missing_blocks = []
        for block in expected_order:
            if block in block_level_results:
                w = nfn_to_weight(block_level_results[block])
                final_weights_list.append(f"{w:.4f}")
            else:
                missing_blocks.append(block)
                # Default weight if block not found in analysis
                # User requested 0.0 to ensure unmeasured blocks (out of scope) do not influence training
                default_val = 0.0
                final_weights_list.append(f"{default_val:.4f}")

        if missing_blocks:
            accelerator.print(f"\n[WARN] The following blocks were not measured (likely due to scope '{args.nfn_eval_scope}'):")
            accelerator.print(f"       {', '.join(missing_blocks)}")
            accelerator.print(f"       Filled with default value: {final_weights_list[expected_order.index(missing_blocks[0])]}")

        accelerator.print(f' "block_lr_weight={",".join(final_weights_list)}"')
        
        # Specific format for SDXL down/mid/up weights
        down_weights = final_weights_list[0:9]
        mid_weight = final_weights_list[9]
        up_weights = final_weights_list[10:19]
        mid_weights_repeated = [mid_weight] * 3
        
        accelerator.print("\nOr use this format for --network_args:")
        accelerator.print(f'--network_args "down_lr_weight={",".join(down_weights)}" "mid_lr_weight={",".join(mid_weights_repeated)}" "up_lr_weight={",".join(up_weights)}"')
        
        accelerator.print(f"\nTotal blocks: {len(final_weights_list)} (9 inputs + 1 middle + 9 outputs)")
        accelerator.print("\n" + "="*60 + "\n")

    def calculate_and_show_nfn_scores(self, args):
        """Calculate and display NFN (Neural Feature Norm) scores for SDXL UNet blocks"""
        logger.info("preparing accelerator for NFN calculation")
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        # Prepare tokenize strategy (needed by dataset)
        tokenize_strategy = self.get_tokenize_strategy(args)
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

        # Prepare latents caching strategy
        latents_caching_strategy = self.get_latents_caching_strategy(args)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        # Prepare text encoding strategy
        text_encoding_strategy = self.get_text_encoding_strategy(args)
        strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

        # Prepare dataset
        logger.info(f"Loading dataset config from {args.dataset_config}")
        from library.config_util import BlueprintGenerator, ConfigSanitizer
        import library.config_util as config_util

        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
        user_config = config_util.load_user_config(args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)[0]

        train_dataset_group.set_current_strategies()

        from multiprocessing import Value
        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

        import os
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # Load models
        weight_dtype = torch.bfloat16
        model_version, text_encoders, vae, unet = self.load_target_model(args, weight_dtype, accelerator)

        # Prepare models with accelerator
        if args.deepspeed:
            from library import deepspeed_utils
            ds_model = deepspeed_utils.prepare_deepspeed_model(args, unet=unet, vae=vae)
            prepared_obj, train_dataloader = accelerator.prepare(ds_model, train_dataloader)
            unwrapped_model = accelerator.unwrap_model(prepared_obj)
            unet = unwrapped_model.models.unet
            vae = unwrapped_model.models.vae
            unwrapped_unet = unet
        else:
            unet, vae, train_dataloader = accelerator.prepare(unet, vae, train_dataloader)
            unwrapped_unet = accelerator.unwrap_model(unet)

        unwrapped_unet.requires_grad_(False)
        unwrapped_unet.eval()

        # Monkey-patch for DeepSpeed and bf16 time embedding issue
        """ if args.mixed_precision == "bf16" and args.deepspeed:
            accelerator.print("Applying monkey-patch to UNet's time_embed for DeepSpeed+bf16.")
            
            original_time_embed_forward = unwrapped_unet.time_embed.forward
            
            def new_time_embed_forward(input_tensor):
                return original_time_embed_forward(input_tensor.to(torch.bfloat16))

            unwrapped_unet.time_embed.forward = new_time_embed_forward """

        noise_scheduler = self.get_noise_scheduler(args, accelerator.device)

        if is_main_process:
            accelerator.print("\n" + "="*80)
            accelerator.print(f"Starting NFN Score Calculation for SDXL UNet")
            accelerator.print(f"Evaluation Scope: {args.nfn_eval_scope}")
            accelerator.print("="*80)

        all_metrics_from_all_batches = []
        metrics_for_single_batch = defaultdict(lambda: {'nfn': 0.0})
        hooks = metrics_utils.attach_nfn_hooks(unwrapped_unet, metrics_for_single_batch)

        if not hooks:
            if is_main_process:
                accelerator.print("Could not attach any hooks. Aborting NFN analysis.")
            return

        if is_main_process:
            accelerator.print(f"[INFO] Attached {len(hooks)} hooks to UNet modules.")

        num_batches_to_process = len(train_dataloader)
        if args.nfn_batch_limit is not None and args.nfn_batch_limit > 0:
            num_batches_to_process = min(num_batches_to_process, args.nfn_batch_limit)

        if is_main_process:
            accelerator.print(f"Analyzing {num_batches_to_process} batches from your dataset...")

        progress_bar = tqdm(
            range(num_batches_to_process),
            smoothing=0,
            disable=not is_main_process,
            desc="NFN Analysis",
        )
        
        data_iter = iter(train_dataloader)

        for i in range(num_batches_to_process):
            try:
                batch = next(data_iter)
            except StopIteration:
                if is_main_process:
                    accelerator.print(f"Dataset exhausted after {i} batches.")
                break

            with torch.no_grad():
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device)
                else:
                    # vae is already prepared
                    images = batch["images"].to(accelerator.device, dtype=weight_dtype)
                    latents = self.encode_images_to_latents(args, vae, images)
                    vae.to("cpu")
                
                latents = self.shift_scale_latents(args, latents)

                noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)
                bsz = latents.shape[0]

                # Create dummy text and size embeddings
                encoder_hidden_states1 = torch.randn(bsz, 77, 768, device=accelerator.device, dtype=weight_dtype)
                encoder_hidden_states2 = torch.randn(bsz, 77, 1280, device=accelerator.device, dtype=weight_dtype)
                pool2 = torch.randn(bsz, 1280, device=accelerator.device, dtype=weight_dtype)
                
                orig_size = torch.tensor([[1024, 1024]] * bsz, device=accelerator.device)
                crop_size = torch.tensor([[0, 0]] * bsz, device=accelerator.device)
                target_size = torch.tensor([[1024, 1024]] * bsz, device=accelerator.device)
                embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

                vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

                with accelerator.autocast():
                    unet(noisy_latents, timesteps, text_embedding, vector_embedding)

                metrics_to_gather = dict(metrics_for_single_batch)
                if accelerator.num_processes == 1:
                    gathered_metrics_list = [metrics_to_gather]
                else:
                    # Add a barrier for synchronization before gathering
                    accelerator.wait_for_everyone()
                    gathered_metrics_list = accelerator.gather_object(metrics_to_gather)

                if is_main_process:
                    if not isinstance(gathered_metrics_list, list):
                        gathered_metrics_list = [gathered_metrics_list]
                    
                    merged_metrics_for_batch = defaultdict(dict)
                    for process_metrics in gathered_metrics_list:
                        if not isinstance(process_metrics, dict):
                            accelerator.print(f"Warning: process_metrics is not a dict, it is {type(process_metrics)}. Value: {process_metrics}. Skipping.")
                            continue
                        for module_name, metric_values in process_metrics.items():
                            if isinstance(metric_values, dict) and 'nfn' in metric_values:
                                if module_name not in merged_metrics_for_batch:
                                    merged_metrics_for_batch[module_name] = {'nfn': 0.0, 'count': 0}
                                merged_metrics_for_batch[module_name]['nfn'] += metric_values['nfn']
                                merged_metrics_for_batch[module_name]['count'] += 1
                    all_metrics_from_all_batches.append(merged_metrics_for_batch)

                metrics_for_single_batch.clear()

            progress_bar.update(1)

        metrics_utils.remove_nfn_hooks(hooks)
        progress_bar.close()

        if is_main_process:
            if not all_metrics_from_all_batches:
                accelerator.print("No metrics were calculated.")
                return

            avg_metrics = metrics_utils.average_metrics(all_metrics_from_all_batches)

            def get_block_info(module_name):
                block_name = "other"
                layer_type = "other"

                # Determine Block Name
                if "input_blocks" in module_name:
                    match = re.search(r"input_blocks\.([0-9]+)", module_name)
                    if match:
                        block_name = f"down_block_{int(match.group(1)):02d}"
                elif "down_blocks" in module_name:
                    match = re.search(r"down_blocks\.([0-9]+)", module_name)
                    if match:
                        block_name = f"down_block_{int(match.group(1)):02d}"
                elif "middle_block" in module_name or "mid_block" in module_name:
                    block_name = "mid_block"
                elif "output_blocks" in module_name:
                    match = re.search(r"output_blocks\.([0-9]+)", module_name)
                    if match:
                        block_name = f"up_block_{int(match.group(1)):02d}"
                elif "up_blocks" in module_name:
                    match = re.search(r"up_blocks\.([0-9]+)", module_name)
                    if match:
                        block_name = f"up_block_{int(match.group(1)):02d}"
                elif "conv_in" in module_name:
                    block_name = "conv_in"
                elif "conv_out" in module_name:
                    block_name = "conv_out"
                elif "time_embedding" in module_name:
                    block_name = "time_embedding"

                # Determine Layer Type (Attn vs MLP/FF)
                if "attn" in module_name:
                    layer_type = "attn"
                elif "ff" in module_name or "mlp" in module_name: # SDXL often uses 'ff.net'
                    layer_type = "mlp"
                
                return block_name, layer_type

            # Initialize list to track unmatched modules for debugging
            self.unmatched_modules = []
            
            unwrapped_unet = accelerator.unwrap_model(unet)
            
            # We now track scores by (block, type)
            # Types: "attn", "mlp", "other"
            block_scores = defaultdict(lambda: {'nfn_sum': 0.0, 'count': 0})

            for module_name, values in avg_metrics.items():
                block_name, layer_type = get_block_info(module_name)
                
                # Scope filtering
                if args.nfn_eval_scope == "attn" and layer_type != "attn":
                    continue
                elif args.nfn_eval_scope == "attn-mlp" and layer_type not in ["attn", "mlp"]:
                    continue
                elif args.nfn_eval_scope == "full-lin" and "conv" in module_name:
                    continue

                key = f"{block_name}|{layer_type}"
                block_scores[key]['nfn_sum'] += values.get('nfn', 1.0)
                block_scores[key]['count'] += 1
                
                if block_name == "other":
                     if len(self.unmatched_modules) < 10:
                        self.unmatched_modules.append(module_name)

            final_results = {
                key: {
                    'nfn': data['nfn_sum'] / data['count'] if data['count'] > 0 else 0.0,
                    'module_count': data['count']
                }
                for key, data in block_scores.items()
            }

            sorted_results = dict(sorted(final_results.items()))

            # Debug: Print unmatched modules if any
            if self.unmatched_modules:
                accelerator.print(f"DEBUG: First 10 unmatched module names: {self.unmatched_modules}")

            self.generate_recommended_weights(sorted_results, args, accelerator)

            # Optionally build and save a LyCoris preset directly from NFN scores
            try:
                if getattr(args, "nfn_preset_output", None):
                    self.build_and_save_lora_preset_from_nfn(sorted_results, args, accelerator)
            except Exception as e:
                accelerator.print(f"[WARN] Failed to write NFN-based preset: {e}")
                import traceback
                traceback.print_exc()

    def build_and_save_lora_preset_from_nfn(self, sorted_results, args, accelerator):
        """Create a LyCORIS TOML preset using NFN scores.
        Targets SDXL Transformer2DModel attn + MLP.
        """
        def clamp(v, lo, hi):
            return max(lo, min(hi, int(round(v))))

        # Helper to extract scores for a specific type (attn/mlp)
        # Returns a map of block_index -> score
        def extract_scores(prefix, layer_type):
            scores = {}
            for key, rec in sorted_results.items():
                # Key format: "down_block_00|attn"
                if "|" not in key: continue
                b_name, l_type = key.split("|")
                
                if l_type != layer_type: continue
                if not b_name.startswith(prefix): continue
                
                # Extract index from "down_block_00"
                try:
                    if prefix == "mid_block":
                        idx = 0
                    else:
                        idx = int(b_name.split("_")[-1])
                    scores[idx] = float(rec.get("nfn", 0.0) or 0.0)
                except:
                    pass
            return scores

        # Gather all scores to calculate global mean for normalization
        all_valid_scores = [d['nfn'] for k, d in sorted_results.items() if d.get('module_count', 0) > 0 and "other" not in k]
        if not all_valid_scores:
            accelerator.print("No valid scores found for preset generation.")
            return

        global_mean = sum(all_valid_scores) / max(1, len(all_valid_scores))
        
        accelerator.print(f"Global Mean NFN: {global_mean:.4f}")

        # Base ranks from args
        base_attn = int(getattr(args, "preset_base_attn", 48))
        base_mlp = int(getattr(args, "preset_base_mlp", 40))
        min_rank = int(getattr(args, "preset_min_rank", 24))
        max_rank = int(getattr(args, "preset_max_rank", 96))
        
        # Algo settings
        algo = getattr(args, "preset_algo", "lora")
        lokr_factor = int(getattr(args, "preset_lokr_factor", -1))

        # Calculate Rank from Score
        # Logic: Rank ~ 1 / Score. 
        # If Score is Low (Misaligned), Rank should be High.
        # If Score is High (Aligned), Rank should be Low.
        def get_params(score, base_rank):
            params = {}
            
            if score <= 1e-6: 
                target_rank = max_rank 
            else:
                # Normalize score by global mean
                norm_score = score / global_mean
                
                # Invert: multiplier = 1.0 / norm_score
                multiplier = 1.0 / (norm_score + 1e-6)
                target_rank = base_rank * multiplier
            
            final_rank = clamp(target_rank, min_rank, max_rank)
            
            params["algo"] = algo
            params["dim"] = final_rank
            params["alpha"] = final_rank  # Alpha = Rank by default here
            
            # Handle LoKr Factor
            # Low Score (Misaligned) -> Uses provided factor (e.g. 8) for higher quality
            # High Score (Aligned) -> Uses -1 for efficiency
            if algo == "lokr" and lokr_factor != -1:
                if score < global_mean:
                    params["factor"] = lokr_factor
                else:
                    params["factor"] = -1
            
            return params

        down_attn = extract_scores("down_block", "attn")
        down_mlp = extract_scores("down_block", "mlp")
        up_attn = extract_scores("up_block", "attn")
        up_mlp = extract_scores("up_block", "mlp")
        mid_attn = extract_scores("mid_block", "attn")
        mid_mlp = extract_scores("mid_block", "mlp")

        # Build name_algo_map
        name_algo_map = {}
        # Patterns (fnmatch style)
        ATTN_PAT = {
            "input": lambda i: f"input_blocks.{i}.1.transformer_blocks.*.attn*",
            "middle": lambda i: f"middle_block.{i}.transformer_blocks.*.attn*",
            "output": lambda i: f"output_blocks.{i}.1.transformer_blocks.*.attn*",
        }
        MLP_PAT = {
            "input": lambda i: f"input_blocks.{i}.1.transformer_blocks.*.ff.net.*",
            "middle": lambda i: f"middle_block.{i}.transformer_blocks.*.ff.net.*",
            "output": lambda i: f"output_blocks.{i}.1.transformer_blocks.*.ff.net.*",
        }

        # Input blocks
        for i in range(9):
            s_attn = down_attn.get(i, global_mean)
            s_mlp = down_mlp.get(i, global_mean)
            
            name_algo_map[ATTN_PAT["input"](i)] = get_params(s_attn, base_attn)
            name_algo_map[MLP_PAT["input"](i)] = get_params(s_mlp, base_mlp)

        # Middle block
        for i in range(1):
            s_attn = mid_attn.get(0, global_mean)
            s_mlp = mid_mlp.get(0, global_mean)
            
            name_algo_map[ATTN_PAT["middle"](i)] = get_params(s_attn, base_attn)
            name_algo_map[MLP_PAT["middle"](i)] = get_params(s_mlp, base_mlp)

        # Output blocks
        for i in range(9):
            s_attn = up_attn.get(i, global_mean)
            s_mlp = up_mlp.get(i, global_mean)
            
            name_algo_map[ATTN_PAT["output"](i)] = get_params(s_attn, base_attn)
            name_algo_map[MLP_PAT["output"](i)] = get_params(s_mlp, base_mlp)

        preset = {
            "enable_conv": False,
            "use_fnmatch": True,
            "unet_target_module": ["Transformer2DModel"],
            "unet_target_name": [
                "input_blocks.*",
                "middle_block.*",
                "output_blocks.*",
            ],
            "text_encoder_target_module": [],
            "text_encoder_target_name": [],
            "name_algo_map": name_algo_map,
        }

        out_path = args.nfn_preset_output
        try:
            with open(out_path, "w") as f:
                toml.dump(preset, f)
            accelerator.print(f"NFN-based LoRA preset saved to: {out_path}")
        except Exception as e:
            accelerator.print(f"[ERROR] Could not write preset to {out_path}: {e}")

def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    parser.add_argument(
        "--calculate_nfn_weights",
        action="store_true",
        help="Calculate NFN weights for blocks and exit / ブロックのNFN重みを計算して終了",
    )
    parser.add_argument(
        "--nfn_min_lr_weight",
        type=float,
        default=0.1,
        help="Minimum learning rate weight for NFN. Default is 0.1 / NFNの最小学習率重み。デフォルトは0.1",
    )
    parser.add_argument(
        "--nfn_max_lr_weight",
        type=float,
        default=2.0,
        help="Maximum learning rate weight for NFN. Default is 2.0 / NFNの最大学習率重み。デフォルトは2.0",
    )
    parser.add_argument(
        "--nfn_target_mean_lr",
        type=float,
        default=None,
        help="If set, use proportional scaling centered around this target mean LR. Ignores min/max weights. / 設定された場合、この目標平均学習率を中心に比例スケーリングを行います。min/maxの重みは無視されます。",
    )
    parser.add_argument(
        "--nfn_preset_output",
        type=str,
        default=None,
        help="If set, save a LyCoris preset TOML calibrated from NFN scores at this path",
    )
    parser.add_argument(
        "--nfn_eval_scope",
        type=str,
        default="attn-mlp",
        choices=["attn", "attn-mlp", "full", "full-lin"],
        help="Scope of layers to use for NFN score aggregation (attn, attn-mlp, full, full-lin). Default is attn-mlp. / NFNスコア集計に使用するレイヤーの範囲。デフォルトはattn-mlp",
    )
    parser.add_argument(
        "--nfn_batch_limit",
        type=int,
        default=100,
        help="Limit the number of batches used for NFN calculation. Default is 100. / NFN計算に使用するバッチ数を制限します。デフォルトは100です。",
    )
    parser.add_argument("--preset_algo", type=str, default="lora", choices=["lora", "lokr", "loha"])
    parser.add_argument("--preset_lokr_factor", type=int, default=-1)
    parser.add_argument("--preset_base_attn", type=int, default=48)
    parser.add_argument("--preset_base_mlp", type=int, default=40)
    parser.add_argument("--preset_min_rank", type=int, default=24)
    parser.add_argument("--preset_max_rank", type=int, default=96)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = SdxlNetworkTrainer()
    if args.calculate_nfn_weights:
        trainer.calculate_and_show_nfn_scores(args)
    else:
        trainer.train(args)
