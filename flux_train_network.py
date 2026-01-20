import argparse
import copy
import math
import random
from typing import Any, Optional, Union

import torch
from accelerate import Accelerator
from accelerate import init_empty_weights
from transformers import CLIPConfig, CLIPTextModel
from library.safetensors_utils import load_safetensors

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import train_network
from library import (
    flux_models,
    flux_train_utils,
    flux_utils,
    sd3_train_utils,
    strategy_base,
    strategy_flux,
    train_util,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

class FluxNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.is_schnell: Optional[bool] = None
        self.is_swapping_blocks: bool = False
        self.model_type: Optional[str] = None
        self.text_encoders = None
        # One-time forward debug flag to avoid undefined loop variable usage
        self._forward_debug_once: bool = False

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        # sdxl_train_util.verify_sdxl_training_args(args)

        self.model_type = args.model_type  # "flux" or "chroma"
        if self.model_type != "chroma":
            self.use_clip_l = True
        else:
            self.use_clip_l = False  # Chroma does not use CLIP-L
            assert args.apply_t5_attn_mask, "apply_t5_attn_mask must be True for Chroma / Chromaではapply_t5_attn_maskを指定する必要があります"

        if args.fp8_base_unet:
            args.fp8_base = True  # if fp8_base_unet is enabled, fp8_base is also enabled for FLUX.1

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning(
                "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled / cache_text_encoder_outputs_to_diskが有効になっているため、cache_text_encoder_outputsも有効になります"
            )
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        # prepare CLIP-L/T5XXL training flags
        self.train_clip_l = not args.network_train_unet_only and self.use_clip_l
        self.train_t5xxl = False  # default is False even if args.network_train_unet_only is False

        if args.max_token_length is not None:
            logger.warning("max_token_length is not used in Flux training / max_token_lengthはFluxのトレーニングでは使用されません")

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing / blocks_to_swapはcpu_offload_checkpointingと併用できません"

        # DeepSpeed and blocks_to_swap are incompatible
        if args.deepspeed and args.blocks_to_swap is not None and args.blocks_to_swap > 0:
            logger.warning(
                "blocks_to_swap is not supported with DeepSpeed. Setting blocks_to_swap to 0."
                " / blocks_to_swapはDeepSpeedと併用できません。blocks_to_swapを0に設定します。"
            )
            args.blocks_to_swap = 0

        # deprecated split_mode option
        if args.split_mode:
            if args.blocks_to_swap is not None:
                logger.warning(
                    "split_mode is deprecated. Because `--blocks_to_swap` is set, `--split_mode` is ignored."
                    " / split_modeは非推奨です。`--blocks_to_swap`が設定されているため、`--split_mode`は無視されます。"
                )
            else:
                logger.warning(
                    "split_mode is deprecated. Please use `--blocks_to_swap` instead. `--blocks_to_swap 18` is automatically set."
                    " / split_modeは非推奨です。代わりに`--blocks_to_swap`を使用してください。`--blocks_to_swap 18`が自動的に設定されました。"
                )
                args.blocks_to_swap = 18  # 18 is safe for most cases

        train_dataset_group.verify_bucket_reso_steps(32)  # TODO check this
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)  # TODO check this

    def post_accelerator_prepare_hook(self, args, accelerator, training_model):
        # If running under DeepSpeed ZeRO-3, refresh the first-layer linears' weights/bias to avoid matmul shape issues
        is_deepspeed_stage3 = (
            args.deepspeed
            and hasattr(accelerator.state, "deepspeed_plugin")
            and accelerator.state.deepspeed_plugin is not None
            and accelerator.state.deepspeed_plugin.zero_stage == 3
        )

        if is_deepspeed_stage3:
            try:
                import deepspeed
            except Exception:
                deepspeed = None
            if deepspeed is not None:
                try:
                    unwrapped = accelerator.unwrap_model(training_model)
                    # training_model may be a DS wrapper containing 'unet' key or Flux directly
                    flux = None
                    if hasattr(unwrapped, 'models') and isinstance(unwrapped.models, torch.nn.ModuleDict):
                        if 'unet' in unwrapped.models:
                            flux = unwrapped.models['unet']
                        elif 'flux' in unwrapped.models:
                            flux = unwrapped.models['flux']
                    else:
                        flux = unwrapped if isinstance(unwrapped, flux_models.Flux) else None

                    if flux is not None:
                        logger.info("[Z3 Fix] Post-prepare: refreshing first-layer linears for Flux under ZeRO-3")
                        try:
                            inner = accelerator.unwrap_model(flux)
                        except Exception:
                            inner = flux

                        import types
                        def _wrap_forward(name, lin):
                            if lin is None or not hasattr(lin, 'weight'):
                                return
                            if hasattr(lin, '_z3_fix_wrapped') and lin._z3_fix_wrapped:
                                return

                            original_forward = lin.forward

                            def wrapped_forward(self, input):
                                # Perform linear with correct orientation without invoking F.linear (to avoid DS override)
                                w = self.weight
                                b = self.bias
                                in_dim = input.shape[-1]
                                # If weight is (in, out), multiply input @ w; else use w.t()
                                if w.shape[0] == in_dim:
                                    out = input.matmul(w)
                                    used = 'w'
                                else:
                                    out = input.matmul(w.t())
                                    used = 'w.t()'
                                if b is not None:
                                    out = out + b
                                if not hasattr(self, '_z3_fix_logged'):
                                    """ try:
                                        logger.info(
                                            f"[Z3 Fix] {name}: input last-dim={in_dim}, weight={tuple(w.shape)}, weight.t={tuple(w.t().shape)}, using={used}, "
                                            f"bias shape={tuple(b.shape) if b is not None else 'None'}"
                                        )
                                    except Exception:
                                        pass """
                                    self._z3_fix_logged = True
                                return out

                            lin.forward = types.MethodType(wrapped_forward, lin)
                            lin._z3_fix_wrapped = True

                        _wrap_forward('img_in', getattr(inner, 'img_in', None))
                        _wrap_forward('txt_in', getattr(inner, 'txt_in', None))
                        _wrap_forward('pos_embed_input', getattr(inner, 'pos_embed_input', None))

                        # Additionally, wrap all Linear modules to avoid DS zero3 linear orientation issues
                        def _wrap_all_linears(module_root):
                            import types
                            wrapped_count = 0
                            for name, module in module_root.named_modules():
                                if isinstance(module, torch.nn.Linear):
                                    if hasattr(module, '_z3_fix_wrapped') and module._z3_fix_wrapped:
                                        continue
                                    original_forward = module.forward

                                    def make_forward(mod_name):
                                        def wrapped(self, input):
                                            w = self.weight
                                            b = self.bias
                                            in_dim = input.shape[-1]
                                            if w.shape[0] == in_dim:
                                                out = input.matmul(w)
                                                used = 'w'
                                            else:
                                                out = input.matmul(w.t())
                                                used = 'w.t()'
                                            if b is not None:
                                                out = out + b
                                            if not hasattr(self, '_z3_fix_logged'):
                                                """ try:
                                                    logger.info(
                                                        f"[Z3 Fix] linear {mod_name}: input last-dim={in_dim}, weight={tuple(w.shape)}, "
                                                        f"weight.t={tuple(w.t().shape)}, using={used}"
                                                    )
                                                except Exception:
                                                    pass """
                                                self._z3_fix_logged = True
                                            return out
                                        return wrapped

                                    try:
                                        module.forward = types.MethodType(make_forward(name), module)
                                        module._z3_fix_wrapped = True
                                        wrapped_count += 1
                                    except Exception:
                                        pass
                            return wrapped_count

                        try:
                            total_wrapped = _wrap_all_linears(inner)
                            logger.info(f"[Z3 Fix] Wrapped {total_wrapped} Linear modules to enforce correct matmul orientation")
                        except Exception:
                            pass
                except Exception:
                    # Best-effort; continue
                    pass

        is_deepspeed_stage3 = (
            args.deepspeed
            and hasattr(accelerator.state, "deepspeed_plugin")
            and accelerator.state.deepspeed_plugin is not None
            and accelerator.state.deepspeed_plugin.zero_stage == 3
        )

        if is_deepspeed_stage3 and self.train_clip_l and self.text_encoders[0] is not None:
            clip_l = self.text_encoders[0]

            # Get the inner model
            if hasattr(clip_l, "module"):
                clip_l_inner = clip_l.module
            else:
                clip_l_inner = clip_l

            # Fix the configuration at multiple levels to prevent corruption
            #logger.info("Applying comprehensive CLIP-L position embedding fix for DeepSpeed ZeRO-3")

            # Fix all config objects
            if hasattr(clip_l_inner, "config"):
                clip_l_inner.config.max_position_embeddings = 77
            if hasattr(clip_l_inner, "text_model"):
                if hasattr(clip_l_inner.text_model, "config"):
                    clip_l_inner.text_model.config.max_position_embeddings = 77
                if hasattr(clip_l_inner.text_model, "embeddings"):
                    embeddings = clip_l_inner.text_model.embeddings

                    # Ensure position_ids buffer is correct
                    if hasattr(embeddings, "position_ids"):
                        if embeddings.position_ids.shape[-1] != 77:
                            #logger.info(f"Fixing position_ids buffer from {embeddings.position_ids.shape[-1]} to 77")
                            if hasattr(embeddings.position_ids, '_is_param'):
                                delattr(embeddings, 'position_ids')
                            embeddings.register_buffer(
                                "position_ids",
                                torch.arange(77, device=accelerator.device).expand((1, -1)),
                                persistent=False,
                            )

                    # Check if position_embedding has correct shape
                    if hasattr(embeddings, "position_embedding"):
                        current_shape = embeddings.position_embedding.weight.shape
                        #logger.info(f"Position embedding weight shape: {current_shape}")

                        if current_shape[0] == 0:
                            """ logger.warning(
                                f"Position embedding has corrupted shape {current_shape}. "
                                f"Attempting to reconstruct it..."
                            ) """

                            # The position embedding weight has been corrupted to size 0
                            # We need to reconstruct it with the correct size
                            import deepspeed

                            hidden_size = 768  # CLIP-L hidden size
                            target_size = 77

                            # Load the original weights
                            #logger.info(f"Loading original CLIP-L weights from {args.clip_l}")
                            from library.safetensors_utils import load_safetensors
                            original_sd = load_safetensors(args.clip_l, device="cpu", disable_mmap=args.disable_mmap_load_safetensors, dtype=torch.bfloat16)

                            # Get the position embedding weights
                            pos_embed_key = "text_model.embeddings.position_embedding.weight"
                            if pos_embed_key in original_sd:
                                pos_embed_weights = original_sd[pos_embed_key]
                                #logger.info(f"Original position embedding shape: {pos_embed_weights.shape}")

                                # Inject the weights using DeepSpeed's GatheredParameters
                                with deepspeed.zero.GatheredParameters([embeddings.position_embedding.weight], modifier_rank=0):
                                    #logger.info(f"Inside GatheredParameters - is_main_process: {accelerator.is_main_process}")
                                    #logger.info(f"Gathered parameter shape: {embeddings.position_embedding.weight.shape}")

                                    if accelerator.is_main_process:
                                        # The gathered shape is correct, but it will be re-corrupted when we exit
                                        # We need to replace the entire embedding module
                                        #logger.info("Replacing entire position_embedding module to avoid re-corruption")

                                        # Create a new embedding with correct weights
                                        new_pos_embed = torch.nn.Embedding(target_size, hidden_size)
                                        new_pos_embed.weight.data.copy_(pos_embed_weights)
                                        new_pos_embed = new_pos_embed.to(
                                            device=accelerator.device,
                                            dtype=clip_l.dtype
                                        )

                                        # This is the key: replace the module OUTSIDE of DeepSpeed's control
                                        # Store the new module to be applied after exiting GatheredParameters
                                        new_embedding_to_apply = new_pos_embed

                                # Exit GatheredParameters context first
                                # Now apply the new embedding as a BUFFER instead of a parameter
                                # Buffers aren't managed by DeepSpeed ZeRO-3 the same way
                                try:
                                    if accelerator.is_main_process:
                                        #logger.info("Converting position_embedding to use buffer instead of parameter")

                                        # Replace the Embedding layer with a custom implementation that uses a buffer
                                        class BufferEmbedding(torch.nn.Module):
                                            def __init__(self, num_embeddings, embedding_dim, weight_data):
                                                super().__init__()
                                                self.num_embeddings = num_embeddings
                                                self.embedding_dim = embedding_dim
                                                # Register as buffer, not parameter
                                                self.register_buffer('weight', weight_data)

                                            def forward(self, input):
                                                return torch.nn.functional.embedding(input, self.weight)

                                        # Create the buffer-based embedding
                                        buffer_pos_embed = BufferEmbedding(
                                            target_size,
                                            hidden_size,
                                            pos_embed_weights.to(device=accelerator.device, dtype=clip_l.dtype)
                                        )

                                        # Replace the position_embedding
                                        embeddings.position_embedding = buffer_pos_embed

                                        #logger.info(f"Position embedding converted to buffer: shape = {embeddings.position_embedding.weight.shape}")

                                except Exception as e:
                                    logger.error(f"Failed to convert position_embedding to buffer: {e}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                                    raise RuntimeError(
                                        "Failed to fix CLIP-L position embeddings. "
                                        "Please try --zero_stage 2 or contact the developers."
                                    )

                                # Synchronize across processes
                                if torch.distributed.is_initialized():
                                    torch.distributed.barrier()

                                # Verify the final shape
                                final_shape = embeddings.position_embedding.weight.shape
                                #logger.info(f"Final position embedding shape after conversion: {final_shape}")

                                del original_sd
                            else:
                                logger.error(f"Could not find {pos_embed_key} in checkpoint")
                                raise RuntimeError("Failed to load position embedding weights")

                    # Monkey-patch the embeddings forward to ignore config corruption
                    original_forward = embeddings.forward

                    def fixed_forward(
                        self,
                        input_ids = None,
                        position_ids = None,
                        inputs_embeds = None,
                    ):
                        # Temporarily fix config max_position_embeddings if it's corrupted
                        old_max_pos = None
                        if hasattr(self, 'config') and hasattr(self.config, 'max_position_embeddings'):
                            old_max_pos = self.config.max_position_embeddings
                            if old_max_pos != 77:
                                self.config.max_position_embeddings = 77

                        # Also check if embeddings has max_position_embeddings directly
                        old_direct_max_pos = None
                        if hasattr(self, 'max_position_embeddings'):
                            old_direct_max_pos = self.max_position_embeddings
                            self.max_position_embeddings = 77

                        # Call original forward
                        try:
                            result = original_forward(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
                        finally:
                            # Restore old values
                            if old_max_pos is not None and hasattr(self, 'config'):
                                self.config.max_position_embeddings = old_max_pos
                            if old_direct_max_pos is not None:
                                self.max_position_embeddings = old_direct_max_pos

                        return result

                    # Bind the method to the embeddings instance
                    import types
                    embeddings.forward = types.MethodType(fixed_forward, embeddings)

                    #logger.info("Applied forward monkey-patch to CLIP-L embeddings")

            max_pos = getattr(clip_l_inner.config, 'max_position_embeddings', 'N/A')
            #logger.info(f"CLIP-L max_position_embeddings after fix: {max_pos}")

    def load_target_model(self, args, weight_dtype, accelerator):
        # currently offload to cpu for some models

        # Check if DeepSpeed ZeRO Stage 3 is enabled
        is_deepspeed_stage3 = (
            args.deepspeed
            and hasattr(accelerator.state, "deepspeed_plugin")
            and accelerator.state.deepspeed_plugin is not None
            and accelerator.state.deepspeed_plugin.zero_stage == 3
        )

        # if the file is fp8 and we are using fp8_base, we can load it as is (fp8)
        loading_dtype = None if args.fp8_base else weight_dtype

        # For DeepSpeed Stage 3, create hollow model and inject weights
        if is_deepspeed_stage3:
            import deepspeed
            from safetensors.torch import load_file
            from tqdm import tqdm

            logger.info("DeepSpeed Stage 3 detected. Creating hollow Flux model and injecting weights chunk-by-chunk.")
            ds_config = accelerator.state.deepspeed_plugin.deepspeed_config

            # Create hollow model with DeepSpeed ZeRO Init context
            with deepspeed.zero.Init(config_dict_or_path=ds_config):
                is_schnell, model = flux_utils.load_flow_model(
                    args.pretrained_model_name_or_path,
                    loading_dtype,
                    "cpu",
                    disable_mmap=args.disable_mmap_load_safetensors,
                    model_type=self.model_type,
                    load_weights=False,  # Create hollow model
                )
            logger.info("Hollow Flux model created.")

            # Load original weights from checkpoint
            logger.info(f"Loading original weights from: {args.pretrained_model_name_or_path} into CPU RAM...")

            if self.model_type != "chroma":
                # Analyze checkpoint to get paths and format
                is_diffusers, _, (num_double_blocks, num_single_blocks), ckpt_paths = flux_utils.analyze_checkpoint_state(
                    args.pretrained_model_name_or_path
                )

                # Load weights from all checkpoint files
                from library.safetensors_utils import load_safetensors
                original_sd = {}
                for ckpt_path in ckpt_paths:
                    original_sd.update(load_safetensors(ckpt_path, device="cpu", disable_mmap=args.disable_mmap_load_safetensors, dtype=loading_dtype))

                # Convert Diffusers to BFL if needed
                if is_diffusers:
                    logger.info("Converting Diffusers format to BFL format")
                    original_sd = flux_utils.convert_diffusers_sd_to_bfl(original_sd, num_double_blocks, num_single_blocks)
            else:
                # For Chroma, just load the single file directly
                from library.safetensors_utils import load_safetensors
                original_sd = load_safetensors(args.pretrained_model_name_or_path, device="cpu", disable_mmap=args.disable_mmap_load_safetensors, dtype=loading_dtype)

            # Remove annoying prefix if present
            for key in list(original_sd.keys()):
                new_key = key.replace("model.diffusion_model.", "")
                if new_key == key:
                    break  # the model doesn't have annoying prefix
                original_sd[new_key] = original_sd.pop(key)

            # Inject weights chunk-by-chunk
            logger.info("Injecting original weights into the sharded model...")
            all_params = list(model.parameters())
            chunk_size = 8
            for i in tqdm(
                range(0, len(all_params), chunk_size),
                desc="Injecting weight chunks",
                disable=not accelerator.is_main_process,
            ):
                chunk = all_params[i : i + chunk_size]
                with deepspeed.zero.GatheredParameters(chunk, modifier_rank=0):
                    if accelerator.is_main_process:
                        for j, param in enumerate(chunk):
                            # Find the parameter's original name
                            param_name = [name for name, p in model.named_parameters() if p is all_params[i + j]][0]
                            if param_name in original_sd:
                                param.data.copy_(original_sd[param_name].to(param.device, param.dtype))

            logger.info("Weight injection complete.")
            # Global fix: correct orientation across all Linear modules after injection
            try:
                import deepspeed
                def _fix_all_linear_orientations(m):
                    corrected = 0
                    for name, module in m.named_modules():
                        if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight') and module.weight is not None:
                            params = [module.weight] + ([module.bias] if module.bias is not None else [])
                            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                                w = module.weight.data
                                in_feat = getattr(module, 'in_features', None)
                                out_feat = getattr(module, 'out_features', None)
                                if in_feat is not None and out_feat is not None and w.shape == (in_feat, out_feat):
                                    with torch.no_grad():
                                        module.weight.data.copy_(w.t().contiguous())
                                    corrected += 1
                                elif not w.is_contiguous():
                                    with torch.no_grad():
                                        module.weight.data.copy_(w.contiguous())
                    return corrected
                num_fixed = _fix_all_linear_orientations(model)
                logger.info(f"Fixed orientation for {num_fixed} Linear modules after weight injection.")
            except Exception as e:
                logger.warning(f"Global Linear orientation fix skipped: {e}")
            del original_sd
        else:
            # Normal loading path for non-DeepSpeed or DeepSpeed Stage 1/2
            # if we load to cpu, flux.to(fp8) takes a long time, so we should load to gpu in future
            _, model = flux_utils.load_flow_model(
                args.pretrained_model_name_or_path,
                loading_dtype,
                "cpu",
                disable_mmap=args.disable_mmap_load_safetensors,
                model_type=self.model_type,
            )

        if args.fp8_base:
            # check dtype of model
            if model.dtype == torch.float8_e4m3fnuz or model.dtype == torch.float8_e5m2 or model.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {model.dtype}")
            elif model.dtype == torch.float8_e4m3fn:
                logger.info("Loaded fp8 FLUX model")
            else:
                logger.info(
                    "Cast FLUX model to fp8. This may take a while. You can reduce the time by using fp8 checkpoint."
                    " / FLUXモデルをfp8に変換しています。これには時間がかかる場合があります。fp8チェックポイントを使用することで時間を短縮できます。"
                )
                model.to(torch.float8_e4m3fn)

        # if args.split_mode:
        #     model = self.prepare_split_model(model, weight_dtype, accelerator)

        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            # Swap blocks between CPU and GPU to reduce memory usage, in forward and backward passes.
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        # Load VAE (always needed)
        ae = flux_utils.load_ae(args.ae, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)

        # Load Text Encoders
        clip_l, t5xxl = None, None

        # Load CLIP-L if it's used and either not caching or being trained.
        if self.use_clip_l:
            if not args.cache_text_encoder_outputs or self.train_clip_l:
                # For DeepSpeed ZeRO-3, load normally but don't use DeepSpeed Init
                # because it will be wrapped later by prepare_deepspeed_model + accelerator.prepare
                clip_l = flux_utils.load_clip_l(
                    args.clip_l, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors
                )
                clip_l.eval()

                # Pre-emptively fix config before DeepSpeed wraps it
                if is_deepspeed_stage3:
                    logger.info("Pre-fixing CLIP-L config for DeepSpeed ZeRO-3")
                    if hasattr(clip_l, "config"):
                        clip_l.config.max_position_embeddings = 77
                    if hasattr(clip_l, "text_model") and hasattr(clip_l.text_model, "config"):
                        clip_l.text_model.config.max_position_embeddings = 77
            else:
                logger.info("Using cached outputs for CLIP-L, so it will not be loaded.")
        else:
            clip_l = flux_utils.dummy_clip_l()  # For Chroma

        # Load T5XXL if not caching or if it's being trained.
        if not args.cache_text_encoder_outputs or self.train_t5xxl:
            # if the file is fp8 and we are using fp8_base (not unet), we can load it as is (fp8)
            if args.fp8_base and not args.fp8_base_unet:
                loading_dtype = None  # as is
            else:
                loading_dtype = weight_dtype

            t5xxl = flux_utils.load_t5xxl(args.t5xxl, loading_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
            t5xxl.eval()

            if args.fp8_base and not args.fp8_base_unet:
                # check dtype of model
                if (
                    t5xxl.dtype == torch.float8_e4m3fnuz
                    or t5xxl.dtype == torch.float8_e5m2
                    or t5xxl.dtype == torch.float8_e5m2fnuz
                ):
                    raise ValueError(f"Unsupported fp8 model dtype: {t5xxl.dtype}")
                elif t5xxl.dtype == torch.float8_e4m3fn:
                    logger.info("Loaded fp8 T5XXL model")
        else:
            logger.info("Using cached outputs for T5XXL, so it will not be loaded.")

        model_version = flux_utils.MODEL_VERSION_FLUX_V1 if self.model_type != "chroma" else flux_utils.MODEL_VERSION_CHROMA
        self.text_encoders = [clip_l, t5xxl]
        return model_version, [clip_l, t5xxl], ae, model

    def get_tokenize_strategy(self, args):
        # This method is called before `assert_extra_args`, so we cannot use `self.is_schnell` here.
        # Instead, we analyze the checkpoint state to determine if it is schnell.
        if args.model_type != "chroma":
            _, is_schnell, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)
        else:
            is_schnell = False
        self.is_schnell = is_schnell

        if args.t5xxl_max_token_length is None:
            if self.is_schnell:
                t5xxl_max_token_length = 256
            else:
                t5xxl_max_token_length = 512
        else:
            t5xxl_max_token_length = args.t5xxl_max_token_length

        logger.info(f"t5xxl_max_token_length: {t5xxl_max_token_length}")
        return strategy_flux.FluxTokenizeStrategy(t5xxl_max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_flux.FluxTokenizeStrategy):
        return [tokenize_strategy.clip_l, tokenize_strategy.t5xxl]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_flux.FluxLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, False)
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_flux.FluxTextEncodingStrategy(apply_t5_attn_mask=args.apply_t5_attn_mask)

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        # check t5xxl is trained or not
        self.train_t5xxl = network.train_t5xxl

        if self.train_t5xxl and args.cache_text_encoder_outputs:
            raise ValueError(
                "T5XXL is trained, so cache_text_encoder_outputs cannot be used / T5XXL学習時はcache_text_encoder_outputsは使用できません"
            )

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            if self.train_clip_l and not self.train_t5xxl:
                return text_encoders[0:1]  # only CLIP-L is needed for encoding because T5XXL is cached
            else:
                return None  # no text encoders are needed for encoding because both are cached
        else:
            return text_encoders  # both CLIP-L and T5XXL are needed for encoding

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [self.train_clip_l, self.train_t5xxl]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            # is_partial is True if we are not using CLIP-L (Chroma) or if we are training one of the text encoders.
            is_partial = not self.use_clip_l or self.train_clip_l or self.train_t5xxl
            return strategy_flux.FluxTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
                is_partial=is_partial,
                apply_t5_attn_mask=args.apply_t5_attn_mask,
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            logger.info("Text encoder output caching is enabled.")

            # IMPORTANT: Even if we're using pre-cached outputs, we MUST call this method
            # because it validates existing caches and registers the NPZ file paths in image_info
            # If T5XXL is None (not loaded), only CLIP-L will be used for encoding during validation
            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            logger.info("Cache validation complete. NPZ file paths have been registered.")

            # Move text encoders to GPU if they need to be trained
            if self.train_clip_l and text_encoders[0] is not None:
                logger.info("Moving CLIP-L to GPU for training.")
                text_encoders[0].to(accelerator.device, dtype=weight_dtype)

            if self.train_t5xxl and text_encoders[1] is not None:
                logger.info("Moving T5-XXL to GPU for training.")
                text_encoders[1].to(accelerator.device)  # dtype is handled by fp8 or weight_dtype in load_target_model

            # The original code caches sample prompts here.
            # This is not possible if T5-XXL is not loaded (which is the case when it's cached and not trained).
            if args.sample_prompts is not None:
                if text_encoders[1] is None:
                    logger.warning(
                        "Sample prompt generation is disabled because T5-XXL is not loaded (due to cached outputs). "
                        + "To enable sampling, ensure T5-XXL is loaded or do not use pre-cached outputs."
                    )
                    self.sample_prompts_te_outputs = None
                    args.sample_prompts = None  # To prevent sample_images from running
                else:
                    # This path is not expected for the user's request, but keeping it for robustness.
                    # It would run if user caches only CLIP-L, which is not the primary scenario.
                    logger.info(f"Cache Text Encoder outputs for sample prompt: {args.sample_prompts}")

                    tokenize_strategy: strategy_flux.FluxTokenizeStrategy = strategy_base.TokenizeStrategy.get_strategy()
                    text_encoding_strategy: strategy_flux.FluxTextEncodingStrategy = (
                        strategy_base.TextEncodingStrategy.get_strategy()
                    )

                    prompts = train_util.load_prompts(args.sample_prompts)
                    sample_prompts_te_outputs = {}  # key: prompt, value: text encoder outputs
                    with accelerator.autocast(), torch.no_grad():
                        for prompt_dict in prompts:
                            for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                                if p not in sample_prompts_te_outputs:
                                    logger.info(f"Cache Text Encoder outputs for prompt: {p}")
                                    tokens_and_masks = tokenize_strategy.tokenize(p)
                                    sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                        tokenize_strategy, text_encoders, tokens_and_masks, args.apply_t5_attn_mask
                                    )
                    self.sample_prompts_te_outputs = sample_prompts_te_outputs

        else:
            # Original logic for non-cached scenario
            logger.info("Caching is disabled. Moving text encoders to GPU for on-the-fly encoding.")
            if text_encoders[0] is not None:
                text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            if text_encoders[1] is not None:
                text_encoders[1].to(accelerator.device)

    def sample_images(self, accelerator, args, epoch, global_step, device, ae, tokenizer, text_encoder, flux):
        text_encoders = text_encoder  # for compatibility
        text_encoders = self.get_models_for_text_encoding(args, accelerator, text_encoders)

        flux_train_utils.sample_images(
            accelerator, args, epoch, global_step, flux, ae, text_encoders, self.sample_prompts_te_outputs
        )

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        return vae.encode(images)

    def shift_scale_latents(self, args, latents):
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet: flux_models.Flux,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )

        # pack latents and get img_ids
        packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
        packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
        img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=accelerator.device)

        # get guidance
        # ensure guidance_scale in args is float
        guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device, dtype=weight_dtype)

        # get modulation vectors for Chroma
        with accelerator.autocast(), torch.no_grad():
            mod_vectors = unet.get_mod_vectors(timesteps=timesteps / 1000, guidance=guidance_vec, batch_size=bsz)

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            # In the partial cache scenario, only the first element (l_pooled) is being trained and requires grad.
            if text_encoder_conds[0] is not None and text_encoder_conds[0].dtype.is_floating_point:
                text_encoder_conds[0].requires_grad_(True)

            img_ids.requires_grad_(True)
            guidance_vec.requires_grad_(True)
            if mod_vectors is not None:
                mod_vectors.requires_grad_(True)

        # Predict the noise residual
        # Debug: Check text_encoder_conds structure
        if text_encoder_conds is None:
            raise ValueError("text_encoder_conds is None - cached text encoder outputs may not be loaded correctly")
        if len(text_encoder_conds) != 4:
            raise ValueError(f"text_encoder_conds should have 4 elements, got {len(text_encoder_conds)}")

        l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoder_conds

        # Validate that critical components are not None
        if self.use_clip_l:
            if l_pooled is None:
                raise ValueError("l_pooled is None - CLIP-L pooled output not loaded from cache")
        if t5_out is None:
            raise ValueError(
                "t5_out is None - T5 encoder output not loaded from cache. "
                "This usually means:\n"
                "  1. The NPZ cache files are missing or corrupted\n"
                "  2. The cache files were created incorrectly\n"
                "  3. The none_or_stack_elements function returned None\n"
                "Please verify your cache files exist and re-cache if needed."
            )
        if txt_ids is None:
            raise ValueError("txt_ids is None - text IDs not loaded from cache")

        if not args.apply_t5_attn_mask:
            t5_attn_mask = None

        def call_dit(img, img_ids, t5_out, txt_ids, l_pooled, timesteps, guidance_vec, t5_attn_mask, mod_vectors):
            # grad is enabled even if unet is not in train mode, because Text Encoder is in train mode
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                # Debug shapes before first projection (log once)
                if not self._forward_debug_once:
                    """ try:
                        logger.info(
                            f"[Forward Debug] img={tuple(img.shape)}, img_ids={tuple(img_ids.shape)}, "
                            f"t5_out={tuple(t5_out.shape) if t5_out is not None else 'None'}, txt_ids={tuple(txt_ids.shape)}, "
                            f"l_pooled={tuple(l_pooled.shape) if l_pooled is not None else 'None'}"
                        )
                    except Exception:
                        pass """
                    self._forward_debug_once = True
                model_pred = unet(
                    img=img,
                    img_ids=img_ids,
                    txt=t5_out,
                    txt_ids=txt_ids,
                    y=l_pooled,
                    timesteps=timesteps / 1000,
                    guidance=guidance_vec,
                    txt_attention_mask=t5_attn_mask,
                    mod_vectors=mod_vectors,
                )
            return model_pred

        model_pred = call_dit(
            img=packed_noisy_model_input,
            img_ids=img_ids,
            t5_out=t5_out,
            txt_ids=txt_ids,
            l_pooled=l_pooled,
            timesteps=timesteps,
            guidance_vec=guidance_vec,
            t5_attn_mask=t5_attn_mask,
            mod_vectors=mod_vectors,
        )

        # unpack latents
        model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

        # apply model prediction type
        model_pred, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)

        # flow matching loss: this is different from SD3
        target = noise - latents

        # differential output preservation
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                unet.prepare_block_swap_before_forward()
                with torch.no_grad():
                    model_pred_prior = call_dit(
                        img=packed_noisy_model_input[diff_output_pr_indices],
                        img_ids=img_ids[diff_output_pr_indices],
                        t5_out=t5_out[diff_output_pr_indices],
                        txt_ids=txt_ids[diff_output_pr_indices],
                        l_pooled=l_pooled[diff_output_pr_indices],
                        timesteps=timesteps[diff_output_pr_indices],
                        guidance_vec=guidance_vec[diff_output_pr_indices] if guidance_vec is not None else None,
                        t5_attn_mask=t5_attn_mask[diff_output_pr_indices] if t5_attn_mask is not None else None,
                        mod_vectors=mod_vectors[diff_output_pr_indices] if mod_vectors is not None else None,
                    )
                network.set_multiplier(1.0)  # may be overwritten by "network_multipliers" in the next step

                model_pred_prior = flux_utils.unpack_latents(model_pred_prior, packed_latent_height, packed_latent_width)
                model_pred_prior, _ = flux_train_utils.apply_model_prediction_type(
                    args,
                    model_pred_prior,
                    noisy_model_input[diff_output_pr_indices],
                    sigmas[diff_output_pr_indices] if sigmas is not None else None,
                )
                target[diff_output_pr_indices] = model_pred_prior.to(target.dtype)

        return model_pred, target, timesteps, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        if self.model_type != "chroma":
            model_description = "schnell" if self.is_schnell else "dev"
        else:
            model_description = "chroma"
        return train_util.get_sai_model_spec(None, args, False, True, False, flux=model_description)

    def update_metadata(self, metadata, args):
        metadata["ss_model_type"] = args.model_type
        metadata["ss_apply_t5_attn_mask"] = args.apply_t5_attn_mask
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_guidance_scale"] = args.guidance_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_model_prediction_type"] = args.model_prediction_type
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        if index == 0:  # CLIP-L
            return super().prepare_text_encoder_grad_ckpt_workaround(index, text_encoder)
        else:  # T5XXL
            text_encoder.encoder.embed_tokens.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        if index == 0:  # CLIP-L
            logger.info(f"prepare CLIP-L for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}")
            text_encoder.to(te_weight_dtype)  # fp8
            text_encoder.text_model.embeddings.to(dtype=weight_dtype)
        else:  # T5XXL

            def prepare_fp8(text_encoder, target_dtype):
                def forward_hook(module):
                    def forward(hidden_states):
                        hidden_gelu = module.act(module.wi_0(hidden_states))
                        hidden_linear = module.wi_1(hidden_states)
                        hidden_states = hidden_gelu * hidden_linear
                        hidden_states = module.dropout(hidden_states)

                        hidden_states = module.wo(hidden_states)
                        return hidden_states

                    return forward

                for module in text_encoder.modules():
                    if module.__class__.__name__ in ["T5LayerNorm", "Embedding"]:
                        # print("set", module.__class__.__name__, "to", target_dtype)
                        module.to(target_dtype)
                    if module.__class__.__name__ in ["T5DenseGatedActDense"]:
                        # print("set", module.__class__.__name__, "hooks")
                        module.forward = forward_hook(module)

            if flux_utils.get_t5xxl_actual_dtype(text_encoder) == torch.float8_e4m3fn and text_encoder.dtype == weight_dtype:
                logger.info(f"T5XXL already prepared for fp8")
            else:
                logger.info(f"prepare T5XXL for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}, add hooks")
                text_encoder.to(te_weight_dtype)  # fp8
                prepare_fp8(text_encoder, weight_dtype)

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        # Always prepare Flux via accelerator regardless of block swapping
        flux: flux_models.Flux = unet
        flux = accelerator.prepare(flux, device_placement=[not self.is_swapping_blocks])
        # Post-prepare fix: ensure first linear weights are contiguous and correctly shaped for DeepSpeed ZeRO-3
        def _fix_deepspeed_zero3_linears(model):
            try:
                import deepspeed
            except Exception:
                return
            # Only apply under ZeRO-3
            is_z3 = (
                args.deepspeed
                and hasattr(accelerator.state, "deepspeed_plugin")
                and accelerator.state.deepspeed_plugin is not None
                and accelerator.state.deepspeed_plugin.zero_stage == 3
            )
            if not is_z3:
                return

            # Unwrap the model to access actual module
            try:
                inner = accelerator.unwrap_model(model)
            except Exception:
                inner = model

            logger.info("[Z3 Fix] Scanning first-layer linears to refresh shapes/contiguity under ZeRO-3")

            targets = []
            if hasattr(inner, "img_in"):
                targets.append(("img_in", inner.img_in))
            if hasattr(inner, "txt_in"):
                targets.append(("txt_in", inner.txt_in))
            if hasattr(inner, "pos_embed_input"):
                targets.append(("pos_embed_input", inner.pos_embed_input))

            for name, lin in targets:
                # Gather parameters when modifying under ZeRO-3
                params = [p for p in lin.parameters()]
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if accelerator.is_main_process:
                        if hasattr(lin, "weight") and lin.weight is not None:
                            w = lin.weight.data
                            in_feat = getattr(lin, "in_features", None)
                            out_feat = getattr(lin, "out_features", None)
                            # Detect transposed storage: expected (out,in). If we see (in,out), transpose it back.
                            if in_feat is not None and out_feat is not None and w.shape == (in_feat, out_feat):
                                w_fixed = w.t().contiguous().clone().to(w.dtype)
                            else:
                                w_fixed = (w.contiguous() if not w.is_contiguous() else w).clone().to(w.dtype)
                            with torch.no_grad():
                                lin.weight.data.copy_(w_fixed)
                        if hasattr(lin, "bias") and lin.bias is not None:
                            b = lin.bias.data
                            b_fixed = (b.contiguous() if not b.is_contiguous() else b).clone().to(b.dtype)
                            with torch.no_grad():
                                lin.bias.data.copy_(b_fixed)
                    # Broadcast updated params
                    accelerator.wait_for_everyone()

                # Debug prints for shapes
                """ try:
                    in_features = lin.in_features if hasattr(lin, "in_features") else None
                    out_features = lin.out_features if hasattr(lin, "out_features") else None
                    logger.info(
                        f"[Z3 Fix] {name}: weight shape={tuple(lin.weight.shape) if hasattr(lin,'weight') else 'N/A'}, "
                        f"bias shape={tuple(lin.bias.shape) if hasattr(lin,'bias') and lin.bias is not None else 'None'}, "
                        f"in_features={in_features}, out_features={out_features}"
                    )
                except Exception as e:
                    logger.warning(f"[Z3 Fix] Failed to print shapes for {name}: {e}") """

        _fix_deepspeed_zero3_linears(flux)
        if self.is_swapping_blocks:
            accelerator.unwrap_model(flux).move_to_device_except_swap_blocks(accelerator.device)
            accelerator.unwrap_model(flux).prepare_block_swap_before_forward()
        return flux

    def generate_recommended_weights(self, sorted_results, args, accelerator):
        """Generate recommended learning rate weights based on NFN scores"""
        accelerator.print("\n" + "="*60)
        accelerator.print("Recommended --network_block_lr_weights")
        accelerator.print("="*60)

        valid_scores = [v['nfn'] for k, v in sorted_results.items() if v.get('module_count', 0) > 0 and k != "other"]
        if not valid_scores:
            accelerator.print("No valid NFN scores found. Cannot generate weights.")
            return

        min_score, max_score = min(valid_scores), max(valid_scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        accelerator.print(f"NFN Score Range: Min={min_score:.4f}, Max={max_score:.4f}")
        accelerator.print(f"Weighting Range: Min LR Weight={args.nfn_min_lr_weight}, Max LR Weight={args.nfn_max_lr_weight}")

        def nfn_to_weight(nfn_score, module_count):
            if module_count == 0: return 0.0
            inverted_score = 1.0 - ((nfn_score - min_score) / score_range)
            weight_range = args.nfn_max_lr_weight - args.nfn_min_lr_weight
            return (inverted_score * weight_range) + args.nfn_min_lr_weight

        weights = []
        # Ensure blocks are sorted: double_blocks first, then single_blocks
        def sort_key(x):
            if x == "other":
                return (2, float('inf'))
            elif x.startswith('double_block_'):
                return (0, int(x.split('_')[-1]))
            elif x.startswith('single_block_'):
                return (1, int(x.split('_')[-1]))
            else:
                return (2, float('inf'))

        sorted_keys = sorted(sorted_results.keys(), key=sort_key)

        for block_name in sorted_keys:
            if block_name == "other": continue
            data = sorted_results[block_name]
            weight = nfn_to_weight(data.get('nfn', 0.0), data.get('module_count', 0))
            weights.append(f"{weight:.4f}")

        accelerator.print("\nCopy and paste these into your training arguments:")
        accelerator.print(f'--network_args "block_lr_weights={",".join(weights)}"')
        accelerator.print(f"\nTotal blocks: {len(weights)} (double_blocks + single_blocks)")
        accelerator.print("\n" + "="*60 + "\n")

    def calculate_and_show_nfn_scores(self, args):
        """Calculate and display NFN (Neural Feature Norm) scores for Flux blocks"""
        from collections import defaultdict
        import re
        from tqdm import tqdm
        from library import metrics as metrics_utils

        # Set model_type if not specified
        if not hasattr(args, 'model_type') or args.model_type is None:
            args.model_type = "flux"  # default to flux
        self.model_type = args.model_type

        # Set use_clip_l based on model_type
        if self.model_type != "chroma":
            self.use_clip_l = True
        else:
            self.use_clip_l = False

        # For NFN calculation, disable DeepSpeed to avoid text encoder loading issues
        # Use block swapping instead for memory efficiency
        original_deepspeed = args.deepspeed
        args.deepspeed = False

        # Ensure blocks_to_swap is set for memory efficiency
        if args.blocks_to_swap is None or args.blocks_to_swap == 0:
            logger.info("Setting blocks_to_swap=35 for NFN calculation to reduce memory usage")
            args.blocks_to_swap = 35

        # Prepare accelerator without DeepSpeed
        logger.info("preparing accelerator for NFN calculation (DeepSpeed disabled, using block swap)")
        accelerator = train_util.prepare_accelerator(args)

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

        # Set strategies for dataset
        train_dataset_group.set_current_strategies()

        # Prepare dataloader
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
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # Load models (use fp16 for NFN to lower memory)
        weight_dtype = torch.float16

        # Load Flux model
        model_version, text_encoder, vae, flux = self.load_target_model(args, weight_dtype, accelerator)
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        # Convert model to fp16 for NFN calculation
        flux.to(dtype=torch.float16)
        flux.requires_grad_(False)
        flux.eval()

        # Use block swapping for memory efficiency (DeepSpeed is disabled for NFN)
        if args.blocks_to_swap and args.blocks_to_swap > 0:
            logger.info(f"enable swap {args.blocks_to_swap} blocks for NFN calculation")
            flux.enable_block_swap(args.blocks_to_swap, accelerator.device)
            flux.move_to_device_except_swap_blocks(accelerator.device)
            flux.prepare_block_swap_before_forward()
        else:
            flux.to(device=accelerator.device)

        # Prepare noise scheduler
        noise_scheduler = self.get_noise_scheduler(args, accelerator.device)

        # Start NFN calculation
        mtype = "Flux" if self.model_type != "chroma" else "Chroma"
        accelerator.print("\n" + "="*80)
        accelerator.print(f"Starting NFN Score Calculation for {mtype}")
        accelerator.print("="*80)

        all_metrics_from_all_batches = []
        metrics_for_single_batch = defaultdict(dict)
        hooks = metrics_utils.attach_nfn_hooks(flux, metrics_for_single_batch)

        if not hooks:
            accelerator.print("Could not attach any hooks. Aborting NFN analysis.")
            return

        accelerator.print(f"[INFO] Attached {len(hooks)} hooks to Flux modules.")

        num_batches_to_process = len(train_dataloader)
        accelerator.print(f"Analyzing {num_batches_to_process} batches from your dataset...")

        progress_bar = tqdm(
            range(num_batches_to_process),
            smoothing=0,
            disable=not accelerator.is_local_main_process,
            desc="NFN Analysis",
        )
        data_iter = iter(train_dataloader)

        for i in range(num_batches_to_process):
            try:
                batch = next(data_iter)
            except StopIteration:
                accelerator.print(f"Dataset exhausted after {i} batches.")
                break

            with torch.no_grad():
                # Get latents
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device)
                else:
                    # Encode images if latents not cached
                    vae.to(accelerator.device, dtype=weight_dtype)
                    # Convert images to VAE dtype (fp16)
                    images = batch["images"].to(accelerator.device, dtype=weight_dtype)
                    latents = self.encode_images_to_latents(args, vae, images)
                    vae.to("cpu")

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
                    args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
                )

                # Pack latents
                packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)
                packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
                img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=accelerator.device)

                # Get text encoder outputs (dummy for NFN) - use shorter seq len to reduce memory
                seq_len = 128
                l_pooled = torch.randn(bsz, 768, device=accelerator.device, dtype=weight_dtype)
                t5_out = torch.randn(bsz, seq_len, 4096, device=accelerator.device, dtype=weight_dtype)
                txt_ids = torch.zeros(bsz, seq_len, 3, device=accelerator.device, dtype=weight_dtype)
                txt_attention_mask = torch.ones(bsz, seq_len, device=accelerator.device, dtype=weight_dtype)
                guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device, dtype=weight_dtype)

                # Forward pass to trigger hooks
                # Force fp16 autocast regardless of global mixed precision to reduce memory
                from contextlib import nullcontext
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if accelerator.device.type == "cuda"
                    else nullcontext()
                )
                with autocast_ctx:
                    mod_vectors = flux.get_mod_vectors(timesteps=timesteps / 1000, guidance=guidance_vec, batch_size=bsz)
                    _ = flux(
                        img=packed_noisy_model_input,
                        img_ids=img_ids,
                        txt=t5_out,
                        txt_ids=txt_ids,
                        y=l_pooled,
                        timesteps=timesteps / 1000,
                        guidance=guidance_vec,
                        txt_attention_mask=txt_attention_mask,
                        mod_vectors=mod_vectors,
                    )

                # Prepare blocks for next forward pass if using block swap
                if args.blocks_to_swap and args.blocks_to_swap > 0:
                    flux.prepare_block_swap_before_forward()

                all_metrics_from_all_batches.append(dict(metrics_for_single_batch))
                metrics_for_single_batch.clear()

            progress_bar.update(1)

        metrics_utils.remove_nfn_hooks(hooks)
        progress_bar.close()

        if not all_metrics_from_all_batches:
            accelerator.print("No metrics were calculated.")
            return

        avg_metrics = metrics_utils.average_metrics(all_metrics_from_all_batches)

        # Extract block names for Flux (double_blocks and single_blocks)
        def get_block_name(module_name):
            if "double_blocks" in module_name:
                match = re.search(r"double_blocks\.(\d+)\.", module_name)
                if match:
                    return f"double_block_{int(match.group(1)):02d}"
            elif "single_blocks" in module_name:
                match = re.search(r"single_blocks\.(\d+)\.", module_name)
                if match:
                    return f"single_block_{int(match.group(1)):02d}"
            return "other"

        # Get expected blocks from Flux model
        num_double_blocks = len(flux.double_blocks) if hasattr(flux, 'double_blocks') else 0
        num_single_blocks = len(flux.single_blocks) if hasattr(flux, 'single_blocks') else 0

        expected_blocks = (
            [f"double_block_{i:02d}" for i in range(num_double_blocks)] +
            [f"single_block_{i:02d}" for i in range(num_single_blocks)] +
            ["other"]
        )
        block_scores = {block: {'nfn_sum': 0.0, 'count': 0} for block in expected_blocks}

        for module_name, values in avg_metrics.items():
            block_name = get_block_name(module_name)
            if block_name in block_scores:
                block_scores[block_name]['nfn_sum'] += values.get('nfn', 1.0)
                block_scores[block_name]['count'] += 1

        final_results = {
            block: {
                'nfn': data['nfn_sum'] / data['count'] if data['count'] > 0 else 0.0,
                'module_count': data['count']
            }
            for block, data in block_scores.items()
        }

        sorted_results = dict(sorted(final_results.items()))

        # Generate recommended weights
        self.generate_recommended_weights(sorted_results, args, accelerator)

        # Restore original DeepSpeed setting (for completeness, though NFN exits after this)
        args.deepspeed = original_deepspeed


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)

    parser.add_argument(
        "--split_mode",
        action="store_true",
        # help="[EXPERIMENTAL] use split mode for Flux model, network arg `train_blocks=single` is required"
        # + "/[実験的] Fluxモデルの分割モードを使用する。ネットワーク引数`train_blocks=single`が必要",
        help="[Deprecated] This option is deprecated. Please use `--blocks_to_swap` instead."
        " / このオプションは非推奨です。代わりに`--blocks_to_swap`を使用してください。",
    )

    # NFN arguments
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

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = FluxNetworkTrainer()

    # Check if NFN calculation is requested
    if args.calculate_nfn_weights:
        trainer.calculate_and_show_nfn_scores(args)
    else:
        trainer.train(args)
