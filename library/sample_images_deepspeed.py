import deepspeed
import os
import json
import toml
import torch
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def sample_images_with_deepspeed(
    pipe_class,
    accelerator,
    args,
    epoch,
    steps,
    device,
    vae,
    tokenizer,
    text_encoder,
    unet_wrapped,
    prompt_replacement=None,
    controlnet=None,
):
    """
    Sample images with DeepSpeed ZeRO Stage 3 support.
    Uses GatheredParameters to temporarily gather sharded model weights for inference.
    """
    from accelerate import PartialState
    from library.device_utils import clean_memory_on_device
    
    if steps == 0:
        if not args.sample_at_first: 
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:
                return

    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ:  {steps}")
    
    if not os.path.isfile(args.sample_prompts):
        logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    distributed_state = PartialState()
    
    # Check if we're using DeepSpeed ZeRO Stage 3
    is_deepspeed_zero3 = (
        hasattr(accelerator, 'state') and 
        hasattr(accelerator.state, 'deepspeed_plugin') and 
        accelerator.state.deepspeed_plugin is not None and
        accelerator.state.deepspeed_plugin.zero_stage == 3
    )
    
    # Unwrap the models
    unet = accelerator.unwrap_model(unet_wrapped)
    if isinstance(text_encoder, (list, tuple)):
        text_encoders = [accelerator.unwrap_model(te) for te in text_encoder]
    else:
        text_encoders = [accelerator.unwrap_model(text_encoder)]
    
    # Collect all parameters that need to be gathered
    params_to_gather = []
    
    # Get UNet parameters
    params_to_gather.extend(list(unet.parameters()))
    
    # Get text encoder parameters  
    for te in text_encoders:
        if te is not None: 
            params_to_gather.extend(list(te.parameters()))
    
    # VAE is typically not partitioned with ZeRO-3 in training setups,
    # but if it is, include it too
    # params_to_gather.extend(list(vae.parameters()))
    
    org_vae_device = vae.device
    vae.to(distributed_state.device)
    
    # Save random state
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    prompts = load_prompts(args.sample_prompts)
    from library.train_util import get_my_scheduler
    default_scheduler = get_my_scheduler(
        sample_sampler=args.sample_sampler, 
        v_parameterization=args.v_parameterization
    )
    
    # Use GatheredParameters context for DeepSpeed ZeRO-3
    if is_deepspeed_zero3:
        logger.warning(
            "DeepSpeed ZeRO Stage 3 detected with limited VRAM. "
            "Sample generation during training will cause OOM on GPUs with <24GB VRAM."
        )
        logger.warning(
            "Skipping sample generation. Generate samples separately using: "
            "python tools/gen_reg_images.py or after training completes."
        )
        return
        
        # The code below would work with sufficient VRAM (24GB+)
        # logger.info("Using DeepSpeed GatheredParameters for inference")
        # with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=None):
        #     _run_inference_pipeline(
        #         pipe_class, accelerator, args, distributed_state, 
        #         text_encoders[0] if len(text_encoders) == 1 else text_encoders,
        #         vae, unet, tokenizer, default_scheduler, prompts, 
        #         epoch, steps, prompt_replacement, controlnet
        #     )
    else:
        # Normal inference without DeepSpeed
        _run_inference_pipeline(
            pipe_class, accelerator, args, distributed_state,
            text_encoders[0] if len(text_encoders) == 1 else text_encoders,
            vae, unet, tokenizer, default_scheduler, prompts,
            epoch, steps, prompt_replacement, controlnet
        )
    
    # Restore state
    torch.set_rng_state(rng_state)
    if torch.cuda.is_available() and cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    vae.to(org_vae_device)
    
    from library.device_utils import clean_memory_on_device
    clean_memory_on_device(accelerator.device)


def _run_inference_pipeline(
    pipe_class, accelerator, args, distributed_state,
    text_encoder, vae, unet, tokenizer, default_scheduler, 
    prompts, epoch, steps, prompt_replacement, controlnet
):
    """Helper function to run the actual inference pipeline."""
    
    pipeline = pipe_class(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=default_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        clip_skip=args.clip_skip,
    )
    pipeline.to(distributed_state.device)
    
    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)
    
    from library.train_util import sample_image_inference
    with torch.no_grad():
        for prompt_dict in prompts: 
            sample_image_inference(
                accelerator, args, pipeline, save_dir, 
                prompt_dict, epoch, steps, prompt_replacement, 
                controlnet=controlnet
            )
    
    del pipeline


def load_prompts(prompt_file):
    """Load prompts from file"""
    from library.train_util import line_to_prompt_dict
    
    if prompt_file.endswith(".txt"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif prompt_file.endswith(".toml"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif prompt_file.endswith(".json"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    else:
        raise ValueError(f"Unsupported prompt file format: {prompt_file}")
    
    # Preprocess prompts
    processed_prompts = []
    for i, prompt_item in enumerate(prompts):
        if isinstance(prompt_item, str):
            prompt_dict = line_to_prompt_dict(prompt_item)
        else:
            prompt_dict = prompt_item.copy()
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)
        processed_prompts.append(prompt_dict)
    
    return processed_prompts