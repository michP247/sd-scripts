# training with captions

import argparse
import math
import os
from multiprocessing import Value
from typing import List
import toml

from tqdm import tqdm

import torch
import torch.nn as nn  # Added this line
from library.device_utils import init_ipex, clean_memory_on_device

# TPU-specific imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.serialization as xser

from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import deepspeed_utils, sdxl_model_util

import library.train_util as train_util

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util
import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.sdxl_original_unet import SdxlUNet2DConditionModel

UNET_NUM_BLOCKS_FOR_BLOCK_LR = 23

def get_block_params_to_optimize(unet: SdxlUNet2DConditionModel, block_lrs: List[float]) -> List[dict]:
    block_params = [[] for _ in range(len(block_lrs))]

    for i, (name, param) in enumerate(unet.named_parameters()):
        if name.startswith("time_embed.") or name.startswith("label_emb."):
            block_index = 0  # 0
        elif name.startswith("input_blocks."):  # 1-9
            block_index = 1 + int(name.split(".")[1])
        elif name.startswith("middle_block."):  # 10-12
            block_index = 10 + int(name.split(".")[1])
        elif name.startswith("output_blocks."):  # 13-21
            block_index = 13 + int(name.split(".")[1])
        elif name.startswith("out."):  # 22
            block_index = 22
        else:
            raise ValueError(f"unexpected parameter name: {name}")

        block_params[block_index].append(param)

    params_to_optimize = []
    for i, params in enumerate(block_params):
        if block_lrs[i] == 0:  # 0のときは学習しない do not optimize when lr is 0
            continue
        params_to_optimize.append({"params": params, "lr": block_lrs[i]})

    return params_to_optimize

def append_block_lr_to_logs(block_lrs, logs, lr_scheduler, optimizer_type):
    names = []
    block_index = 0
    while block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR + 2:
        if block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            if block_lrs[block_index] == 0:
                block_index += 1
                continue
            names.append(f"block{block_index}")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            names.append("text_encoder1")
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR + 1:
            names.append("text_encoder2")

        block_index += 1

    train_util.append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)

def train(args, train_dataloader=None):
    print("Starting training...")
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    sdxl_train_util.verify_sdxl_training_args(args)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    assert (
        not args.weighted_captions
    ), "weighted_captions is not supported currently"
    assert (
        not args.train_text_encoder or not args.cache_text_encoder_outputs
    ), "cache_text_encoder_outputs is not supported when training text encoder"

    if args.block_lr:
        block_lrs = [float(lr) for lr in args.block_lr.split(",")]
        assert (
            len(block_lrs) == UNET_NUM_BLOCKS_FOR_BLOCK_LR
        ), f"block_lr must have {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / block_lrは{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値を指定してください"
    else:
        block_lrs = None

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # Initialize the random number series

    tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)

    # Prepare the dataset
    if args.dataset_config is not None: #Check this FIRST
        logger.info(f"Load dataset config from {args.dataset_config}")
        user_config = config_util.load_user_config(args.dataset_config)
        # Additional logging to ensure correct loading
        print(f"Using dataset config: {user_config}")
    elif args.dataset_class is not None: # Check for custom datasets only if dataset_config is None.
        print("Loading arbitrary dataset class...")
        train_dataset_group = train_util.load_arbitrary_dataset(args, [tokenizer1, tokenizer2])
    else: 
        print("No dataset configuration or class provided. Using default training method.")
        ignored = ["train_data_dir", "in_json"]
        if any(getattr(args, attr) is not None for attr in ignored):
            logger.warning(
                "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                    ", ".join(ignored)
                )
            )
    if args.dataset_class is not None: #Check for custom datasets only if dataset_config is None.
        train_dataset_group = train_util.load_arbitrary_dataset(args, [tokenizer1, tokenizer2])
    else: #no dataset_config or dataset_class
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else: #Training with captions
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }
    #After the conditional block:
    if args.dataset_class is None: #Only generate the blueprint if NOT using a custom dataset class
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
            blueprint = blueprint_generator.generate(user_config, args, tokenizer=[tokenizer1, tokenizer2])
            train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, [tokenizer1, tokenizer2])

    print(f"Length of train_dataset_group: {len(train_dataset_group)}")

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

    # Prepare Accelerator
    logger.info("prepare accelerator")

    if getattr(args, 'use_tpu', False):
        # TPU-specific initialization
        device = xm.xla_device()
        accelerator = train_util.prepare_accelerator(args, device=device)
    else:
        # Original GPU/CPU setup
        accelerator = train_util.prepare_accelerator(args)

    #Prepare a type that supports mixed Precision and Cast as appropriate
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # Read the model
    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)
    # logit_scale = logit_scale.to(accelerator.device, dtype=weight_dtype)

    # verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path

    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:
        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())
        # assert save_stable_diffusion_format, "save_model_as must be ckpt or safetensors / save_model_asはckptかsafetensorsである必要があります"

    # Diffusers version of the XFORMERS used function to configure a flag
    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    # Incorporate XFORMERS or Memory Effication Attension into the model
    if args.diffusers_xformers:
        # もうU-Netを独自にしたので動かないけどVAEのxformersは動くはず
        accelerator.print("Use xformers by Diffusers")
        # set_diffusers_xformers_flag(unet, True)
        set_diffusers_xformers_flag(vae, True)
    else:
        # Windows版のxformersはfloatで学習できなかったりするのでxformersを使わない設定も可能にしておく必要がある
        accelerator.print("Disable Diffusers' xformers")
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

    # move weights to TPU if training on TPU
    if getattr(args, 'use_tpu', False):
        unet.to(device)
        unet.time_embed.to(device)
        unet.label_emb.to(device)

    # Prepare learning
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()
        with torch.no_grad():
            train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
        vae.to("cpu")
        if not getattr(args, 'use_tpu', False): #Only clean memory if not TPU.
            clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # Prepare learning: Make the model appropriate
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    train_unet = args.learning_rate > 0
    train_text_encoder1 = False
    train_text_encoder2 = False

    if args.train_text_encoder:
        # TODO each option for two text encoders?
        accelerator.print("enable text encoder training")
        if args.gradient_checkpointing:
            text_encoder1.gradient_checkpointing_enable()
            text_encoder2.gradient_checkpointing_enable()
        lr_te1 = args.learning_rate_te1 if args.learning_rate_te1 is not None else args.learning_rate  # 0 means not train
        lr_te2 = args.learning_rate_te2 if args.learning_rate_te2 is not None else args.learning_rate  # 0 means not train
        train_text_encoder1 = lr_te1 > 0
        train_text_encoder2 = lr_te2 > 0

        # caching one text encoder output is not supported
        if not train_text_encoder1:
            text_encoder1.to(weight_dtype)
        if not train_text_encoder2:
            text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(train_text_encoder1)
        text_encoder2.requires_grad_(train_text_encoder2)
        text_encoder1.train(train_text_encoder1)
        text_encoder2.train(train_text_encoder2)
    else:
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
        text_encoder1.requires_grad_(False)
        text_encoder2.requires_grad_(False)
        text_encoder1.eval()
        text_encoder2.eval()

        # Cache the output of TextEncoder
        if args.cache_text_encoder_outputs:
            # Text Encodes are eval and no grad
            with torch.no_grad(), accelerator.autocast():
                train_dataset_group.cache_text_encoder_outputs(
                    (tokenizer1, tokenizer2),
                    (text_encoder1, text_encoder2),
                    accelerator.device,
                    None,
                    args.cache_text_encoder_outputs_to_disk,
                    accelerator.is_main_process,
                )
            accelerator.wait_for_everyone()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    unet.requires_grad_(train_unet)
    if not train_unet:
        unet.to(accelerator.device, dtype=weight_dtype)  # because of unet is not prepared

    training_models = []
    params_to_optimize = []
    if train_unet:
      logger.info("Converting SdxlUNet2DConditionModel to TPU...")
      new_unet_dict = {}
      for key, item in accelerator.unwrap_model(unet).state_dict().items():
        if getattr(args, 'use_tpu', False):
          new_unet_dict[key] = item.to(xm.xla_device())
        else:
          new_unet_dict[key] = item.to(accelerator.device) #Keep original codepath for other devices.
      accelerator.unwrap_model(unet).load_state_dict(new_unet_dict)
      logger.info("Converted Successfully to TPU.")

      if block_lrs is None:
          params_to_optimize.append({"params": list(unet.parameters()), "lr": args.learning_rate})
      else:
          params_to_optimize.extend(get_block_params_to_optimize(unet, block_lrs))

    if train_text_encoder1:
        training_models.append(text_encoder1)
        params_to_optimize.append({"params": list(text_encoder1.parameters()), "lr": args.learning_rate_te1 or args.learning_rate})
    if train_text_encoder2:
        training_models.append(text_encoder2)
        params_to_optimize.append({"params": list(text_encoder2.parameters()), "lr": args.learning_rate_te2 or args.learning_rate})

    # calculate number of trainable parameters
    n_params = 0
    for params in params_to_optimize:
        for p in params["params"]:
            n_params += p.numel()

    accelerator.print(f"train unet: {train_unet}, text_encoder1: {train_text_encoder1}, text_encoder2: {train_text_encoder2}")
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    # Prepare the classes required for learning
    accelerator.print("prepare optimizer, data loader etc.")
    _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)

    # Prepare Dataloader

    # Calculate the number of training steps
    if args.max_train_epochs is not None:
        num_samples = len(train_dataset_group.datasets[0]) # Access the first dataset in the group, since it is not a concatenated dataset but a list containing a single FineTuningDataset.
        num_update_steps_per_epoch = math.ceil(num_samples / args.gradient_accumulation_steps)
        # Modify the following line
        num_update_steps_per_epoch = math.ceil(num_update_steps_per_epoch / accelerator.num_processes)
        args.max_train_steps = args.max_train_epochs * num_update_steps_per_epoch
        print("Steps set correctly for multi-epoch TPU training.")  # Debugging print
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is: {args.max_train_steps}"
        )
    print(f"Number of images in dataset: {train_dataset_group.num_train_images}")
    print(f"Number of batches: {len(train_dataloader)}")
    print(f"Dataset class: {args.dataset_class if args.dataset_class else 'default'}")
    
    # Send training steps to the dataset side as well
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # Prepare LR Scheduler
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # Experimental functions: FP16/BF16, including gradient, set the entire model to learn to FP16/BF16
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        unet.to(weight_dtype)
        text_encoder1.to(weight_dtype)
        text_encoder2.to(weight_dtype)

    # freeze last layer and final_layer_norm in te1 since we use the output of the penultimate layer
    if train_text_encoder1:
        text_encoder1.text_model.encoder.layers[-1].requires_grad_(False)
        text_encoder1.text_model.final_layer_norm.requires_grad_(False)

    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            args,
            unet=unet if train_unet else None,
            text_encoder1=text_encoder1 if train_text_encoder1 else None,
            text_encoder2=text_encoder2 if train_text_encoder2 else None,
        )
        # most of ZeRO stage uses optimizer partitioning, so we have to prepare optimizer and ds_model at the same time. # pull/1139#issuecomment-1986790007
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]

    else:
        # acceleratorがなんかよろしくやってくれるらしい
        if train_unet:
            unet = accelerator.prepare(unet)
        if train_text_encoder1:
            text_encoder1 = accelerator.prepare(text_encoder1)
        if train_text_encoder2:
            text_encoder2 = accelerator.prepare(text_encoder2)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # Move to the CPU when caching TextEncoder output
    if args.cache_text_encoder_outputs:
        # move Text Encoders for sampling images. Text Encoder doesn't work on CPU with fp16
        text_encoder1.to("cpu", dtype=torch.float32)
        text_encoder2.to("cpu", dtype=torch.float32)
        clean_memory_on_device(accelerator.device)
    else:
        # make sure Text Encoders are on GPU
        text_encoder1.to(accelerator.device)
        text_encoder2.to(accelerator.device)

    # Experimental function: FP16 learning including gradient Enables Grad Scale in FP16 by patching Pytorch
    if args.full_fp16:
        # During deepseed training, accelerate not handles fp16/bf16|mixed precision directly via scaler. Let deepspeed engine do.
        # -> But we think it's ok to patch accelerator even if deepspeed is enabled.
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # Calculate the number of Epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    try:
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    except ZeroDivisionError:
        raise ValueError("Number of update steps per epoch is zero. Please ensure that your dataloader is not empty.")
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # Training
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    # accelerator.print(
    #     f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    # )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    if args.zero_terminal_snr:
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers("finetuning" if args.log_tracker_name is None else args.log_tracker_name, init_kwargs=init_kwargs)

    # For --sample_at_first
    sdxl_train_util.sample_images(
        accelerator, args, 0, global_step, accelerator.device, vae, [tokenizer1, tokenizer2], [text_encoder1, text_encoder2], unet
    )

    loss_recorder = train_util.LossRecorder()

    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        logger.info(f"Number of images in training dataset: {len(train_dataset_group)}")

        if not train_dataset_group:
            logger.error("Training dataset is empty. Please check your dataset configuration.")
            return
        
        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            with accelerator.accumulate(*training_models):
                if getattr(args, 'use_tpu', False):
                    # TPU Codepath: Transfer tensors within the TPU context.
                    input_ids1 = batch["input_ids"].to(device)
                    input_ids2 = batch["input_ids2"].to(device)
                
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(device, dtype=weight_dtype)
                    else:
                        latents = vae.encode(batch["images"].to(device, dtype=vae_dtype)).latent_dist.sample().to(device, dtype=weight_dtype)
                    latents = latents * sdxl_model_util.VAE_SCALE_FACTOR

                    orig_size = batch["original_sizes_hw"].to(device)
                    crop_size = batch["crop_top_lefts"].to(device)
                    target_size = batch["target_sizes_hw"].to(device)
                else: #Original non-TPU codepath.
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                    else:
                        with torch.no_grad():
                            # latentに変換
                            latents = vae.encode(batch["images"].to(vae_dtype)).latent_dist.sample().to(weight_dtype)

                            # NaNが含まれていれば警告を表示し0に置き換える
                            if torch.any(torch.isnan(latents)):
                                accelerator.print("NaN found in latents, replacing with zeros")
                                latents = torch.nan_to_num(latents, 0, out=latents)
                    latents = latents * sdxl_model_util.VAE_SCALE_FACTOR

                    #Move all inputs to the device BEFORE doing anything else.
                    input_ids1 = batch["input_ids"].to(accelerator.device)
                    input_ids2 = batch["input_ids2"].to(accelerator.device)

                    orig_size = batch["original_sizes_hw"].to(accelerator.device)
                    crop_size = batch["crop_top_lefts"].to(accelerator.device)
                    target_size = batch["target_sizes_hw"].to(accelerator.device)

                if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
                    input_ids1 = batch["input_ids"]
                    input_ids2 = batch["input_ids2"]
                    with torch.set_grad_enabled(args.train_text_encoder):
                        # Get the text embedding for conditioning
                        # TODO support weighted captions
                        # if args.weighted_captions:
                        #     encoder_hidden_states = get_weighted_text_embeddings(
                        #         tokenizer,
                        #         text_encoder,
                        #         batch["captions"],
                        #         accelerator.device,
                        #         args.max_token_length // 75 if args.max_token_length else 1,
                        #         clip_skip=args.clip_skip,
                        #     )
                        # else:
                        input_ids1 = input_ids1.to(accelerator.device)
                        input_ids2 = input_ids2.to(accelerator.device)
                        # unwrap_model is fine for models not wrapped by accelerator
                        encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                            args.max_token_length,
                            input_ids1,
                            input_ids2,
                            tokenizer1,
                            tokenizer2,
                            text_encoder1,
                            text_encoder2,
                            None if not args.full_fp16 else weight_dtype,
                            accelerator=accelerator,
                        )
                else:
                    encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
                    encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
                    pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

                    # # verify that the text encoder outputs are correct
                    # ehs1, ehs2, p2 = train_util.get_hidden_states_sdxl(
                    #     args.max_token_length,
                    #     batch["input_ids"].to(text_encoder1.device),
                    #     batch["input_ids2"].to(text_encoder1.device),
                    #     tokenizer1,
                    #     tokenizer2,
                    #     text_encoder1,
                    #     text_encoder2,
                    #     None if not args.full_fp16 else weight_dtype,
                    # )
                    # b_size = encoder_hidden_states1.shape[0]
                    # assert ((encoder_hidden_states1.to("cpu") - ehs1.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
                    # assert ((encoder_hidden_states2.to("cpu") - ehs2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
                    # assert ((pool2.to("cpu") - p2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
                    # logger.info("text encoder outputs verified")

                # Get size embeddings. These operations need to be on the right device.
                orig_size = orig_size.to(device if getattr(args, 'use_tpu', False) else accelerator.device)
                crop_size = crop_size.to(device if getattr(args, 'use_tpu', False) else accelerator.device)
                target_size = target_size.to(device if getattr(args, 'use_tpu', False) else accelerator.device)
                
                embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, device if getattr(args, 'use_tpu', False) else accelerator.device).to(weight_dtype)
                vector_embedding = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, device if getattr(args, 'use_tpu', False) else accelerator.device).to(weight_dtype)

                # Concatenate embeddings after they have been computed
                text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)
                #Place text_embedding on the correct device.
                text_embedding = text_embedding.to(device if getattr(args, 'use_tpu', False) else accelerator.device)

                #Move noise and related tensors to device
                noise = torch.randn_like(latents, device=device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                timesteps = timesteps.to(device)

                # device was set earlier, no need to do this:
                # if getattr(args, 'use_tpu', False):
                #     noisy_latents = noisy_latents.to(accelerator.device, dtype=weight_dtype)
                #     timesteps = timesteps.to(accelerator.device)

                print(f"Device of noisy_latents: {noisy_latents.device}")
                print(f"Device of timesteps: {timesteps.device}")
                print(f"Device of text_embedding: {text_embedding.device}")
                print(f"Device of vector_embedding: {vector_embedding.device}")

                with accelerator.autocast(): #Keep unet inside the TPU context to prevent errors.
                    noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)

                target = noise.to(device if getattr(args, 'use_tpu', False) else accelerator.device) #target noise needs to be placed onto the device.
                loss = train_util.conditional_loss(noise_pred.float(), target.float(), reduction="none", loss_type=args.loss_type, huber_c=huber_c) #Loss calculation also needs to be on-device, to keep operations within the XLA graph

                if (
                    args.min_snr_gamma
                    or args.scale_v_pred_loss_like_noise_pred
                    or args.v_pred_like_loss
                    or args.debiased_estimation_loss
                    or args.masked_loss
                ):
                    # do not mean over batch dimension for snr weight or scale v-pred loss
                    loss = train_util.conditional_loss(noise_pred.float(), target.float(), reduction="none", loss_type=args.loss_type, huber_c=huber_c)
                    if args.masked_loss:
                        loss = apply_masked_loss(loss, batch)
                    loss = loss.mean([1, 2, 3])

                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)
                    if args.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                    if args.debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler)

                    loss = loss.mean()  # mean over batch dimension
                else:
                    loss = train_util.conditional_loss(noise_pred.float(), target.float(), reduction="mean", loss_type=args.loss_type, huber_c=huber_c)

                #Backward pass
                accelerator.backward(loss)

                if getattr(args, 'use_tpu', False):
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        # Clip gradients if necessary
                        params_to_clip = []
                        for m in training_models:
                            params_to_clip.extend(m.parameters())
                        
                        xm.all_reduce("sum", [p.grad for p in params_to_clip if p.grad is not None])
                        #Perform gradient clipping using Pytorch
                        nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                        #accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                    #Optimizer step
                    #optimizer.step()
                    if getattr(args, 'use_tpu', False):
                        xm.optimizer_step(optimizer, barrier=True) # Use xm.optimizer_step
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        xm.mark_step() # Increment step counter
                else:
                    # Original GPU/CPU logic
                    if not (args.fused_backward_pass or args.fused_optimizer_groups):
                        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                            params_to_clip = []
                            for m in training_models:
                                params_to_clip.extend(m.parameters())
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

            # TPU-specific logging and synchronization
            if getattr(args, 'use_tpu', False):
                # Example: Log loss on the master core
                if xm.is_master_ordinal():
                    current_loss = loss.detach().item()
                    print(f"Epoch {epoch+1}, Step {step+1}, Loss: {current_loss}")

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                #Sample images
                sdxl_train_util.sample_images(
                    accelerator,
                    args,
                    None,
                    global_step,
                    accelerator.device,
                    vae,
                    [tokenizer1, tokenizer2],
                    [text_encoder1, text_encoder2],
                    unet,
                )

                # Save the model at each specified step
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(text_encoder1),
                            accelerator.unwrap_model(text_encoder2),
                            accelerator.unwrap_model(unet),
                            vae,
                            logit_scale,
                            ckpt_info,
                        )
            #Log metrics
            current_loss = loss.detach().item()  # Since it is an average, batch size should be irrelevant.
            if args.logging_dir is not None:
                logs = {"loss": current_loss}
                if block_lrs is None:
                    train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=train_unet)
                else:
                    append_block_lr_to_logs(block_lrs, logs, lr_scheduler, args.optimizer_type)  # U-Net is included in block_lrs

                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(text_encoder1),
                    accelerator.unwrap_model(text_encoder2),
                    accelerator.unwrap_model(unet),
                    vae,
                    logit_scale,
                    ckpt_info,
                )

        sdxl_train_util.sample_images(
            accelerator,
            args,
            epoch + 1,
            global_step,
            accelerator.device,
            vae,
            [tokenizer1, tokenizer2],
            [text_encoder1, text_encoder2],
            unet,
        )

    is_main_process = accelerator.is_main_process
    # if is_main_process:
    unet = accelerator.unwrap_model(unet)
    text_encoder1 = accelerator.unwrap_model(text_encoder1)
    text_encoder2 = accelerator.unwrap_model(text_encoder2)

    accelerator.end_training()

    if args.save_state or args.save_state_on_train_end:        
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # I'm going to use memory after this, so I'm going to turn this off.

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        if getattr(args, 'use_tpu', False):
            sdxl_train_util.save_sd_model_on_train_end(  # Correctly use this function
                args,
                src_path,
                save_stable_diffusion_format,
                use_safetensors,
                save_dtype,
                epoch,
                global_step,
                text_encoder1,  # Pass unwrapped models
                text_encoder2,  # Pass unwrapped models
                unet,         # Pass unwrapped models
                vae,
                logit_scale,
                ckpt_info,
            )
        logger.info("model saved.")

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)

    parser.add_argument(
        "--learning_rate_te1",
        type=float,
        default=None,
        help="learning rate for text encoder 1 (ViT-L) / text encoder 1 (ViT-L)の学習率",
    )
    parser.add_argument(
        "--learning_rate_te2",
        type=float,
        default=None,
        help="learning rate for text encoder 2 (BiG-G) / text encoder 2 (BiG-G)の学習率",
    )

    parser.add_argument(
        "--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--block_lr",
        type=str,
        default=None,
        help=f"learning rates for each block of U-Net, comma-separated, {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / "
        + f"U-Netの各ブロックの学習率、カンマ区切り、{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値",
    )
    parser.add_argument("--use_cpu", action="store_true", help="use CPU instead of GPU")
    parser.add_argument("--sdxl", action="store_true", help="Use SDXL model / SDXLモデルを使用する")
    parser.add_argument("--additional_parameters", type=str, default="")
    parser.add_argument("--disable_mmap_load_safetensors", action="store_true")
    parser.add_argument("--dynamo_mode", type=str, default="default")
    parser.add_argument("--dynamo_use_dynamic", action="store_true")
    parser.add_argument("--dynamo_use_fullgraph", action="store_true")
    parser.add_argument("--epoch", type=int, default=1) #Consider renaming this to something like starting_epoch
    parser.add_argument("--fused_backward_pass", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--learning_rate_te", type=float, default=0)
    parser.add_argument("--lr_warmup", type=float, default=0)  # Deprecated in diffusers, use lr_warmup_steps
    parser.add_argument("--main_process_port", type=int, default=0)
    parser.add_argument("--max_resolution", type=str, default="512,512")
    parser.add_argument("--mem_eff_save", action="store_true")
    parser.add_argument("--model_list", type=str, default="custom")
    parser.add_argument("--model_prediction_type", type=str, default="raw")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--no_token_padding", action="store_true")
    parser.add_argument("--noise_offset_type", type=str, default="Original")
    parser.add_argument("--num_cpu_threads_per_process", type=int, default=2)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--save_as_bool", action="store_true")
    parser.add_argument("--save_clip", action="store_true")
    parser.add_argument("--single_blocks_to_swap", type=str, default=None)  # For single block swapping
    parser.add_argument("--skip_cache_check", action="store_true")
    parser.add_argument("--split_mode", action="store_true")
    parser.add_argument("--stop_text_encoder_training", type=int, default=0)
    parser.add_argument("--timestep_sampling", type=str, default="sigmoid")
    #parser.add_argument("--use_tpu", action='store_true', default=False)
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal")
    parser.add_argument("--network_multiplier", type=float, default=1.0)

    return parser

if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_commadnd_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)