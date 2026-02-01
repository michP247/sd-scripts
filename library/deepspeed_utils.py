import os
import argparse
import torch
from accelerate import DeepSpeedPlugin
import sys

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def add_deepspeed_arguments(parser: argparse.ArgumentParser):
    # DeepSpeed Arguments. https://huggingface.co/docs/accelerate/usage_guides/deepspeed
    parser.add_argument("--deepspeed", nargs="?", const=True, default=False, help="enable deepspeed training, optionally pass a config file")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3], help="Possible options are 0,1,2,3.")
    parser.add_argument(
        "--offload_optimizer_device",
        type=str,
        default=None,
        choices=[None, "cpu", "nvme"],
        help="Possible options are none|cpu|nvme. Only applicable with ZeRO Stages 2 and 3.",
    )
    parser.add_argument(
        "--offload_optimizer_nvme_path",
        type=str,
        default=None,
        help="Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--offload_param_device",
        type=str,
        default=None,
        choices=[None, "cpu", "nvme"],
        help="Possible options are none|cpu|nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--offload_param_nvme_path",
        type=str,
        default=None,
        help="Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--zero3_init_flag",
        action="store_true",
        help="Flag to indicate whether to enable `deepspeed.zero.Init` for constructing massive models."
        "Only applicable with ZeRO Stage-3.",
    )
    parser.add_argument(
        "--zero3_save_16bit_model",
        action="store_true",
        help="Flag to indicate whether to save 16-bit model. Only applicable with ZeRO Stage-3.",
    )
    parser.add_argument(
        "--fp16_master_weights_and_gradients",
        action="store_true",
        help="fp16_master_and_gradients requires optimizer to support keeping fp16 master and gradients while keeping the optimizer states in fp32.",
    )
    # ZenFlow Arguments
    parser.add_argument(
        "--zenflow",
        action="store_true",
        help="Enable ZenFlow selective gradient optimization.  Requires CPU offload and PyTorch >= 2.1.",
    )
    parser.add_argument(
        "--zenflow_topk_ratio",
        type=float,
        default=0.1,
        help="ZenFlow top-k ratio for gradient selection (0.0-1.0). Default: 0.1 (10%%)",
    )
    parser.add_argument(
        "--zenflow_overlap_step",
        action="store_true",
        default=True,
        help="Enable ZenFlow optimizer step overlapping with computation.  Default: True",
    )
    parser.add_argument(
        "--zenflow_full_warm_up_rounds",
        type=int,
        default=0,
        help="Number of initial full gradient update rounds before switching to selective updates. Default: 0",
    )
    parser.add_argument(
        "--zenflow_select_interval",
        type=int,
        default=1,
        help="Interval (in steps) for reselecting important gradients. Default: 1 (every step)",
    )
    parser.add_argument(
        "--zenflow_update_interval",
        type=int,
        default=1,
        help="Interval (in steps) for updating accumulated gradients. Default: 1 (every step)",
    )
    # DeepCompile Arguments
    parser.add_argument(
        "--deepcompile",
        action="store_true",
        help="Enable DeepCompile torch.compile optimization for distributed training.",
    )
    parser.add_argument(
        "--deepcompile_backend",
        type=str,
        default="inductor",
        choices=["inductor", "eager"],
        help="Backend for DeepCompile. 'inductor' for performance, 'eager' for debugging.",
    )
    parser.add_argument(
        "--free_activation",
        action="store_true",
        help="Eagerly free large activations to save memory.",
    )
    parser.add_argument(
        "--offload_activation",
        action="store_true",
        help="Offload activations to CPU during training.",
    )
        # Add after other arguments
    parser.add_argument(
        "--compile_network",
        action="store_true",
        help="Compile the LoRA/network module with torch.compile for faster training.",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode. 'reduce-overhead' is good for training, 'max-autotune' for max performance but slower compile.",
    )
    parser.add_argument(
        "--optimize_zero",
        type=int,
        default=0,
        choices=[0, 1],
        help="Apply additional memory optimizations for ZeRO-3 to reduce VRAM usage.",
    )

def prepare_deepspeed_plugin(args: argparse.Namespace):
    if not args.deepspeed:
        return None

    try:
        import deepspeed
    except ImportError as e:
        logger.error(
            "deepspeed is not installed. please install deepspeed in your environment with following command. DS_BUILD_OPS=0 pip install deepspeed"
        )
        exit(1)

    # Use accelerate's DeepSpeedPlugin to correctly handle the base config
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=args.zero_stage,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=args.max_grad_norm,
        offload_optimizer_device=args.offload_optimizer_device,
        offload_optimizer_nvme_path=args.offload_optimizer_nvme_path,
        offload_param_device=args.offload_param_device,
        offload_param_nvme_path=args.offload_param_nvme_path,
        zero3_init_flag=args.zero3_init_flag,
        zero3_save_16bit_model=args.zero3_save_16bit_model,
    )

    # Get the config dictionary that the plugin has built
    ds_config = deepspeed_plugin.deepspeed_config

    # Set batch sizes
    ds_config["train_micro_batch_size_per_gpu"] = args.train_batch_size
    ds_config["train_batch_size"] = (
        args.train_batch_size * args.gradient_accumulation_steps * int(os.environ.get("WORLD_SIZE", 1))
    )

    # Set mixed precision
    deepspeed_plugin.set_mixed_precision(args.mixed_precision)

    # Ensure the zero_optimization dictionary exists if we are using any stage of ZeRO
    if args.zero_stage > 0 and "zero_optimization" not in ds_config:
        ds_config["zero_optimization"] = {}
    
    # Configure parameter offloading for ZeRO-3
    if args.zero_stage == 3:
        ds_config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
        if args.offload_param_device:
            if "offload_param" not in ds_config["zero_optimization"]:
                ds_config["zero_optimization"]["offload_param"] = {}
            ds_config["zero_optimization"]["offload_param"]["device"] = args.offload_param_device
            ds_config["zero_optimization"]["offload_param"]["pin_memory"] = True
            if args.offload_param_device == "nvme":
                ds_config["zero_optimization"]["offload_param"]["nvme_path"] = args.offload_param_nvme_path
                # Buffer size must be >= largest param tensor (SDXL has ~63M element params)
                # buffer_count must be >= params swapped simultaneously during backward
                # Full fine-tuning needs more buffers than LoRA due to gradient checkpointing
                # Total GPU memory for staging: buffer_size * 2 bytes * buffer_count
                # 70M * 2 * 16 = ~2.24GB GPU memory for NVMe param staging
                ds_config["zero_optimization"]["offload_param"]["buffer_size"] = 70_000_000
                ds_config["zero_optimization"]["offload_param"]["buffer_count"] = 16
                ds_config["zero_optimization"]["offload_param"]["max_in_cpu"] = 5e9  # 5GB in CPU staging
        #ds_config["zero_optimization"]["stage3_max_live_parameters"] = 1e8
        #ds_config["zero_optimization"]["stage3_max_reuse_distance"] = 1e8
        ds_config["log_trace_cache_warnings"] = True

    # Configure optimizer offloading for ZeRO-2 and ZeRO-3
    if args.zero_stage >= 2:
        if args.offload_optimizer_device:
            if "offload_optimizer" not in ds_config["zero_optimization"]:
                ds_config["zero_optimization"]["offload_optimizer"] = {}
            ds_config["zero_optimization"]["offload_optimizer"]["device"] = args.offload_optimizer_device
            ds_config["zero_optimization"]["offload_optimizer"]["pin_memory"] = True
            if args.offload_optimizer_device == "nvme":
                ds_config["zero_optimization"]["offload_optimizer"]["nvme_path"] = args.offload_optimizer_nvme_path
                # NVMe optimizer offload also needs buffer configuration
                ds_config["zero_optimization"]["offload_optimizer"]["buffer_count"] = 8
                ds_config["zero_optimization"]["offload_optimizer"]["fast_init"] = False

    # Add memory optimization settings for Stage 3
    if args.optimize_zero == 1 and args.zero_stage == 3:
        # Optimize for low VRAM
        ds_config["zero_optimization"].update({
            "stage3_allgather_sequential": True,
            # "stage3_max_live_parameters": int(5e5),
            # "stage3_max_reuse_distance": int(5e5),
            # "stage3_prefetch_bucket_size": int(3e4),
            # "stage3_param_persistence_threshold": int(1e4),
            # "reduce_bucket_size": int(5e4),
            # "allgather_bucket_size": int(5e4),
            # "sub_group_size": int(5e7),
            # "overlap_comm": True,
            # "contiguous_gradients": True,
            # "reduce_scatter": True,
            # "allgather_partitions": True,
            # "round_robin_gradients": True,
        })
        logger.info("[DeepSpeed] Stage 3 memory optimization settings applied for low VRAM")

    # === DEEPCOMPILE CONFIG ===
    """ if hasattr(args, 'deepcompile') and args.deepcompile:
        ds_config["compile"] = {
            "deepcompile": True,
            "free_activation": getattr(args, 'free_activation', False),
            "free_activation_threshold": 10 * 1024 * 1024,  # 10MB
            "offload_activation": getattr(args, 'offload_activation', False),
            "offload_opt_states": False,
            "double_buffer": True,
            "symmetric_memory": False,
            "debug_log": False,
            "offload_parameters": args.offload_param_device == "cpu" if hasattr(args, 'offload_param_device') else False,
            "sync_before_reduce": False,
            "sync_after_reduce": False,
        }
        logger.info("[DeepSpeed] DeepCompile config added to ds_config") """

    # === ZENFLOW CONFIG ===
    # if hasattr(args, 'zenflow') and args.zenflow:
    #     ds_config["zero_allow_untested_optimizer"] = True
    #     # Check PyTorch version
    #     pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    #     if pytorch_version < (2, 1):
    #         logger.error(
    #             f"ZenFlow requires PyTorch >= 2.1, but found PyTorch {torch.__version__}. "
    #             "Please upgrade PyTorch or disable ZenFlow."
    #         )
    #         exit(1)
        
    #     # Check CPU offload requirement
    #     if args.zero_stage >= 2 and args.offload_optimizer_device != "cpu":
    #         logger.error(
    #             "ZenFlow requires CPU offload to be enabled. "
    #             "Please set --offload_optimizer_device cpu when using --zenflow"
    #         )
    #         exit(1)
        
    #     # Get ZenFlow parameters from args with defaults
    #     topk_ratio = getattr(args, 'zenflow_topk_ratio', 0.1)
    #     overlap_step = getattr(args, 'zenflow_overlap_step', True)
    #     full_warm_up_rounds = getattr(args, 'zenflow_full_warm_up_rounds', 100)
    #     select_interval = getattr(args, 'zenflow_select_interval', 1)
    #     update_interval = getattr(args, 'zenflow_update_interval', 1)
        
    #     # Configure ZenFlow with concrete values (NO "auto" strings!)
    #     ds_config["zero_optimization"]["zenflow"] = {
    #         "topk_ratio": topk_ratio,
    #         "select_strategy": "epoch",  # Options: "epoch", "step"
    #         "select_interval": select_interval,
    #         "update_interval": update_interval,
    #         "overlap_step": overlap_step,
    #         "full_warm_up_rounds": full_warm_up_rounds,
    #     }
        
    #     logger.info(f"[DeepSpeed] ZenFlow enabled:")
    #     logger.info(f"  - topk_ratio: {topk_ratio}")
    #     logger.info(f"  - select_strategy: epoch")
    #     logger.info(f"  - select_interval: {select_interval}")
    #     logger.info(f"  - update_interval: {update_interval}")
    #     logger.info(f"  - overlap_step: {overlap_step}")
    #     logger.info(f"  - full_warm_up_rounds: {full_warm_up_rounds}")

    # # === FIX FOR BFLOAT16 COMMUNICATION ===
    # # Ensure bf16 is properly configured for gradient communication
    # if args.mixed_precision == "bf16":
    #     if "bf16" not in ds_config:
    #         ds_config["bf16"] = {}
    #     ds_config["bf16"]["enabled"] = True
        
    #     # Also set communication data type explicitly
    #     ds_config["communication_data_type"] = "bf16"
        
    #     logger.info("[DeepSpeed] BF16 communication enabled")


    # Configure AIO if any NVMe offloading is used
    is_optimizer_nvme_offload = args.zero_stage >= 2 and args.offload_optimizer_device == "nvme"
    is_param_nvme_offload = args.zero_stage == 3 and args.offload_param_device == "nvme"

    if is_optimizer_nvme_offload or is_param_nvme_offload:
        ds_config["aio"] = {
            "single_submit": False, "overlap_events": True, "num_threads": 2,
            "queue_depth": 128, "block_size": 8388608, "use_gds": True
        }
        logger.info("[DeepSpeed] NVMe offloading configured.")

    # Configure optimizer
    # Ensure learning rate is always set - use unet_lr, fallback to learning_rate, or default to 5e-5
    lr = getattr(args, "unet_lr", None) or getattr(args, "learning_rate", None) or 5e-5

    ds_config["optimizer"] = {
        "type": "Adam",
        "params": {
            "lr": lr,
            "betas": [0.9, 0.999],
            "eps": 1e-08,
            "weight_decay": 0,
        },
    }

    # Configure scheduler
    """ ds_config["scheduler"] = {}

    if args.lr_scheduler == "cosine":
        ds_config["scheduler"] = {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": args.max_train_steps,
                "warmup_min_ratio": 0.0,
                "warmup_num_steps": (int(args.lr_warmup_steps * args.max_train_steps) if isinstance(args.lr_warmup_steps, float) else int(args.lr_warmup_steps)) if hasattr(args, 'lr_warmup_steps') and args.lr_warmup_steps else 0,
                "cos_min_ratio": 0.0,
            }
        }
    if args.lr_scheduler == "constant" or args.lr_scheduler == "constant_with_warmup":
        ds_config["scheduler"] = {
            "type": "WarmupLR",
            "params": {
                "warmup_num_steps": (int(args.lr_warmup_steps * args.max_train_steps) if isinstance(args.lr_warmup_steps, float) else int(args.lr_warmup_steps)) if hasattr(args, 'lr_warmup_steps') and args.lr_warmup_steps else 0,
                "warmup_min_lr": 0.0,
                "warmup_max_lr": lr,
            }
        } """

    if args.mixed_precision and args.mixed_precision.lower() == "fp16":
        ds_config["fp16"]["initial_scale_power"] = 0

    if args.offload_optimizer_device is not None:
        logger.info("[DeepSpeed] start to manually build cpu_adam.")
        deepspeed.ops.op_builder.CPUAdamBuilder().load()
        logger.info("[DeepSpeed] building cpu_adam done.")

    return deepspeed_plugin
    
# Accelerate library does not support multiple models for deepspeed. So, we need to wrap multiple models into a single model.
def prepare_deepspeed_model(args: argparse.Namespace, **models):
    # remove None from models
    models = {k: v for k, v in models.items() if v is not None}

    class DeepSpeedWrapper(torch.nn.Module):
        def __init__(self, **kw_models) -> None:
            super().__init__()
            self.models = torch.nn.ModuleDict()

            for key, model in kw_models.items():
                if isinstance(model, list):
                    model = torch.nn.ModuleList(model)
                assert isinstance(
                    model, torch.nn.Module
                ), f"model must be an instance of torch.nn.Module, but got {key} is {type(model)}"
                self.models.update(torch.nn.ModuleDict({key: model}))

        def get_models(self):
            return self.models

    ds_model = DeepSpeedWrapper(**models)
    return ds_model
