import os
import argparse
import torch
from accelerate import DeepSpeedPlugin, Accelerator

from . utils import setup_logging

from .device_utils import get_preferred_device

setup_logging()
import logging

logger = logging.getLogger(__name__)


def add_deepspeed_arguments(parser: argparse.ArgumentParser):
    # DeepSpeed Arguments.  https://huggingface.co/docs/accelerate/usage_guides/deepspeed
    parser.add_argument("--deepspeed", action="store_true", help="enable deepspeed training")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3], help="Possible options are 0,1,2,3.")
    parser.add_argument(
        "--offload_optimizer_device",
        type=str,
        default=None,
        choices=[None, "cpu", "nvme"],
        help="Possible options are none|cpu|nvme.  Only applicable with ZeRO Stages 2 and 3.",
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
        help="Possible options are none|cpu|nvme.  Only applicable with ZeRO Stage 3.",
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
        help="Flag to indicate whether to save 16-bit model.  Only applicable with ZeRO Stage-3.",
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
        help="ZenFlow top-k ratio for gradient selection (0.0-1.0). Default:  0.1 (10%%)",
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


def prepare_deepspeed_args(args:  argparse.Namespace):
    if not args.deepspeed:
        return

    # To avoid RuntimeError: DataLoader worker exited unexpectedly with exit code 1.
    args.max_data_loader_n_workers = 1


def prepare_deepspeed_plugin(args: argparse.Namespace):
    if not args.deepspeed:
        return None

    try:
        import deepspeed
    except ImportError as e: 
        logger.error(
            "deepspeed is not installed. please install deepspeed in your environment with following command.  DS_BUILD_OPS=0 pip install deepspeed"
        )
        exit(1)

    # Check ZenFlow requirements
    if hasattr(args, 'zenflow') and args.zenflow:
        # Check PyTorch version
        import torch
        pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if pytorch_version < (2, 1):
            logger.error(
                f"ZenFlow requires PyTorch >= 2.1, but found PyTorch {torch.__version__}.  "
                "Please upgrade PyTorch or disable ZenFlow."
            )
            exit(1)
        
        # Check CPU offload requirement
        if args.zero_stage >= 2 and args.offload_optimizer_device != "cpu":
            logger.error(
                "ZenFlow requires CPU offload to be enabled. "
                "Please set --offload_optimizer_device cpu when using --zenflow"
            )
            exit(1)
        
        logger.info(f"[DeepSpeed] ZenFlow enabled with topk_ratio={args.zenflow_topk_ratio}, overlap_step={args.zenflow_overlap_step}")

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=args.zero_stage,
        gradient_accumulation_steps=args. gradient_accumulation_steps,
        gradient_clipping=args.max_grad_norm,
        offload_optimizer_device=args. offload_optimizer_device,
        offload_optimizer_nvme_path=args.offload_optimizer_nvme_path,
        offload_param_device=args.offload_param_device,
        offload_param_nvme_path=args.offload_param_nvme_path,
        zero3_init_flag=args.zero3_init_flag,
        zero3_save_16bit_model=args.zero3_save_16bit_model,
    )
    deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size
    deepspeed_plugin. deepspeed_config["train_batch_size"] = (
        args.train_batch_size * args.gradient_accumulation_steps * int(os.environ["WORLD_SIZE"])
    )
    
    deepspeed_plugin.set_mixed_precision(args.mixed_precision)
    if args.mixed_precision. lower() == "fp16":
        deepspeed_plugin.deepspeed_config["fp16"]["initial_scale_power"] = 0  # preventing overflow. 
    if args.full_fp16 or args.fp16_master_weights_and_gradients:
        if args. offload_optimizer_device == "cpu" and args.zero_stage == 2:
            deepspeed_plugin.deepspeed_config["fp16"]["fp16_master_weights_and_grads"] = True
            logger. info("[DeepSpeed] full fp16 enable.")
        else:
            logger.info(
                "[DeepSpeed]full fp16, fp16_master_weights_and_grads currently only supported using ZeRO-Offload with DeepSpeedCPUAdam on ZeRO-2 stage."
            )

    # Configure ZeRO optimization settings
    if "zero_optimization" not in deepspeed_plugin.deepspeed_config:
        deepspeed_plugin.deepspeed_config["zero_optimization"] = {}
    
    zero_config = deepspeed_plugin.deepspeed_config["zero_optimization"]

    # Configure optimizer
    # Ensure learning rate is always set - use unet_lr, fallback to learning_rate, or default to 5e-5
    lr = getattr(args, "unet_lr", None) or getattr(args, "learning_rate", None) or 5e-5

    zero_config["optimizer"] = {
        "type": "Adam",
        "params": {
            "lr": lr,
            "betas": [0.9, 0.999],
            "eps": 1e-08,
            "weight_decay": 0.1,
        },
    }
    
    # Add ZenFlow configuration if enabled
    if hasattr(args, 'zenflow') and args.zenflow:
        # Use explicit integer/string values instead of "auto"
        # These values work with Accelerate's validation
        zenflow_config = {
            "topk_ratio": args.zenflow_topk_ratio,
            "select_strategy": "epoch",  # Use "epoch" or "step" instead of "auto"
            "select_interval": getattr(args, 'zenflow_select_interval', 1),
            "update_interval": getattr(args, 'zenflow_update_interval', 1),
            "overlap_step": args.zenflow_overlap_step,
            "offload":  True,
            "full_warm_up_rounds": args.zenflow_full_warm_up_rounds,
            "auto_ratio": 0.99,
        }
        
        # If using epoch strategy, we need steps_per_epoch
        # This will be set by DeepSpeed automatically if not provided
        if zenflow_config["select_strategy"] == "epoch" and hasattr(args, 'max_train_steps'):
            # Calculate approximate steps per epoch
            # This is a rough estimate; DeepSpeed will adjust if needed
            if hasattr(args, 'max_train_epochs') and args.max_train_epochs: 
                zenflow_config["steps_per_epoch"] = args.max_train_steps // args.max_train_epochs
        
        zero_config["zenflow"] = zenflow_config
        logger.info(f"[DeepSpeed] ZenFlow configuration added to DeepSpeed config: {zenflow_config}")
    
    # Add memory optimization settings for Stage 3
    if args.zero_stage == 3:
        # Optimize for low VRAM (6GB)
        zero_config. update({
            "stage3_max_live_parameters": int(5e5),
            "stage3_max_reuse_distance": int(5e5),
            "stage3_prefetch_bucket_size": int(3e4),
            "stage3_param_persistence_threshold": int(1e4),
            "reduce_bucket_size": int(5e4),
            "allgather_bucket_size": int(5e4),
            "sub_group_size": int(5e7),
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "allgather_partitions": True,
            "round_robin_gradients": True,
        })
        logger.info("[DeepSpeed] Stage 3 memory optimization settings applied for low VRAM")
    
    # Add general optimization settings
    if args.zero_stage >= 2:
        zero_config.update({
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
        })

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
