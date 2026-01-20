from safetensors.torch import load_file
import sys
import torch

if len(sys.argv) < 2:
    print("Usage: python inspect_lora.py <path_to_lora.safetensors>")
    sys.exit(1)

lora_path = sys.argv[1]
try:
    state_dict = load_file(lora_path)
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if tensor.dtype == torch.int64:
            # Skip printing stats for non-float tensors like 'ss_epoch'
            print(f"{key}: {tensor.shape} | dtype: {tensor.dtype} | value: {tensor.item()}")
        else:
            # Move to float32 for accurate stats
            tensor = tensor.to(torch.float32)
            mean = tensor.mean().item()
            std = tensor.std().item()
            max_val = tensor.max().item()
            min_val = tensor.min().item()
            print(f"{key}: {tensor.shape} | dtype: {state_dict[key].dtype} | mean: {mean:.6f} | std: {std:.6f} | max: {max_val:.6f} | min: {min_val:.6f}")

except Exception as e:
    print(f"Error loading or reading file: {e}")
