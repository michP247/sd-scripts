import torch
from collections import defaultdict
import numpy as np
import contextlib

def attach_nfn_hooks(model_to_hook, metrics_dict):
    """
    Attaches NFN hooks to a model to calculate alignment metrics as per the PLoP paper.
    Measures the ratio of actual weight-input alignment vs random baseline.
    """
    
    # Check for DeepSpeed availability
    try:
        import deepspeed
        has_deepspeed = True
    except ImportError:
        has_deepspeed = False

    def hook_fn_nfn(name):
        def hook(module, p_input, p_output):
            # Context manager for gathering parameters if using ZeRO-3
            # We check ds_id to see if it's a DeepSpeed parameter
            gather_ctx = contextlib.nullcontext()
            if has_deepspeed and hasattr(module, 'weight') and hasattr(module.weight, 'ds_id'):
                gather_ctx = deepspeed.zero.GatheredParameters([module.weight], modifier_rank=None)

            try:
                with gather_ctx:
                    # We need the weight to calculate NFN
                    if not hasattr(module, 'weight') or module.weight is None:
                        return

                    input_tensor = p_input[0].to(torch.float32)
                    # Avoid calculating if input is empty
                    if input_tensor.numel() == 0:
                        return

                    weight = module.weight.to(torch.float32)
                    is_conv = isinstance(module, torch.nn.Conv2d)

                    # 1. Normalize Weights (Root Mean Square Norm)
                    # PLoP uses (W**2).mean().sqrt()
                    w_norm = (weight ** 2).mean().sqrt()
                    w_normalized = weight / (w_norm + 1e-8)

                    # 2. Prepare and Normalize Input
                    if is_conv:
                        # For Conv2d: Normalize over channel dimension at each spatial location
                        # Input: (N, Cin, H, W)
                        # We want to treat each (N, H, W) location as a vector of size Cin
                        # But we need to preserve structure for F.conv2d to handle stride/padding correctly
                        
                        # Calculate norm per-pixel (N, 1, H, W)
                        input_norm_per_pixel = torch.linalg.vector_norm(input_tensor, dim=1, keepdim=True)
                        input_normalized = input_tensor / (input_norm_per_pixel + 1e-8)
                        
                        # Create random baseline with same shape
                        input_random = torch.randn_like(input_tensor)
                        input_random_norm = torch.linalg.vector_norm(input_random, dim=1, keepdim=True)
                        input_random_normalized = input_random / (input_random_norm + 1e-8)
                        
                        # Compute outputs using NORMALIZED weights and inputs
                        # We use the module's convolution parameters
                        # Note: Bias is ignored for NFN as we measure alignment with W
                        out_actual = torch.nn.functional.conv2d(
                            input_normalized, w_normalized, bias=None, 
                            stride=module.stride, padding=module.padding, 
                            dilation=module.dilation, groups=module.groups
                        )
                        
                        out_random = torch.nn.functional.conv2d(
                            input_random_normalized, w_normalized, bias=None, 
                            stride=module.stride, padding=module.padding, 
                            dilation=module.dilation, groups=module.groups
                        )
                        
                        # 3. Measure Output Norms (Alignment Score)
                        # Out: (N, Cout, H_out, W_out) -> Norm over Cout
                        # Then average over (N, H, W)
                        score_actual = torch.linalg.vector_norm(out_actual, dim=1).mean().item()
                        score_random = torch.linalg.vector_norm(out_random, dim=1).mean().item()

                    else:
                        # Linear layer
                        # Input: (N, ..., Din) -> Flatten to (Batch, Din)
                        input_flat = input_tensor.reshape(-1, input_tensor.shape[-1])
                        
                        input_vec_norm = torch.linalg.vector_norm(input_flat, dim=1, keepdim=True)
                        input_normalized = input_flat / (input_vec_norm + 1e-8)
                        
                        input_random = torch.randn_like(input_flat)
                        input_random_norm = torch.linalg.vector_norm(input_random, dim=1, keepdim=True)
                        input_random_normalized = input_random / (input_random_norm + 1e-8)
                        
                        # Linear projection with normalized weights
                        # w_normalized is (Dout, Din)
                        out_actual = torch.nn.functional.linear(input_normalized, w_normalized)
                        out_random = torch.nn.functional.linear(input_random_normalized, w_normalized)
                        
                        score_actual = torch.linalg.vector_norm(out_actual, dim=1).mean().item()
                        score_random = torch.linalg.vector_norm(out_random, dim=1).mean().item()

                    # 4. Compute NFN Ratio
                    nfn_score = score_actual / (score_random + 1e-8)
                    
                    metrics_dict[name]['nfn'] = nfn_score

            except Exception as e:
                print(f"\n[ERROR] in NFN hook for {name}: {e}")
                if 'input_tensor' in locals():
                    print(f"  Input shape: {input_tensor.shape}")
                if 'weight' in locals():
                    print(f"  Weight shape: {weight.shape}")
                elif hasattr(module, 'weight') and module.weight is not None:
                    print(f"  Weight (sharded/raw) shape: {module.weight.shape}")

        return hook

    hooks = []
    for name, module in model_to_hook.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            # Skip normalization layers or embeddings if they accidentally get caught
            if 'norm' not in name.lower() and 'emb' not in name.lower():
                hooks.append(module.register_forward_hook(hook_fn_nfn(name)))
    return hooks

def remove_nfn_hooks(hooks):
    for hook in hooks:
        hook.remove()

def average_metrics(metrics_list):
    """
    Averages scores for any module that produced at least one valid result.
    """
    if not metrics_list:
        return {}

    sum_metrics = defaultdict(lambda: {'nfn': 0.0, 'count': 0})

    for metrics_dict in metrics_list:
        for name, values in metrics_dict.items():
            if isinstance(values, dict) and 'nfn' in values:
                if isinstance(values['nfn'], (int, float)) and np.isfinite(values['nfn']):
                    sum_metrics[name]['nfn'] += values['nfn']
                    sum_metrics[name]['count'] += 1
            elif isinstance(values, (int, float)) and np.isfinite(values):
                # Handle case where values is directly the NFN score
                sum_metrics[name]['nfn'] += values
                sum_metrics[name]['count'] += 1

    avg_metrics = {}
    for name, data in sum_metrics.items():
        if data['count'] > 0:
            avg_metrics[name] = {'nfn': data['nfn'] / data['count']}

    return avg_metrics
