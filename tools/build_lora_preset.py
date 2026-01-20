#!/usr/bin/env python3
"""
Build a LyCORIS preset (TOML) for SDXL UNet using data-aware ranks from
block learning-rate weights (NFN-like saliency). Sticks to LoRA with DoRA.

Usage examples:
  python tools/build_lora_preset.py \
    --output /home/en1u2/AI-Stuff/lyco_presets/sdxl_dataaware_lora.toml \
    --block_lr_weights "0.1000,0.3166,1.0994,1.9848,1.8143,1.8956,2.0000,1.8519,1.7998,1.8304,1.8200,1.8486,1.8772,1.9282,1.9434,1.9142,1.8286,1.8756,0.4671" \
    --base_attn 48 --base_mlp 40 --min_rank 24 --max_rank 96

Notes:
- Targets SDXL Transformer2DModel modules only (attn + MLP), keeps norms off.
- Generates name-based rank overrides; alpha = rank * alpha_scale; DoRA enabled.
- Maps block indices: input (0..9), middle (0..2), output (0..9).
  The provided block weights are applied to output blocks; input/middle use base ranks.
"""
import argparse
import toml
import os
import math


def get_alpha(rank, alpha_scale, use_rslora):
    if use_rslora:
        return alpha_scale * math.sqrt(rank)
    return rank * alpha_scale


# Use fnmatch-style glob patterns to match Kohya names when preset sets use_fnmatch=true
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


def clamp(v, lo, hi):
    return max(lo, min(hi, int(round(v))))

def build_name_map(block_weights, base_attn, base_mlp, lo, hi, gamma=1.0, alpha_scale=1.0, threshold=0.0, fixed_rank=False, algo="lora", min_factor=-1, max_factor=-1, use_rslora=False):
    name_map = {}

    # Ensure we have enough weights for SDXL (9 input + 1 mid + 9 output = 19)
    # If not, pad or truncate safely
    if len(block_weights) < 19:
        # Fallback: duplicate last weight or pad 1.0
        block_weights = list(block_weights) + [1.0] * (19 - len(block_weights))
    
    # Extract weights for each section
    # NFN order: Down 0-8, Mid, Up 0-8
    w_input = block_weights[0:9]
    w_mid = block_weights[9]
    w_output = block_weights[10:19]

    def get_params(base, w):
        if w < threshold:
            return None
        
        if fixed_rank:
            rank = clamp(base, lo, hi)
        else:
            w_eff = (w ** gamma) if gamma != 1.0 else w
            rank = clamp(base * w_eff, lo, hi)
            
        params = {
            "algo": algo,
            "dim": rank,
            "alpha": get_alpha(rank, alpha_scale, use_rslora),
        }
        
        if algo == "lokr" and min_factor != -1 and max_factor != -1:
            # Dynamic factor scaling
            # w is "importance" or "misalignment" (Higher = Needs more adaptation)
            # Request: "well-aligned (Low W) names should get higher factors"
            # So: Low W -> Max Factor, High W -> Min Factor
            
            # Normalize w roughly to 0.0 - 2.0 range (assuming mean ~1.0)
            # We clamp w to 0.1 - 2.0 for factor calculation to avoid extremes
            w_clamped = max(0.1, min(2.0, w))
            
            # Linear interpolation:
            # t = 0 (w=0.1) -> max_factor
            # t = 1 (w=2.0) -> min_factor
            t = (w_clamped - 0.1) / (2.0 - 0.1)
            
            # Invert t because High W -> Min Factor
            t = 1.0 - t
            
            factor_val = min_factor + t * (max_factor - min_factor)
            params["factor"] = clamp(factor_val, min_factor, max_factor)
            
        elif algo == "lokr":
             params["factor"] = -1
        
        return params

    # Input blocks (0-8)
    for i in range(9):
        w = w_input[i]
        attn_params = get_params(base_attn, w)
        mlp_params = get_params(base_mlp, w)
        
        if attn_params:
            name_map[ATTN_PAT["input"](i)] = attn_params
        if mlp_params:
            name_map[MLP_PAT["input"](i)] = mlp_params

    # Middle block (only index 1 usually has transformers in SDXL UNet)
    # We map the single mid_block weight to this.
    # The pattern matches middle_block.*.transformer_blocks...
    mid_attn_params = get_params(base_attn, w_mid)
    mid_mlp_params = get_params(base_mlp, w_mid)
    
    for i in range(3): # Checking all 3 sub-blocks of mid just in case patterns match
        if mid_attn_params:
            name_map[ATTN_PAT["middle"](i)] = mid_attn_params
        if mid_mlp_params:
            name_map[MLP_PAT["middle"](i)] = mid_mlp_params

    # Output blocks (0-8)
    for i in range(9):
        w = w_output[i]
        attn_params = get_params(base_attn, w)
        mlp_params = get_params(base_mlp, w)

        if attn_params:
            name_map[ATTN_PAT["output"](i)] = attn_params
        if mlp_params:
            name_map[MLP_PAT["output"](i)] = mlp_params

    return name_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="output TOML preset path")
    ap.add_argument(
        "--block_lr_weights",
        required=True,
        help="comma-separated floats or path to a file with one line of comma-separated weights",
    )
    ap.add_argument("--base_attn", type=int, default=48)
    ap.add_argument("--base_mlp", type=int, default=40)
    ap.add_argument("--min_rank", type=int, default=24)
    ap.add_argument("--max_rank", type=int, default=96)
    ap.add_argument("--gamma", type=float, default=1.0, help="exponent applied to block weights to accentuate differences ( >1 spreads, <1 compresses )")
    ap.add_argument("--alpha_scale", type=float, default=1.0, help="Factor to scale alpha relative to rank (alpha = rank * alpha_scale)")
    ap.add_argument(
        "--scope",
        type=str,
        default="attn-mlp",
        choices=["full", "full-lin", "attn-mlp", "attn-only", "unet-transformer-only"],
        help="Layer scope to target (mirrors LyCORIS presets)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="NFN weight threshold. Blocks with weight < threshold will be disabled (rank 0).",
    )
    ap.add_argument(
        "--fixed_rank",
        action="store_true",
        help="If set, enabled blocks get fixed base_attn/base_mlp ranks instead of scaling by weight.",
    )
    ap.add_argument(
        "--algo",
        type=str,
        default="lora",
        choices=["lora", "lokr", "loha"],
        help="Algorithm to use (lora, lokr, loha). Default is lora.",
    )
    ap.add_argument(
        "--min_lokr_factor",
        type=int,
        default=4,
        help="Minimum Factor for LoKr algorithm (applied to High Importance/Misaligned layers).",
    )
    ap.add_argument(
        "--max_lokr_factor",
        type=int,
        default=16,
        help="Maximum Factor for LoKr algorithm (applied to Low Importance/Aligned layers).",
    )
    ap.add_argument(
        "--rslora",
        action="store_true",
        help="Use rsLoRA scaling (alpha = alpha_scale * sqrt(rank)) instead of linear/constant scaling.",
    )
    # DoRA is controlled at runtime via --network_args dora_wd and not persisted in presets
    ap.add_argument(
        "--arch",
        type=str,
        default="sdxl",
        choices=["sdxl", "flux", "qwen", "zimage"],
        help="Target model architecture for naming/patterns",
    )
    args = ap.parse_args()

    # Read weights
    bw_arg = args.block_lr_weights
    try:
        if "," in bw_arg:
            weights = [float(x) for x in bw_arg.split(",") if x.strip()]
        else:
            with open(bw_arg, "r") as f:
                line = f.read().strip()
            weights = [float(x) for x in line.split(",") if x.strip()]
    except Exception:
        weights = []

    # Determine scope
    scope = args.scope
    enable_conv = scope != "full-lin"
    unet_modules = []
    te_modules = []
    include_attn = True
    include_mlp = scope in ("full", "full-lin", "attn-mlp")
    # UNet targets by architecture
    if args.arch == "sdxl":
        if scope in ("full", "full-lin"):
            unet_modules = [
                "Transformer2DModel",
                "ResnetBlock2D",
                "Downsample2D",
                "Upsample2D",
            ]
        else:
            unet_modules = ["Transformer2DModel"]
    elif args.arch == "flux":
        # Flux double/single stream blocks
        unet_modules = ["DoubleStreamBlock", "SingleStreamBlock"]
    elif args.arch == "qwen":
        # Qwen Image transformer block class
        unet_modules = ["QwenImageTransformerBlock"]
    elif args.arch == "zimage":
        unet_modules = ["ZImageTransformerBlock"]
    # TE targets
    if scope in ("full", "full-lin", "attn-mlp", "attn-only"):
        # Include TE modules (disable later by setting LR=0 if needed)
        if scope == "attn-only":
            te_modules = ["CLIPAttention", "CLIPSdpaAttention"]
        else:
            te_modules = ["CLIPAttention", "CLIPSdpaAttention", "CLIPMLP"]
    else:
        te_modules = []

    # Build name map only for transformer attn/MLP paths when included
    name_map = {}
    if include_attn or include_mlp:
        if args.arch == "sdxl":
            name_map = build_name_map(
                weights, args.base_attn, args.base_mlp, args.min_rank, args.max_rank, args.gamma, args.alpha_scale,
                threshold=args.threshold, fixed_rank=args.fixed_rank, algo=args.algo, 
                min_factor=args.min_lokr_factor, max_factor=args.max_lokr_factor, use_rslora=args.rslora
            )
            if not include_mlp:
                name_map = {k: v for k, v in name_map.items() if "ff.net" not in k}
        elif args.arch == "flux":
            # Flux: split NFN weights into DoubleStream (19) and SingleStream (38)
            # and assign per-block ranks to concrete submodules.
            # If insufficient weights are provided, pad with 1.0s.
            dbl_cnt, sng_cnt = 19, 38
            total_needed = dbl_cnt + sng_cnt
            if len(weights) < total_needed:
                # pad to expected length so builder still works
                weights = list(weights) + [1.0] * (total_needed - len(weights))
            w_double = [float(w) for w in weights[:dbl_cnt]]
            w_single = [float(w) for w in weights[dbl_cnt:dbl_cnt + sng_cnt]]

            def scaled(v):
                return (float(v) ** args.gamma) if args.gamma != 1.0 else float(v)

            def make_params(dim, w):
                p = {"algo": args.algo, "dim": dim, "alpha": get_alpha(dim, args.alpha_scale, args.rslora)}
                if args.algo == "lokr" and args.min_lokr_factor != -1:
                    # Simple logic for Flux using same interpolation
                    w_clamped = max(0.1, min(2.0, w))
                    t = (w_clamped - 0.1) / (2.0 - 0.1)
                    t = 1.0 - t # Low W -> Max Factor
                    factor_val = args.min_lokr_factor + t * (args.max_lokr_factor - args.min_lokr_factor)
                    p["factor"] = clamp(factor_val, args.min_lokr_factor, args.max_lokr_factor)
                return p

            # DoubleStreamBlock per-index mapping (Flux uses 'double_blocks')
            for i in range(dbl_cnt):
                w = scaled(w_double[i])
                attn_dim = clamp(args.base_attn * w, args.min_rank, args.max_rank)
                mlp_dim = clamp(args.base_mlp * w, args.min_rank, args.max_rank)
                mod_dim = mlp_dim

                if include_attn:
                    name_map[f"double_blocks.{i}.img_attn.*"] = make_params(attn_dim, w)
                    name_map[f"double_blocks.{i}.txt_attn.*"] = make_params(attn_dim, w)
                if include_mlp:
                    name_map[f"double_blocks.{i}.img_mlp.*"] = make_params(mlp_dim, w)
                    name_map[f"double_blocks.{i}.txt_mlp.*"] = make_params(mlp_dim, w)
                    name_map[f"double_blocks.{i}.img_mod.*"] = make_params(mod_dim, w)
                    name_map[f"double_blocks.{i}.txt_mod.*"] = make_params(mod_dim, w)

            # SingleStreamBlock per-index mapping (Flux uses 'single_blocks')
            for j in range(sng_cnt):
                w = scaled(w_single[j])
                single_dim = clamp(args.base_attn * w, args.min_rank, args.max_rank)
                single_mod_dim = clamp(args.base_mlp * w, args.min_rank, args.max_rank)

                if include_attn or include_mlp:
                    name_map[f"single_blocks.{j}.linear1"] = make_params(single_dim, w)
                    name_map[f"single_blocks.{j}.linear2"] = make_params(single_dim, w)
                name_map[f"single_blocks.{j}.modulation.*"] = make_params(single_mod_dim, w)
        
        elif args.arch == "qwen":
            # Qwen naming: transformer_blocks.<i>
            for i, w in enumerate(weights):
                w_eff = (float(w) ** args.gamma) if args.gamma != 1.0 else float(w)
                attn_rank = clamp(args.base_attn * w_eff, args.min_rank, args.max_rank)
                mlp_rank = clamp(args.base_mlp * w_eff, args.min_rank, args.max_rank)
                
                def make_qwen_params(dim, w):
                    p = {"algo": args.algo, "dim": dim, "alpha": get_alpha(dim, args.alpha_scale, args.rslora)}
                    if args.algo == "lokr" and args.min_lokr_factor != -1:
                        w_clamped = max(0.1, min(2.0, w))
                        t = (w_clamped - 0.1) / (2.0 - 0.1)
                        t = 1.0 - t 
                        factor_val = args.min_lokr_factor + t * (args.max_lokr_factor - args.min_lokr_factor)
                        p["factor"] = clamp(factor_val, args.min_lokr_factor, args.max_lokr_factor)
                    return p

                if include_attn:
                    name_map[f"transformer_blocks.{i}.attn.*"] = make_qwen_params(attn_rank, w)
                if include_mlp:
                    # MLPs (img_mlp, txt_mlp) and Mods (img_mod, txt_mod)
                    for pat in ("img_mlp.*", "txt_mlp.*", "img_mod.*", "txt_mod.*"):
                        name_map[f"transformer_blocks.{i}.{pat}"] = make_qwen_params(mlp_rank, w)
        
        elif args.arch == "zimage":
            # Z-Image logic
            # Assume weights: noise(2) + context(2) + layers(30) = 34
            n_noise, n_ctx, n_layers = 2, 2, 30
            total_needed = n_noise + n_ctx + n_layers
            if len(weights) < total_needed:
                weights = list(weights) + [1.0] * (total_needed - len(weights))
            
            w_noise = weights[:n_noise]
            w_ctx = weights[n_noise:n_noise+n_ctx]
            w_layers = weights[n_noise+n_ctx:]

            def scaled(v):
                return (float(v) ** args.gamma) if args.gamma != 1.0 else float(v)

            def make_z_params(dim, w):
                p = {"algo": args.algo, "dim": dim, "alpha": get_alpha(dim, args.alpha_scale, args.rslora)}
                if args.algo == "lokr" and args.min_lokr_factor != -1:
                    w_clamped = max(0.1, min(2.0, w))
                    t = (w_clamped - 0.1) / (2.0 - 0.1)
                    t = 1.0 - t 
                    factor_val = args.min_lokr_factor + t * (args.max_lokr_factor - args.min_lokr_factor)
                    p["factor"] = clamp(factor_val, args.min_lokr_factor, args.max_lokr_factor)
                return p

            # Noise Refiner
            for i in range(n_noise):
                w = scaled(w_noise[i])
                attn_dim = clamp(args.base_attn * w, args.min_rank, args.max_rank)
                mlp_dim = clamp(args.base_mlp * w, args.min_rank, args.max_rank)
                
                if include_attn:
                    name_map[f"noise_refiner.{i}.attention.*"] = make_z_params(attn_dim, w)
                if include_mlp:
                    name_map[f"noise_refiner.{i}.feed_forward.*"] = make_z_params(mlp_dim, w)

            # Context Refiner
            for i in range(n_ctx):
                w = scaled(w_ctx[i])
                attn_dim = clamp(args.base_attn * w, args.min_rank, args.max_rank)
                mlp_dim = clamp(args.base_mlp * w, args.min_rank, args.max_rank)
                
                if include_attn:
                    name_map[f"context_refiner.{i}.attention.*"] = make_z_params(attn_dim, w)
                if include_mlp:
                    name_map[f"context_refiner.{i}.feed_forward.*"] = make_z_params(mlp_dim, w)

            # Layers
            for i in range(n_layers):
                if i >= len(w_layers): break
                w = scaled(w_layers[i])
                attn_dim = clamp(args.base_attn * w, args.min_rank, args.max_rank)
                mlp_dim = clamp(args.base_mlp * w, args.min_rank, args.max_rank)
                
                if include_attn:
                    name_map[f"layers.{i}.attention.*"] = make_z_params(attn_dim, w)
                if include_mlp:
                    name_map[f"layers.{i}.feed_forward.*"] = make_z_params(mlp_dim, w)

    # Module-level defaults to ensure non-matched modules get reasonable ranks
    module_algo_map = {}
    # Transformer blocks fallback defaults
    if args.arch == "sdxl":
        dim = int(args.base_attn)
        module_algo_map["Transformer2DModel"] = {
            "algo": args.algo,
            "dim": dim,
            "alpha": get_alpha(dim, args.alpha_scale, args.rslora),
        }
    elif args.arch == "flux":
        # Avoid per-module dim/alpha defaults on Flux; rely on name-based rules
        for m in ("DoubleStreamBlock", "SingleStreamBlock"):
            module_algo_map[m] = {"algo": args.algo}
    elif args.arch == "zimage":
        dim = int(args.base_attn)
        module_algo_map["ZImageTransformerBlock"] = {
            "algo": args.algo,
            "dim": dim,
            "alpha": get_alpha(dim, args.alpha_scale, args.rslora),
        }
    # UNet conv/res blocks (only if full/full-lin)
    if scope in ("full", "full-lin"):
        dim = int(args.base_mlp)
        for m in ("ResnetBlock2D", "Downsample2D", "Upsample2D"):
            module_algo_map[m] = {
                "algo": args.algo,
                "dim": dim,
                "alpha": get_alpha(dim, args.alpha_scale, args.rslora),
            }
    # TE modules (if included)
    for m in te_modules:
        base = args.base_attn if "Attention" in m else args.base_mlp
        dim = int(base)
        module_algo_map[m] = {"algo": args.algo, "dim": dim, "alpha": get_alpha(dim, args.alpha_scale, args.rslora)}

    # Target names by arch (fnmatch patterns)
    if args.arch == "sdxl":
        target_names = ["input_blocks.*", "middle_block.*", "output_blocks.*"]
    elif args.arch == "flux":
        # Include container patterns to disable recursive traversal into blocks,
        # then list leaf patterns to create LoRAs directly with per-name ranks.
        target_names = [
            "double_blocks.*",
            "single_blocks.*",
            "double_blocks.*.img_attn.*",
            "double_blocks.*.txt_attn.*",
            "double_blocks.*.img_mlp.*",
            "double_blocks.*.txt_mlp.*",
            "double_blocks.*.img_mod.*",
            "double_blocks.*.txt_mod.*",
            "single_blocks.*.linear1",
            "single_blocks.*.linear2",
            "single_blocks.*.modulation.*",
        ]
    elif args.arch == "zimage":
        target_names = ["noise_refiner.*", "context_refiner.*", "layers.*"]
    else:
        target_names = ["transformer_blocks.*"]

    preset = {
        "enable_conv": enable_conv,
        "use_fnmatch": True,
        "unet_target_module": unet_modules,
        "unet_target_name": target_names,
        "text_encoder_target_module": te_modules,
        "text_encoder_target_name": [],
        "module_algo_map": module_algo_map,
        "name_algo_map": name_map,
    }

    # Do not persist DoRA flag in preset; prefer controlling DoRA at runtime via --network_args dora_wd and not persisted in presets
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        toml.dump(preset, f)
    print(f"Wrote preset to {args.output}")


if __name__ == "__main__":
    main()