#!/usr/bin/env python3
"""
Fix DoRA (dora_scale) tensor orientations in a LyCORIS .safetensors file
to improve compatibility with merge tools that expect out-dim scaling of
shape (out_dim, 1) for linear layers.

Strategy:
- For each lora module prefix, read lora_up/down to infer out_dim/in_dim.
- If a dora_scale exists and is 2D, coerce it to shape (out_dim, 1) when
  it is found as (1, out_dim) or any other 2D shape of size out_dim.
- Leaves conv dora_scales (with >2 dims) untouched.

Usage:
  python tools/fix_dora_scale_orientation.py \
    --input /path/in.safetensors --output /path/out_fixed.safetensors
"""
import argparse
from collections import defaultdict
from safetensors.torch import safe_open, load_file, save_file
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    state = load_file(args.input)
    # Group keys by module prefix
    by_mod = defaultdict(dict)
    for k, v in state.items():
        if "." not in k:
            continue
        mod, tail = k.split(".", 1)
        by_mod[mod][tail] = v

    fixes = 0
    for mod, entries in by_mod.items():
        out_dim = None
        in_dim = None
        # Infer dims from lora_up/down if present
        up = entries.get("lora_up.weight")
        down = entries.get("lora_down.weight")
        if up is not None and up.ndim >= 2:
            out_dim = up.shape[0]
        if down is not None and down.ndim >= 2:
            in_dim = down.shape[1]

        ds_key = "dora_scale"
        if ds_key not in entries:
            continue
        ds = entries[ds_key]
        # Only attempt fix for 2D scales (linear layers)
        if ds.ndim != 2 or out_dim is None:
            continue

        target = (out_dim, 1)

        # If already correct shape, skip
        if tuple(ds.shape) == target:
            continue

        # Common case: saved as (1, out_dim) — transpose to (out_dim, 1)
        if ds.shape[0] == 1 and ds.shape[1] == out_dim:
            state[f"{mod}.{ds_key}"] = ds.T.contiguous()
            fixes += 1
            continue

        # If one of dims matches out_dim, coerce to (out_dim, 1)
        if ds.shape[0] == out_dim or ds.shape[1] == out_dim:
            # Collapse to column vector of length out_dim
            col = ds.reshape(-1)
            if col.numel() == out_dim:
                state[f"{mod}.{ds_key}"] = col.reshape(out_dim, 1).contiguous()
                fixes += 1
                continue

        # Otherwise leave as-is

    # Preserve metadata
    meta = {}
    try:
        with safe_open(args.input, framework="pt", device="cpu") as f:
            meta = dict(f.metadata() or {})
    except Exception:
        meta = {}

    save_file(state, args.output, metadata=meta)
    print(f"Fixed {fixes} dora_scale tensors. Saved to {args.output}")


if __name__ == "__main__":
    main()

