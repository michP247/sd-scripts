#!/usr/bin/env python3
"""
Convert incorrectly saved LyCORIS Full (algo=full) weights that use
".weight"/".bias" per target module into the correct
".diff"/".diff_b" keys expected by loaders.

Usage:
  python tools/convert_full_lora_weights.py --input /path/in.lora.safetensors \
      --output /path/out_fixed.safetensors

Notes:
- This assumes the current tensors correspond to adapter deltas (not merged
  base weights). With the DeepSpeed-safe FullModule, the adapter parameters
  are stored separately and equal to the diff directly.
- Metadata is preserved and augmented when possible.
"""
import argparse
from collections import OrderedDict

from safetensors.torch import load_file, save_file, safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input .safetensors file")
    ap.add_argument("--output", required=True, help="output .safetensors file")
    args = ap.parse_args()

    # Load tensors and metadata
    tensors = load_file(args.input)
    meta = {}
    try:
        with safe_open(args.input, framework="pt", device="cpu") as f:
            meta = dict(f.metadata() or {})
    except Exception:
        meta = {}

    # Rename keys: <name>.weight -> <name>.diff, <name>.bias -> <name>.diff_b
    out = OrderedDict()
    num_weight = num_bias = 0
    for k, v in tensors.items():
        if k.endswith(".weight"):
            out[k[:-7] + ".diff"] = v
            num_weight += 1
        elif k.endswith(".bias"):
            out[k[:-5] + ".diff_b"] = v
            num_bias += 1
        else:
            # keep other keys as-is (e.g., alpha, dora_scale, etc.)
            out[k] = v

    if not num_weight and not num_bias:
        print("No '.weight' or '.bias' keys found; nothing to convert.")
        return

    # Ensure minimal LyCORIS metadata hints exist
    meta = dict(meta or {})
    meta.setdefault("ss_network_module", "lycoris.kohya")
    # Leave ss_v* flags untouched if present; many loaders only rely on keys

    save_file(out, args.output, metadata=meta)
    print(f"Converted: weights={num_weight}, bias={num_bias}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

