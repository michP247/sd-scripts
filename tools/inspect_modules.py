import torch
from diffusers import UNet2DConditionModel
import re

# Mock SDXL config (simplified)
config = {
    "sample_size": 128,
    "in_channels": 4,
    "out_channels": 4,
    "layers_per_block": 2,
    "block_out_channels": (320, 640, 1280),
    "attention_head_dim": (5, 10, 20),
    "use_linear_projection": True,
    "class_embed_type": "timestep",
    "transformer_layers_per_block": (1, 2, 10),
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 256,
    "projection_class_embeddings_input_dim": 2816,
    "cross_attention_dim": 2048,
}

# Create a dummy UNet structure to inspect names
# Note: Loading the full model takes time/memory, so we'll simulate the structure or just match strings
# The user patterns are:
# input_blocks.{i}.1.transformer_blocks.*.attn*
# input_blocks.{i}.1.transformer_blocks.*.ff.net.*

patterns = [
    r"input_blocks\.\d+\.1\.transformer_blocks\..*\.attn.*",
    r"input_blocks\.\d+\.1\.transformer_blocks\..*\.ff\.net\..*",
    r"middle_block\.\d+\.transformer_blocks\..*\.attn.*",
    r"middle_block\.\d+\.transformer_blocks\..*\.ff\.net\..*",
    r"output_blocks\.\d+\.1\.transformer_blocks\..*\.attn.*",
    r"output_blocks\.\d+\.1\.transformer_blocks\..*\.ff\.net\..*",
]

# Common SDXL modules that might be missed:
# - norm1, norm2, norm3 (if targeting linear, but usually ignored by lycoris default unless enabled)
# - time_emb_proj (in ResNets)
# - conv1, conv2 (in ResNets)
# - proj_in, proj_out (usually covered by attn*)

print("Analyzing potential misses...")
print("If 'attn*' matches 'to_q', 'to_k', 'to_v', 'to_out.0', what is left?")

# Let's assume the user matched "attn*" which covers:
# - to_q
# - to_k
# - to_v
# - to_out.0

# And "ff.net.*" which covers:
# - net.0.proj
# - net.2

# What about "norm"? 
# What about "time_emb_proj"?

print("Checking standard transformer block structure:")
modules = [
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
    "ff.net.0.proj", "ff.net.2",
    "norm1", "norm2", "norm3"
]

misses = []
for m in modules:
    # Prepend a standard prefix
    full_name = f"input_blocks.4.1.transformer_blocks.0.{m}"
    
    matched = False
    for p in patterns:
        if re.match(p, full_name):
            matched = True
            break
    
    if not matched:
        misses.append(m)

print("Modules in a Transformer Block NOT matched by current patterns:")
for m in misses:
    print(f" - {m}")
