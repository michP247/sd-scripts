import json
import os
from dataclasses import replace
from typing import List, Optional, Tuple, Union

import einops
import torch
from accelerate import init_empty_weights  # Note: We'll need to adjust this for TPUs
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import CLIPConfig, CLIPTextModel, T5Config, T5EncoderModel

import torch_xla.core.xla_model as xm

from library import flux_models
from library.utils import load_safetensors

MODEL_VERSION_FLUX_V1 = "flux1"
MODEL_NAME_DEV = "dev"
MODEL_NAME_SCHNELL = "schnell"

def print_num_params(model, name):
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in {name}: {num_params / 1e6:.2f} million")

def analyze_checkpoint_state(ckpt_path: str) -> Tuple[bool, bool, Tuple[int, int], List[str]]:
    """
    Analyzes the state of a checkpoint, calculates the number of blocks, and returns whether it's Diffusers or BFL, dev or schnell.

    Args:
        ckpt_path (str): Path to the checkpoint file or directory.

    Returns:
        Tuple[bool, bool, Tuple[int, int], List[str]]:
            - bool: Flag indicating whether it's Diffusers.
            - bool: Flag indicating whether it's Schnell.
            - Tuple[int, int]: Number of double and single blocks.
            - List[str]: List of keys included in the checkpoint.
    """
    print(f"Checking the state dict: Diffusers or BFL, dev or schnell")

    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "transformer", "diffusion_pytorch_model-00001-of-00003.safetensors")
    if "00001-of-00003" in ckpt_path:
        ckpt_paths = [ckpt_path.replace("00001-of-00003", f"0000{i}-of-00003") for i in range(1, 4)]
    else:
        ckpt_paths = [ckpt_path]

    keys = []
    for ckpt_path in ckpt_paths:
        with safe_open(ckpt_path, framework="pt") as f:
            keys.extend(f.keys())

    if keys[0].startswith("model.diffusion_model."):
        keys = [key.replace("model.diffusion_model.", "") for key in keys]

    is_diffusers = "transformer_blocks.0.attn.add_k_proj.bias" in keys
    is_schnell = not ("guidance_in.in_layer.bias" in keys or "time_text_embed.guidance_embedder.linear_1.bias" in keys)

    if not is_diffusers:
        max_double_block_index = max(
            [int(key.split(".")[1]) for key in keys if key.startswith("double_blocks.") and key.endswith(".img_attn.proj.bias")]
        )
        max_single_block_index = max(
            [int(key.split(".")[1]) for key in keys if key.startswith("single_blocks.") and key.endswith(".modulation.lin.bias")]
        )
    else:
        max_double_block_index = max(
            [
                int(key.split(".")[1])
                for key in keys
                if key.startswith("transformer_blocks.") and key.endswith(".attn.add_k_proj.bias")
            ]
        )
        max_single_block_index = max(
            [
                int(key.split(".")[1])
                for key in keys
                if key.startswith("single_transformer_blocks.") and key.endswith(".attn.to_k.bias")
            ]
        )

    num_double_blocks = max_double_block_index + 1
    num_single_blocks = max_single_block_index + 1

    return is_diffusers, is_schnell, (num_double_blocks, num_single_blocks), ckpt_paths

def load_flow_model(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
) -> Tuple[bool, flux_models.Flux]:
    is_diffusers, is_schnell, (num_double_blocks, num_single_blocks), ckpt_paths = analyze_checkpoint_state(
        ckpt_path
    )
    name = MODEL_NAME_DEV if not is_schnell else MODEL_NAME_SCHNELL

    # Build model on CPU first
    print(f"Building Flux model {name} from {'Diffusers' if is_diffusers else 'BFL'} checkpoint")
    params = flux_models.configs[name].params

    if params.depth != num_double_blocks:
        print(f"Setting the number of double blocks from {params.depth} to {num_double_blocks}")
        params = replace(params, depth=num_double_blocks)
    if params.depth_single_blocks != num_single_blocks:
        print(f"Setting the number of single blocks from {params.depth_single_blocks} to {num_single_blocks}")
        params = replace(params, depth_single_blocks=num_single_blocks)

    # Construct the model on the CPU
    model = flux_models.Flux(params)
    if dtype is not None:
        model = model.to(dtype)

    # Load state_dict
    print(f"Loading state dict from {ckpt_path}")

    # Use torch.load to load the state dict on each process
    if xm.get_ordinal() == 0:
        merged_sd = {}
        for ckpt_path in ckpt_paths:
            sd = load_safetensors(ckpt_path, device="cpu", disable_mmap=disable_mmap, dtype=dtype)
            merged_sd.update(sd)
        # Save merged state_dict to a temporary file
        torch.save(merged_sd, "merged_state_dict.pth")

    # Synchronize all processes
    xm.rendezvous("load_checkpoint")

    # Load the merged state dict on all processes
    merged_sd = torch.load("merged_state_dict.pth", map_location="cpu")

    if is_diffusers:
        print("Converting Diffusers to BFL")
        merged_sd = convert_diffusers_sd_to_bfl(merged_sd, num_double_blocks, num_single_blocks)
        print("Converted Diffusers to BFL")

    for key in list(merged_sd.keys()):
        new_key = key.replace("model.diffusion_model.", "")
        if new_key == key:
            break
        merged_sd[new_key] = merged_sd.pop(key)

    # Wrap with MpModelWrapper before loading state_dict
    model = xm.MpModelWrapper(model)

    info = model.load_state_dict(merged_sd, strict=False)
    print(f"Loaded Flux: {info}")

    # Move the model to the target device after loading the state dict
    #model.to(device)

    print_num_params(model, "Flux")

    return is_schnell, model

def load_ae(
    ckpt_path: str, dtype: torch.dtype, device: Union[str, torch.device], disable_mmap: bool = False
) -> flux_models.AutoEncoder:
    print("Building AutoEncoder")
    # Construct the model on the CPU
    ae = flux_models.AutoEncoder(flux_models.configs[MODEL_NAME_DEV].ae_params).to(dtype)

    print(f"Loading state dict from {ckpt_path}")

    # Load state_dict on the master process
    if xm.get_ordinal() == 0:
        sd = load_safetensors(ckpt_path, device="cpu", disable_mmap=disable_mmap, dtype=dtype)
        # Save state_dict to a temporary file
        torch.save(sd, "ae_state_dict.pth")

    # Synchronize all processes
    xm.rendezvous("load_ae_checkpoint")

    # Load the state dict on all processes
    sd = torch.load("ae_state_dict.pth", map_location="cpu")

    info = ae.load_state_dict(sd, strict=False)
    print(f"Loaded AE: {info}")

    # Move the model to the target device after loading the state dict
    #ae.to(device)
    print_num_params(ae, "AutoEncoder")

    return ae

def load_controlnet(
    ckpt_path: Optional[str],
    is_schnell: bool,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
):
    print("Building ControlNet")
    name = MODEL_NAME_DEV if not is_schnell else MODEL_NAME_SCHNELL
    # Construct the model on the CPU
    controlnet = flux_models.ControlNetFlux(flux_models.configs[name].params).to(dtype)

    if ckpt_path is not None:
        print(f"Loading state dict from {ckpt_path}")

        # Load state_dict on the master process
        if xm.get_ordinal() == 0:
            sd = load_safetensors(ckpt_path, device="cpu", disable_mmap=disable_mmap, dtype=dtype)
            # Save state_dict to a temporary file
            torch.save(sd, "controlnet_state_dict.pth")

        # Synchronize all processes
        xm.rendezvous("load_controlnet_checkpoint")

        # Load the state dict on all processes
        sd = torch.load("controlnet_state_dict.pth", map_location="cpu")

        info = controlnet.load_state_dict(sd, strict=False)
        print(f"Loaded ControlNet: {info}")

    # Move the model to the target device after loading the state dict
    controlnet.to(device)
    print_num_params(controlnet, "ControlNet")

    return controlnet

def load_clip_l(
    ckpt_path: Optional[str],
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> CLIPTextModel:
    print("Building CLIP-L")
    CLIPL_CONFIG = {
        "_name_or_path": "clip-vit-large-patch14/",
        "architectures": ["CLIPModel"],
        "initializer_factor": 1.0,
        "logit_scale_init_value": 2.6592,
        "model_type": "clip",
        "projection_dim": 768,
        # "text_config": {
        "_name_or_path": "",
        "add_cross_attention": False,
        "architectures": None,
        "attention_dropout": 0.0,
        "bad_words_ids": None,
        "bos_token_id": 0,
        "chunk_size_feed_forward": 0,
        "cross_attention_hidden_size": None,
        "decoder_start_token_id": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "dropout": 0.0,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": 2,
        "finetuning_task": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "hidden_act": "quick_gelu",
        "hidden_size": 768,
        "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {"LABEL_0": 0, "LABEL_1": 1},
        "layer_norm_eps": 1e-05,
        "length_penalty": 1.0,
        "max_length": 20,
        "max_position_embeddings": 77,
        "min_length": 0,
        "model_type": "clip_text_model",
        "no_repeat_ngram_size": 0,
        "num_attention_heads": 12,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_hidden_layers": 12,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": 1,
        "prefix": None,
        "problem_type": None,
        "projection_dim": 768,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "sep_token_id": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": True,
        "tokenizer_class": None,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": None,
        "torchscript": False,
        "transformers_version": "4.16.0.dev0",
        "use_bfloat16": False,
        "vocab_size": 49408,
        "hidden_act": "gelu",
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "num_attention_heads": 20,
        "num_hidden_layers": 32,
        # },
        # "text_config_dict": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "projection_dim": 768,
        # },
        # "torch_dtype": "float32",
        # "transformers_version": None,
    }
    config = CLIPConfig(**CLIPL_CONFIG)
    # Construct model on CPU first
    clip = CLIPTextModel(config)

    if state_dict is not None:
        sd = state_dict
    else:
        print(f"Loading state dict from {ckpt_path}")

        # Load state_dict on the master process
        if xm.get_ordinal() == 0:
            sd = load_safetensors(ckpt_path, device="cpu", disable_mmap=disable_mmap, dtype=dtype)
            # Save state_dict to a temporary file
            torch.save(sd, "clip_l_state_dict.pth")

        # Synchronize all processes
        xm.rendezvous("load_clip_l_checkpoint")

        # Load the state dict on all processes
        sd = torch.load("clip_l_state_dict.pth", map_location="cpu")

    # Wrap with MpModelWrapper before loading state_dict
    clip = xm.MpModelWrapper(clip)

    info = clip.load_state_dict(sd, strict=False)
    print(f"Loaded CLIP-L: {info}")

    # Move the model to the target device after loading
    #clip.to(device)
    print_num_params(clip, "CLIP-L")

    return clip

def load_t5xxl(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> T5EncoderModel:
    T5_CONFIG_JSON = """
{
  "architectures": [
    "T5EncoderModel"
  ],
  "classifier_dropout": 0.0,
  "d_ff": 10240,
  "d_kv": 64,
  "d_model": 4096,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 24,
  "num_heads": 64,
  "num_layers": 24,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.41.2",
  "use_cache": true,
  "vocab_size": 32128
}
"""
    config = json.loads(T5_CONFIG_JSON)
    config = T5Config(**config)
    # Construct model on CPU first
    t5xxl = T5EncoderModel(config)

    if state_dict is not None:
        sd = state_dict
    else:
        print(f"Loading state dict from {ckpt_path}")

        # Load state_dict on the master process
        if xm.get_ordinal() == 0:
            sd = load_safetensors(ckpt_path, device="cpu", disable_mmap=disable_mmap, dtype=dtype)
            # Save state_dict to a temporary file
            torch.save(sd, "t5xxl_state_dict.pth")

        # Synchronize all processes
        xm.rendezvous("load_t5xxl_checkpoint")

        # Load the state dict on all processes
        sd = torch.load("t5xxl_state_dict.pth", map_location="cpu")

    # Wrap with MpModelWrapper before loading state_dict
    t5xxl = xm.MpModelWrapper(t5xxl)

    info = t5xxl.load_state_dict(sd, strict=False)
    print(f"Loaded T5xxl: {info}")

    # Move the model to the target device after loading
    #t5xxl.to(device)
    print_num_params(t5xxl, "T5XXL")

    return t5xxl

def get_t5xxl_actual_dtype(t5xxl: T5EncoderModel) -> torch.dtype:
    # nn.Embedding is the first layer, but it could be casted to bfloat16 or float32
    return t5xxl.encoder.block[0].layer[0].SelfAttention.q.weight.dtype

def prepare_img_ids(batch_size: int, packed_latent_height: int, packed_latent_width: int):
    img_ids = torch.zeros(packed_latent_height, packed_latent_width, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_latent_height)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_latent_width)[None, :]
    img_ids = einops.repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    return img_ids

def unpack_latents(x: torch.Tensor, packed_latent_height: int, packed_latent_width: int) -> torch.Tensor:
    """
    x: [b (h w) (c ph pw)] -> [b c (h ph) (w pw)], ph=2, pw=2
    """
    x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)
    return x

def pack_latents(x: torch.Tensor) -> torch.Tensor:
    """
    x: [b c (h ph) (w pw)] -> [b (h w) (c ph pw)], ph=2, pw=2
    """
    x = einops.rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    return x

# region Diffusers

NUM_DOUBLE_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38

BFL_TO_DIFFUSERS_MAP = {
    "time_in.in_layer.weight": ["time_text_embed.timestep_embedder.linear_1.weight"],
    "time_in.in_layer.bias": ["time_text_embed.timestep_embedder.linear_1.bias"],
    "time_in.out_layer.weight": ["time_text_embed.timestep_embedder.linear_2.weight"],
    "time_in.out_layer.bias": ["time_text_embed.timestep_embedder.linear_2.bias"],
    "vector_in.in_layer.weight": ["time_text_embed.text_embedder.linear_1.weight"],
    "vector_in.in_layer.bias": ["time_text_embed.text_embedder.linear_1.bias"],
    "vector_in.out_layer.weight": ["time_text_embed.text_embedder.linear_2.weight"],
    "vector_in.out_layer.bias": ["time_text_embed.text_embedder.linear_2.bias"],
    "guidance_in.in_layer.weight": ["time_text_embed.guidance_embedder.linear_1.weight"],
    "guidance_in.in_layer.bias": ["time_text_embed.guidance_embedder.linear_1.bias"],
    "guidance_in.out_layer.weight": ["time_text_embed.guidance_embedder.linear_2.weight"],
    "guidance_in.out_layer.bias": ["time_text_embed.guidance_embedder.linear_2.bias"],
    "txt_in.weight": ["context_embedder.weight"],
    "txt_in.bias": ["context_embedder.bias"],
    "img_in.weight": ["x_embedder.weight"],
    "img_in.bias": ["x_embedder.bias"],
    "double_blocks.().img_mod.lin.weight": ["norm1.linear.weight"],
    "double_blocks.().img_mod.lin.bias": ["norm1.linear.bias"],
    "double_blocks.().txt_mod.lin.weight": ["norm1_context.linear.weight"],
    "double_blocks.().txt_mod.lin.bias": ["norm1_context.linear.bias"],
    "double_blocks.().img_attn.qkv.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"],
    "double_blocks.().img_attn.qkv.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias"],
    "double_blocks.().txt_attn.qkv.weight": ["attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight"],
    "double_blocks.().txt_attn.qkv.bias": ["attn.add_q_proj.bias", "attn.add_k_proj.bias", "attn.add_v_proj.bias"],
    "double_blocks.().img_attn.norm.query_norm.scale": ["attn.norm_q.weight"],
    "double_blocks.().img_attn.norm.key_norm.scale": ["attn.norm_k.weight"],
    "double_blocks.().txt_attn.norm.query_norm.scale": ["attn.norm_added_q.weight"],
    "double_blocks.().txt_attn.norm.key_norm.scale": ["attn.norm_added_k.weight"],
    "double_blocks.().img_mlp.0.weight": ["ff.net.0.proj.weight"],
    "double_blocks.().img_mlp.0.bias": ["ff.net.0.proj.bias"],
    "double_blocks.().img_mlp.2.weight": ["ff.net.2.weight"],
    "double_blocks.().img_mlp.2.bias": ["ff.net.2.bias"],
    "double_blocks.().txt_mlp.0.weight": ["ff_context.net.0.proj.weight"],
    "double_blocks.().txt_mlp.0.bias": ["ff_context.net.0.proj.bias"],
    "double_blocks.().txt_mlp.2.weight": ["ff_context.net.2.weight"],
    "double_blocks.().txt_mlp.2.bias": ["ff_context.net.2.bias"],
    "double_blocks.().img_attn.proj.weight": ["attn.to_out.0.weight"],
    "double_blocks.().img_attn.proj.bias": ["attn.to_out.0.bias"],
    "double_blocks.().txt_attn.proj.weight": ["attn.to_add_out.weight"],
    "double_blocks.().txt_attn.proj.bias": ["attn.to_add_out.bias"],
    "single_blocks.().modulation.lin.weight": ["norm.linear.weight"],
    "single_blocks.().modulation.lin.bias": ["norm.linear.bias"],
    "single_blocks.().linear1.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "proj_mlp.weight"],
    "single_blocks.().linear1.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias", "proj_mlp.bias"],
    "single_blocks.().linear2.weight": ["proj_out.weight"],
    "single_blocks.().norm.query_norm.scale": ["attn.norm_q.weight"],
    "single_blocks.().norm.key_norm.scale": ["attn.norm_k.weight"],
    "single_blocks.().linear2.weight": ["proj_out.weight"],
    "single_blocks.().linear2.bias": ["proj_out.bias"],
    "final_layer.linear.weight": ["proj_out.weight"],
    "final_layer.linear.bias": ["proj_out.bias"],
    "final_layer.adaLN_modulation.1.weight": ["norm_out.linear.weight"],
    "final_layer.adaLN_modulation.1.bias": ["norm_out.linear.bias"],
}

def make_diffusers_to_bfl_map(num_double_blocks: int, num_single_blocks: int) -> dict[str, tuple[int, str]]:
    # make reverse map from diffusers map
    diffusers_to_bfl_map = {}  # key: diffusers_key, value: (index, bfl_key)
    for b in range(num_double_blocks):
        for key, weights in BFL_TO_DIFFUSERS_MAP.items():
            if key.startswith("double_blocks."):
                block_prefix = f"transformer_blocks.{b}."
                for i, weight in enumerate(weights):
                    diffusers_to_bfl_map[f"{block_prefix}{weight}"] = (i, key.replace("()", f"{b}"))
    for b in range(num_single_blocks):
        for key, weights in BFL_TO_DIFFUSERS_MAP.items():
            if key.startswith("single_blocks."):
                block_prefix = f"single_transformer_blocks.{b}."
                for i, weight in enumerate(weights):
                    diffusers_to_bfl_map[f"{block_prefix}{weight}"] = (i, key.replace("()", f"{b}"))
    for key, weights in BFL_TO_DIFFUSERS_MAP.items():
        if not (key.startswith("double_blocks.") or key.startswith("single_blocks.")):
            for i, weight in enumerate(weights):
                diffusers_to_bfl_map[weight] = (i, key)
    return diffusers_to_bfl_map

def convert_diffusers_sd_to_bfl(
    diffusers_sd: dict[str, torch.Tensor], num_double_blocks: int = NUM_DOUBLE_BLOCKS, num_single_blocks: int = NUM_SINGLE_BLOCKS
) -> dict[str, torch.Tensor]:
    diffusers_to_bfl_map = make_diffusers_to_bfl_map(num_double_blocks, num_single_blocks)

    # iterate over three safetensors files to reduce memory usage
    flux_sd = {}
    for diffusers_key, tensor in diffusers_sd.items():
        if diffusers_key in diffusers_to_bfl_map:
            index, bfl_key = diffusers_to_bfl_map[diffusers_key]
            if bfl_key not in flux_sd:
                flux_sd[bfl_key] = []
            flux_sd[bfl_key].append((index, tensor))
        else:
            print(f"Error: Key not found in diffusers_to_bfl_map: {diffusers_key}")
            raise KeyError(f"Key not found in diffusers_to_bfl_map: {diffusers_key}")

    # concat tensors if multiple tensors are mapped to a single key, sort by index
    for key, values in flux_sd.items():
        if len(values) == 1:
            flux_sd[key] = values[0][1]
        else:
            flux_sd[key] = torch.cat([value[1] for value in sorted(values, key=lambda x: x[0])])

    # special case for final_layer.adaLN_modulation.1.weight and final_layer.adaLN_modulation.1.bias
    def swap_scale_shift(weight):
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        return new_weight

    if "final_layer.adaLN_modulation.1.weight" in flux_sd:
        flux_sd["final_layer.adaLN_modulation.1.weight"] = swap_scale_shift(flux_sd["final_layer.adaLN_modulation.1.weight"])
    if "final_layer.adaLN_modulation.1.bias" in flux_sd:
        flux_sd["final_layer.adaLN_modulation.1.bias"] = swap_scale_shift(flux_sd["final_layer.adaLN_modulation.1.bias"])

    return flux_sd