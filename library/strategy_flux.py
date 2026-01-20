import os
import glob
from typing import Any, List, Optional, Tuple, Union
import torch
import numpy as np
from transformers import CLIPTokenizer, T5TokenizerFast

from library import flux_utils, train_util
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


CLIP_L_TOKENIZER_ID = "openai/clip-vit-large-patch14"
T5_XXL_TOKENIZER_ID = "google/t5-v1_1-xxl"


class FluxTokenizeStrategy(TokenizeStrategy):
    def __init__(self, t5xxl_max_length: int = 512, tokenizer_cache_dir: Optional[str] = None) -> None:
        self.t5xxl_max_length = t5xxl_max_length
        self.clip_l = self._load_tokenizer(CLIPTokenizer, CLIP_L_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
        self.t5xxl = self._load_tokenizer(T5TokenizerFast, T5_XXL_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)

    def tokenize(self, text: Union[str, List[str]]) -> dict:
        text = [text] if isinstance(text, str) else text

        l_tokens = self.clip_l(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        t5_tokens = self.t5xxl(text, max_length=self.t5xxl_max_length, padding="max_length", truncation=True, return_tensors="pt")

        # The text encoding strategy expects "g_tokens" for the T5 part.
        return {
            "l_tokens": l_tokens["input_ids"],
            "g_tokens": t5_tokens["input_ids"],
            "t5_attn_mask": t5_tokens["attention_mask"],
        }


class FluxTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self, apply_t5_attn_mask: Optional[bool] = None) -> None:
        """
        Args:
            apply_t5_attn_mask: Default value for apply_t5_attn_mask.
        """
        self.apply_t5_attn_mask = apply_t5_attn_mask

    def encode_tokens(

        self,

        tokenize_strategy: "FluxTokenizeStrategy",

        text_encoders: List[torch.nn.Module],

        tokens_and_masks: dict,

        cached_text_encoder_outputs: Optional[dict] = None,

    ) -> List[torch.Tensor]:

        clip_l = text_encoders[0] if text_encoders is not None and len(text_encoders) > 0 else None

        t5xxl = text_encoders[1] if text_encoders is not None and len(text_encoders) > 1 else None



        # text_encoders are not prepared by accelerator, so we need to manually move to device

        if tokens_and_masks.get("l_tokens") is not None:

            l_tokens = tokens_and_masks["l_tokens"]

            if clip_l is not None:

                l_pooled = clip_l(l_tokens.to(clip_l.device))["pooler_output"]

            elif cached_text_encoder_outputs is not None:

                l_pooled = cached_text_encoder_outputs[0]  # use cached l_pooled

            else:

                l_pooled = None

        else:

            l_pooled = None



        g_tokens = tokens_and_masks.get("g_tokens")

        if g_tokens is not None:

            if t5xxl is not None:

                # T5XXL

                if self.apply_t5_attn_mask:

                    g_attn_mask = torch.ones_like(g_tokens, device=g_tokens.device)

                else:

                    g_attn_mask = None

                t5_out = t5xxl(g_tokens.to(t5xxl.device), attention_mask=g_attn_mask)["last_hidden_state"]

            elif cached_text_encoder_outputs is not None:

                # use cached t5_out

                t5_out = cached_text_encoder_outputs[1]

                g_attn_mask = cached_text_encoder_outputs[3]

            else:

                t5_out = None

                g_attn_mask = None

        else:

            t5_out = None

            g_attn_mask = None



        if tokens_and_masks.get("txt_ids") is not None:

            txt_ids = tokens_and_masks["txt_ids"]

        else:

            if cached_text_encoder_outputs is not None:

                txt_ids = cached_text_encoder_outputs[2]  # use cached txt_ids

            elif g_tokens is not None:

                bs, seq_len = g_tokens.shape

                # txt_ids are positional embeddings for text tokens

                # 3 channels: (is_text, y_pos, x_pos)

                txt_ids = torch.zeros(bs, seq_len, 3, dtype=torch.long, device=g_tokens.device)

                txt_ids[:, :, 0] = 1  # is_text=1

                txt_ids[:, :, 1] = 0  # y_pos=0

                txt_ids[:, :, 2] = torch.arange(seq_len, device=g_tokens.device).unsqueeze(0)

            else:

                txt_ids = None



        return [l_pooled, t5_out, txt_ids, g_attn_mask]


class FluxTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    FLUX_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_flux_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        apply_t5_attn_mask: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)
        self.apply_t5_attn_mask = apply_t5_attn_mask

        self.warn_fp8_weights = False

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + FluxTextEncoderOutputsCachingStrategy.FLUX_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            has_l_pooled = "l_pooled" in npz
            if not self.is_partial and not has_l_pooled:  # full cache needs l_pooled
                return False
            if self.is_partial and has_l_pooled:  # partial cache must not have l_pooled
                return False

            if "t5_out" not in npz:
                return False
            if "txt_ids" not in npz:
                return False
            if "t5_attn_mask" not in npz:
                return False
            if "apply_t5_attn_mask" not in npz:
                return False
            # npz_apply_t5_attn_mask = npz["apply_t5_attn_mask"]
            # if npz_apply_t5_attn_mask != self.apply_t5_attn_mask:
            #     return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        try:
            #logger.info(f"[DEBUG] Attempting to load NPZ from: {npz_path}")
            data = np.load(npz_path)
            #logger.info(f"[DEBUG] NPZ loaded successfully. Keys: {list(data.keys())}")

            l_pooled = data["l_pooled"] if "l_pooled" in data else None
            t5_out = data["t5_out"]
            txt_ids = data["txt_ids"]
            t5_attn_mask = data["t5_attn_mask"]
            # apply_t5_attn_mask should be same as self.apply_t5_attn_mask

            #logger.info(f"[DEBUG] Loaded from {npz_path}: l_pooled shape={l_pooled.shape if l_pooled is not None else 'None'}, t5_out shape={t5_out.shape}, txt_ids shape={txt_ids.shape}, t5_attn_mask shape={t5_attn_mask.shape}")
            result = [l_pooled, t5_out, txt_ids, t5_attn_mask]
            #logger.info(f"[DEBUG] Returning result list with {len(result)} elements")
            return result
        except FileNotFoundError:
            logger.error(f"[ERROR] Text encoder outputs NPZ file not found: {npz_path}")
            return None
        except KeyError as e:
            logger.error(f"[ERROR] Missing key in NPZ file {npz_path}: {e}")
            logger.error(f"[ERROR] Available keys were: {list(data.keys()) if 'data' in locals() else 'NPZ not loaded'}")
            return None
        except Exception as e:
            logger.error(f"[ERROR] Error loading text encoder outputs from {npz_path}: {e}")
            import traceback
            logger.error(f"[ERROR] Traceback: {traceback.format_exc()}")
            return None

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        if not self.warn_fp8_weights:
            if models[1] is not None and flux_utils.get_t5xxl_actual_dtype(models[1]) == torch.float8_e4m3fn:
                logger.warning(
                    "T5 model is using fp8 weights for caching. This may affect the quality of the cached outputs."
                    " / T5モデルはfp8の重みを使用しています。これはキャッシュの品質に影響を与える可能性があります。"
                )
            self.warn_fp8_weights = True

        flux_text_encoding_strategy: FluxTextEncodingStrategy = text_encoding_strategy
        captions = [info.caption for info in infos]

        tokens_and_masks = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            # attn_mask is applied in text_encoding_strategy.encode_tokens if apply_t5_attn_mask is True
            l_pooled, t5_out, txt_ids, _ = flux_text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens_and_masks)

        if l_pooled is not None:
            if l_pooled.dtype == torch.bfloat16:
                l_pooled = l_pooled.float()
            l_pooled = l_pooled.cpu().numpy()

        if t5_out.dtype == torch.bfloat16:
            t5_out = t5_out.float()
        if txt_ids.dtype == torch.bfloat16:
            txt_ids = txt_ids.float()

        t5_out = t5_out.cpu().numpy()
        txt_ids = txt_ids.cpu().numpy()
        t5_attn_mask = tokens_and_masks["t5_attn_mask"].cpu().numpy()

        for i, info in enumerate(infos):
            t5_out_i = t5_out[i]
            txt_ids_i = txt_ids[i]
            t5_attn_mask_i = t5_attn_mask[i]
            apply_t5_attn_mask_i = self.apply_t5_attn_mask

            if self.cache_to_disk:
                save_kwargs = {
                    "t5_out": t5_out_i,
                    "txt_ids": txt_ids_i,
                    "t5_attn_mask": t5_attn_mask_i,
                    "apply_t5_attn_mask": apply_t5_attn_mask_i,
                }
                if not self.is_partial:
                    save_kwargs["l_pooled"] = l_pooled[i]

                np.savez(info.text_encoder_outputs_npz, **save_kwargs)

            else:
                # it's fine that attn mask is not None. it's overwritten before calling the model if necessary
                if self.is_partial:
                    info.text_encoder_outputs = (None, t5_out_i, txt_ids_i, t5_attn_mask_i)
                else:
                    info.text_encoder_outputs = (l_pooled[i], t5_out_i, txt_ids_i, t5_attn_mask_i)


class FluxLatentsCachingStrategy(LatentsCachingStrategy):
    FLUX_LATENTS_NPZ_SUFFIX = "_flux.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return FluxLatentsCachingStrategy.FLUX_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + FluxLatentsCachingStrategy.FLUX_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)  # support multi-resolution

    # TODO remove circular dependency for ImageInfo
    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor).to("cpu")
        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)


if __name__ == "__main__":
    # test code for FluxTokenizeStrategy
    # tokenizer = sd3_models.SD3Tokenizer()
    strategy = FluxTokenizeStrategy(256)
    text = "hello world"

    l_tokens, g_tokens, t5_tokens = strategy.tokenize(text)
    # print(l_tokens.shape)
    print(l_tokens)
    print(g_tokens)
    print(t5_tokens)

    texts = ["hello world", "the quick brown fox jumps over the lazy dog"]
    l_tokens_2 = strategy.clip_l(texts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
    g_tokens_2 = strategy.clip_g(texts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
    t5_tokens_2 = strategy.t5xxl(
        texts, max_length=strategy.t5xxl_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    print(l_tokens_2)
    print(g_tokens_2)
    print(t5_tokens_2)

    # compare
    print(torch.allclose(l_tokens, l_tokens_2["input_ids"][0]))
    print(torch.allclose(g_tokens, g_tokens_2["input_ids"][0]))
    print(torch.allclose(t5_tokens, t5_tokens_2["input_ids"][0]))

    text = ",".join(["hello world! this is long text"] * 50)
    l_tokens, g_tokens, t5_tokens = strategy.tokenize(text)
    print(l_tokens)
    print(g_tokens)
    print(t5_tokens)

    print(f"model max length l: {strategy.clip_l.model_max_length}")
    print(f"model max length g: {strategy.clip_g.model_max_length}")
    print(f"model max length t5: {strategy.t5xxl.model_max_length}")
