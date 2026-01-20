import os
from typing import List, Optional, Union
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from library import train_util
from library.strategy_base import TokenizeStrategy, TextEncodingStrategy, TextEncoderOutputsCachingStrategy

# Define default IDs or let user provide them
# QWEN_ID = "Qwen/Qwen2.5-7B" # Placeholder
# SIGLIP_ID = "google/siglip-so400m-patch14-384" # Placeholder

class ZImageTokenizeStrategy(TokenizeStrategy):
    def __init__(self, qwen_id: str, max_length: int = 512, tokenizer_cache_dir: Optional[str] = None) -> None:
        self.max_length = max_length
        self.qwen_tokenizer = self._load_tokenizer(AutoTokenizer, qwen_id, tokenizer_cache_dir=tokenizer_cache_dir, trust_remote_code=True)
        # SigLip usually uses a processor (image processor + tokenizer), but here likely just visual signals?
        # Or does SigLip take text too? SigLip is CLIP-like, so it has a text encoder.
        # "Visual Encoder: It uses SigLip-2 Embedding processed by a Semantic Processor" -> Usually implies Image input.
        # "Z-Image-Edit (Image Editing): Inputs: An input image (via VAE/SigLip path)..."
        # "Z-Image (Text-to-Image): Inputs: Text prompt... via Qwen3-4B"
        # So for T2I training, do we use SigLip?
        # The architecture diagram shows SigLip path for "Z-Image-Edit".
        # For pure T2I, maybe only Qwen is used?
        # Or maybe SigLip is used for "Image Latents" or some conditioning?
        # "Visual Encoder: It uses SigLip-2 Embedding processed by a Semantic Processor... SigLip... strong vision-language model... to understand input images for editing tasks."
        # If we are training T2I, maybe we don't need SigLip unless we are doing image-variation or editing training.
        # But if the user wants "Z-image Turbo" training, they might want to train the whole pipeline.
        
        # Let's assume for standard T2I, we mainly need Qwen.
        # If the model expects SigLip embeddings (even empty ones?), we need to handle that.
        pass

    def tokenize(self, text: Union[str, List[str]]) -> dict:
        text = [text] if isinstance(text, str) else text
        # Tokenize with Qwen
        # Qwen tokenizer might need specific handling (e.g. pad_token)
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

        tokens = self.qwen_tokenizer(
            text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }

class ZImageTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self) -> None:
        pass

    def encode_tokens(
        self,
        tokenize_strategy: "ZImageTokenizeStrategy",
        text_encoders: List[torch.nn.Module],
        tokens_and_masks: dict,
        cached_text_encoder_outputs: Optional[dict] = None,
    ) -> List[torch.Tensor]:
        
        qwen_model = text_encoders[0]
        
        input_ids = tokens_and_masks["input_ids"].to(qwen_model.device)
        attention_mask = tokens_and_masks["attention_mask"].to(qwen_model.device)
        
        with torch.no_grad():
            # Get Qwen embeddings
            # Assuming we want the last hidden state
            outputs = qwen_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state
        
        return [hidden_states, attention_mask]

# Define Caching Strategy similar to Flux
