
import sys
import os
import argparse
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from library import config_util, train_util, strategy_base, strategy_flux
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.utils import setup_logging

setup_logging()
import logging
logger = logging.getLogger(__name__)


def main():
    """
    Minimal script to test the dataset's __getitem__ logic for partial caching.
    """
    # 1. Mock arguments to simulate the training script environment
    args = argparse.Namespace()
    args.dataset_config = "/home/en1u2/AI-Stuff/musubi-tuner/src/musubi_tuner/hmi_data.toml"
    args.batch_size = 1
    args.cache_latents_to_disk = True
    args.cache_text_encoder_outputs_to_disk = True
    args.text_encoder_batch_size = 1
    args.skip_cache_check = False
    args.apply_t5_attn_mask = False
    args.resolution = "1024,1024"
    args.bucket_reso_steps = 64
    args.min_bucket_reso = 256
    args.max_bucket_reso = 1024
    args.bucket_no_upscale = False
    args.prior_loss_weight = 1.0
    args.network_multiplier = 1.0
    args.debug_dataset = False
    
    # Unused args, but required by the dataset preparation logic
    args.train_data_dir = None
    args.reg_data_dir = None
    args.in_json = None
    args.masked_loss = False
    args.shuffle_caption = False
    args.keep_tokens = 0
    args.token_warmup_step = 0
    args.caption_dropout_rate = 0.0
    args.caption_tag_dropout_rate = 0.0


    print("--- Setting up strategies for partial caching test ---")
    # 2. Set up strategies to mimic training clip_l with cached t5
    caching_strategy = strategy_flux.FluxTextEncoderOutputsCachingStrategy(
        cache_to_disk=True,
        batch_size=args.text_encoder_batch_size,
        skip_disk_cache_validity_check=args.skip_cache_check,
        is_partial=True,  # Key for our use case: indicates clip_l is trained
        apply_t5_attn_mask=args.apply_t5_attn_mask,
    )
    strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(caching_strategy)
    
    # Mock other strategies that are needed by the dataset
    strategy_base.TokenizeStrategy.set_strategy(None)
    strategy_base.LatentsCachingStrategy.set_strategy(strategy_flux.FluxLatentsCachingStrategy(True, 1, False))


    print("\n--- Loading dataset from config ---")
    # 3. Load the dataset using the same logic as the training script
    blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
    user_config = config_util.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args)
    train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)[0]

    # 4. Prepare the dataset (make buckets, etc.)
    train_dataset_group.prepare(args)
    train_dataset_group.set_current_strategies()


    print("\n--- Attempting to fetch one item from the dataset ---")
    # 5. Fetch a single item to test the __getitem__ logic
    try:
        # The collate function is where the final batch is assembled.
        # To test the full logic, we get one item and then manually collate it.
        single_item = train_dataset_group[0]
        
        collator = train_util.collator_class(0, 0, train_dataset_group)
        batch = collator([single_item])

        print("\n--- Successfully fetched and collated item! ---")
        
        print("\nBatch keys:", batch.keys())
        
        te_outputs = batch.get("text_encoder_outputs")
        if te_outputs is not None:
            print("\n'text_encoder_outputs' content:")
            print(f"  Type: {type(te_outputs)}")
            print(f"  Length: {len(te_outputs)}")
            for i, output in enumerate(te_outputs):
                if output is None:
                    print(f"  Item {i}: None")
                else:
                    print(f"  Item {i}: Tensor with shape {output.shape} and dtype {output.dtype}")
        else:
            print("\n'text_encoder_outputs' is None or not present.")

    except Exception as e:
        print("\n--- An error occurred while fetching/collating an item ---")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
