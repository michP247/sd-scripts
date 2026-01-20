#!/usr/bin/env python3
"""
Test script to verify selective text encoder caching works correctly.
This script simulates the user's training command with text_encoder_lr set to "2.5e-4 2.5e-5 0"
which should result in:
- Training clip_l with learning rate 2.5e-4
- Training clip_g with learning rate 2.5e-5
- Not training the third value (0 is for U-Net)
- Caching only clip_g outputs since its learning rate is 0
"""

import argparse
import sys
import os

# Add the sd-scripts directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdxl_train_network import SdxlNetworkTrainer

def test_selective_te_caching():
    # Create a mock args object similar to what would be parsed from the user's command
    class MockArgs:
        def __init__(self):
            # Text encoder learning rates: clip_l=2.5e-4, clip_g=2.5e-5, unet=0
            self.text_encoder_lr = [2.5e-4, 2.5e-5, 0]
            self.cache_text_encoder_outputs = True
            self.cache_text_encoder_outputs_to_disk = True
            self.skip_cache_check = False
            self.weighted_captions = False
            self.lowram = False
    
    args = MockArgs()
    trainer = SdxlNetworkTrainer()
    
    # Test the caching strategy creation
    caching_strategy = trainer.get_text_encoder_outputs_caching_strategy(args)
    
    print("Testing selective text encoder caching...")
    print(f"Text encoder learning rates: {args.text_encoder_lr}")
    print(f"Caching strategy created: {caching_strategy}")
    
    # Check if the caching strategy has the correct cache_te2_only flag
    if hasattr(caching_strategy, 'cache_te2_only'):
        print(f"cache_te2_only flag: {caching_strategy.cache_te2_only}")
        if caching_strategy.cache_te2_only:
            print("✅ SUCCESS: Caching strategy correctly set to cache only TE2 when its learning rate is 0")
        else:
            print("❌ FAILURE: Caching strategy not set to cache only TE2")
    else:
        print("❌ FAILURE: Caching strategy missing cache_te2_only attribute")
    
    return caching_strategy is not None and hasattr(caching_strategy, 'cache_te2_only') and caching_strategy.cache_te2_only

if __name__ == "__main__":
    success = test_selective_te_caching()
    sys.exit(0 if success else 1)