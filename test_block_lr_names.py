#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
import os

# Add the lycoris directory to the path to import the modified module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from lycoris.kohya import create_network, LycorisNetworkKohya

# --- Mock Model Components ---
# We create a minimal mock UNet that mimics the structure of a Qwen Image model
# to test the block_lr_weights logic without loading the actual model.

class MockQwenTransformerBlock(nn.Module):
    def __init__(self, block_idx):
        # Use the exact class name that the 'full-lin' preset looks for
        self.__class__.__name__ = "QwenImageTransformerBlock"
        super().__init__()
        self.block_idx = block_idx
        # Mock internal structure that LyCORIS will target
        self.img_mod = nn.Linear(128, 128)
        self.img_attn = nn.Linear(128, 128)
        self.img_mlp_net = nn.Sequential(
            nn.Linear(128, 512),
            nn.Linear(512, 128)
        )
        self.txt_mod = nn.Linear(128, 128)
        self.txt_attn = nn.Linear(128, 128)
        self.txt_mlp_net = nn.Sequential(
            nn.Linear(128, 512),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        return x

class MockUNet(nn.Module):
    def __init__(self, num_blocks=60):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([
            MockQwenTransformerBlock(i) for i in range(num_blocks)
        ])

# --- Test Configuration ---
# Use the same block_lr_weights from your training command
BLOCK_LR_WEIGHTS_STR = "0.1000,0.3166,1.0994,1.9848,1.8143,1.8956,2.0000,1.8519,1.7998,1.8304,1.8200,1.8486,1.8772,1.9282,1.9434,1.9142,1.8286,1.8756,0.4671"
BLOCK_LR_WEIGHTS = [float(w) for w in BLOCK_LR_WEIGHTS_STR.split(",")]

# --- Test Execution ---
def main():
    try:
        print("Starting test for block learning rate names...")
        print(f"Number of block LR weights provided: {len(BLOCK_LR_WEIGHTS)}")
        print(f"First 10 weights: {BLOCK_LR_WEIGHTS[:10]}")
        print("-" * 50)

        # Create mock UNet and text encoder (None for this test)
        mock_unet = MockUNet(num_blocks=len(BLOCK_LR_WEIGHTS))
        mock_text_encoder = None

        # Create the LyCORIS network using the same parameters as your training script
        # We only need to pass the relevant arguments for this test
        network = create_network(
            multiplier=1.0,
            network_dim=4,
            network_alpha=1,
            vae=None,
            text_encoder=mock_text_encoder,
            unet=mock_unet,
            # Pass arguments that affect optimizer param creation
            network_module="lora",
            block_lr_weights=BLOCK_LR_WEIGHTS_STR,
            preset="full-lin", # Using the same preset as your command
            train_norm=True,
            use_tucker=True,
            dora_wd=False,
            wd_on_out=False,
        )

        # Prepare optimizer parameters
        # This is the key function we are testing
        all_params, lr_descriptions = network.prepare_optimizer_params(
            unet_lr=5e-4,
            learning_rate=5e-4
        )

        print(f"Total parameter groups created: {len(all_params)}")
        print(f"Total learning rate descriptions created: {len(lr_descriptions)}")
        print("-" * 50)

        # Verify the descriptions
        print("Learning Rate Descriptions:")
        unique_descriptions = sorted(list(set(lr_descriptions)))
        
        if len(unique_descriptions) == len(BLOCK_LR_WEIGHTS):
            print(f"SUCCESS: Found {len(unique_descriptions)} unique descriptions, matching the number of weights.")
        else:
            print(f"WARNING: Expected {len(BLOCK_LR_WEIGHTS)} unique descriptions, but found {len(unique_descriptions)}.")
            
        print("\nFirst 20 unique descriptions:")
        for i, desc in enumerate(unique_descriptions[:20]):
            print(f"  {i+1:2d}. {desc}")
        
        if len(unique_descriptions) > 20:
            print("  ...")
            print(f"Last 5 unique descriptions:")
            for i, desc in enumerate(unique_descriptions[-5:], start=len(unique_descriptions)-4):
                print(f"  {i+1:2d}. {desc}")

        # Check for the specific pattern we expect
        print("\nChecking for 'qwen_block-' pattern in descriptions...")
        qwen_blocks = [d for d in unique_descriptions if "qwen_block-" in d]
        print(f"Found {len(qwen_blocks)} descriptions matching 'qwen_block-' pattern.")

        if len(qwen_blocks) > 0:
            print("\nExample 'qwen_block-' descriptions:")
            for desc in qwen_blocks[:5]:
                print(f"  - {desc}")

    except Exception as e:
        print(f"An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()