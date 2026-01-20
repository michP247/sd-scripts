#!/usr/bin/env python3
"""
Test script to verify block_lr_weights functionality
"""

import torch
import argparse
from lycoris.kohya import create_network, LycorisNetworkKohya

def test_block_lr_weights():
    # Create a simple UNet-like structure for testing
    class MockUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Add some mock modules that would be found in a real UNet
            self.input_blocks = torch.nn.ModuleList([
                torch.nn.Linear(10, 10),
                torch.nn.Linear(10, 10),
            ])
            self.middle_block = torch.nn.ModuleList([
                torch.nn.Linear(10, 10),
            ])
            self.output_blocks = torch.nn.ModuleList([
                torch.nn.Linear(10, 10),
                torch.nn.Linear(10, 10),
            ])
            self.time_embedding = torch.nn.Linear(10, 10)
            self.conv_in = torch.nn.Linear(10, 10)
            self.conv_out = torch.nn.Linear(10, 10)
    
    # Create mock text encoder
    class MockTextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Linear(10, 10)
    
    # Create instances
    unet = MockUNet()
    text_encoder = MockTextEncoder()
    
    # Test with block_lr_weights
    block_lr_weights = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0"
    
    print("Testing block_lr_weights functionality...")
    print(f"Block LR weights: {block_lr_weights}")
    
    # Create network with block_lr_weights
    network = create_network(
        multiplier=1.0,
        network_dim=8,
        network_alpha=16,
        vae=None,
        text_encoder=text_encoder,
        unet=unet,
        block_lr_weights=block_lr_weights,
        algo="locon",
        dora_wd=True,
        preset="full",
        conv_dim=8,
        conv_alpha=16
    )
    
    # Check if block_lr_weights was set correctly
    print(f"Network block_lr_weights: {network.block_lr_weights}")
    print(f"Network block_lr: {network.block_lr}")
    
    # Test parameter preparation
    params, lr_descriptions = network.prepare_optimizer_params_with_multiple_te_lrs(
        text_encoder_lr=[1e-5, 5e-6],
        unet_lr=1e-4,
        learning_rate=1e-4
    )
    
    print(f"Number of parameter groups: {len(params)}")
    print(f"LR descriptions: {lr_descriptions}")
    
    # Check if block-specific descriptions are present
    block_descriptions = [desc for desc in lr_descriptions if "unet_block" in desc]
    print(f"Block-specific LR descriptions: {block_descriptions}")
    
    if block_descriptions:
        print("SUCCESS: Block learning rate descriptions found!")
    else:
        print("FAILURE: Block learning rate descriptions not found!")
    
    # Test without block_lr_weights
    print("\nTesting without block_lr_weights...")
    network_no_blocks = create_network(
        multiplier=1.0,
        network_dim=8,
        network_alpha=16,
        vae=None,
        text_encoder=text_encoder,
        unet=unet,
        algo="locon",
        dora_wd=True,
        preset="full",
        conv_dim=8,
        conv_alpha=16
    )
    
    params_no_blocks, lr_descriptions_no_blocks = network_no_blocks.prepare_optimizer_params_with_multiple_te_lrs(
        text_encoder_lr=[1e-5, 5e-6],
        unet_lr=1e-4,
        learning_rate=1e-4
    )
    
    print(f"Number of parameter groups (no blocks): {len(params_no_blocks)}")
    print(f"LR descriptions (no blocks): {lr_descriptions_no_blocks}")
    
    # Check if standard descriptions are present
    standard_descriptions = [desc for desc in lr_descriptions_no_blocks if desc in ["unet", "textencoder1", "textencoder2"]]
    print(f"Standard LR descriptions: {standard_descriptions}")
    
    if standard_descriptions:
        print("SUCCESS: Standard learning rate descriptions found!")
    else:
        print("FAILURE: Standard learning rate descriptions not found!")

if __name__ == "__main__":
    test_block_lr_weights()