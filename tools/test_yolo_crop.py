#!/usr/bin/env python3
"""Quick test script for YOLOv9 face detection and cropping"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from copy_dataset_sdxl_resolutions import SmartCropper, CroppingConfig
from library.utils import setup_logging

setup_logging()
import logging
logger = logging.getLogger(__name__)

def test_yolo_crop():
    # Input and output paths
    input_path = "/home/en1u2/AI-Stuff/References/latest/main2.png"
    output_path = "/home/en1u2/AI-Stuff/References/latest/main2-cropped.png"
    
    # Target resolution (wide landscape: 576 pixels wide × 1728 pixels tall)
    target_width = 576
    target_height = 1728
    
    logger.info(f"Loading image: {input_path}")
    image = Image.open(input_path)
    logger.info(f"Original size: {image.width}x{image.height}")
    
    # Create config with YOLOv9 face detection
    config = CroppingConfig(
        padding_ratio=0.2,
        yolo_conf_threshold=0.5,
        yolo_model="face_yolov9c.pt"  # In tools/ directory
    )
    
    # Create cropper
    logger.info("Initializing SmartCropper with YOLOv9 face detection...")
    cropper = SmartCropper(config)
    
    # Perform crop
    logger.info(f"Cropping to {target_width}x{target_height}...")
    cropped = cropper.crop_to_resolution(image, target_width, target_height)
    
    # Save result
    logger.info(f"Saving to: {output_path}")
    cropped.save(output_path, quality=95)
    
    logger.info(f"Done! Cropped image size: {cropped.width}x{cropped.height}")
    logger.info(f"Output saved to: {output_path}")

if __name__ == "__main__":
    test_yolo_crop()