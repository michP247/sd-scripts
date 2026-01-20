#!/usr/bin/env python3
"""
SDXL Dataset Resolution Copier
Copies a dataset to all 40 SDXL training resolutions with intelligent cropping

This tool creates copies of your dataset at all resolutions SDXL was trained on,
enabling comprehensive multi-resolution training coverage.

Requires: ultralytics (for YOLOv9 face detection)
Install: pip install ultralytics
YOLOv9 face model: Place face_yolov9c.pt in tools/ directory
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import library.train_util as train_util
from library.utils import setup_logging, resize_image

setup_logging()
import logging
logger = logging.getLogger(__name__)

# Try to import ultralytics for YOLOv9 face detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("ultralytics available for YOLOv9 face detection")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not found. Install with 'pip install ultralytics' for YOLOv9 face detection. Falling back to Haar Cascade.")

# SDXL's 40 training resolutions in (width, height) format
# Folders are named as WIDTHxHEIGHT to match standard convention
SDXL_TRAINING_RESOLUTIONS = [
    (2048, 512), (1984, 512), (1920, 512), (1856, 512),
    (1792, 576), (1728, 576), (1664, 576),
    (1600, 640), (1536, 640),
    (1472, 704), (1408, 704),
    (1344, 768), (1280, 768),
    (1216, 832), (1152, 832),
    (1152, 896), (1088, 896),
    (1088, 960), (1024, 960),
    (1024, 1024),
    (960, 1024), (960, 1088), (896, 1088),
    (896, 1152), (832, 1152),
    (832, 1216), (768, 1280),
    (768, 1344), (704, 1408), (704, 1472), (704, 1344),
    (640, 1536), (640, 1600),
    (576, 1664), (576, 1728), (576, 1792),
    (512, 1856), (512, 1920), (512, 1984), (512, 2048)
]


@dataclass
class CroppingConfig:
    """Configuration for YOLOv9 face detection cropping"""
    padding_ratio: float = 0.2  # Padding around detected faces
    aspect_ratio_tolerance: float = 0.1  # Tolerance for simple resize without cropping
    yolo_conf_threshold: float = 0.5  # Confidence threshold for face detection
    yolo_model: str = "tools/face_yolov9c.pt"  # YOLOv9 face detection model
    

class SmartCropper:
    """YOLOv9 face-centered cropping system"""
    
    def __init__(self, config: CroppingConfig, interpolation: str = "lanczos"):
        self.config = config
        self.interpolation = interpolation
        
        # Initialize YOLOv9 face detection model
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        try:
            logger.info(f"Loading YOLOv9 face detection model: {config.yolo_model}")
            self.yolo_model = YOLO(config.yolo_model)
            logger.info("YOLOv9 face detection model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv9 model from {config.yolo_model}: {e}")
    
    def crop_to_resolution(self, image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """Main entry point for cropping to target resolution using YOLOv9 face detection"""
        source_ar = image.width / image.height
        target_ar = target_width / target_height
        
        # If aspect ratios are similar, just resize
        if abs(source_ar - target_ar) < self.config.aspect_ratio_tolerance:
            return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Detect faces with YOLOv9
        detections = self._detect_with_yolo(image)
        
        if detections:
            # Crop around detected faces
            return self._crop_around_subjects(image, detections, target_width, target_height)
        else:
            # No faces detected - use center crop
            logger.warning("No faces detected, using center crop")
            return self._center_crop(image, target_width, target_height)
    
    def _detect_with_yolo(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Detect faces using YOLOv9 face detection model"""
        try:
            # Run inference
            results = self.yolo_model(image, conf=self.config.yolo_conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    
                    # YOLOv9 face model only detects faces, no class filtering needed
                    if conf >= self.config.yolo_conf_threshold:
                        # Get bounding box in xyxy format
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        detections.append((x, y, w, h))
            
            if detections:
                logger.debug(f"YOLOv9 detected {len(detections)} face(s)")
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLOv9 face detection failed: {e}")
            return []
    
    def _crop_around_subjects(self, image: Image.Image, faces: List[Tuple[int, int, int, int]],
                              target_width: int, target_height: int) -> Image.Image:
        """Crop around detected faces with padding"""
        # Combine all face regions
        min_x = min(x for x, y, w, h in faces)
        min_y = min(y for x, y, w, h in faces)
        max_x = max(x + w for x, y, w, h in faces)
        max_y = max(y + h for x, y, w, h in faces)
        
        # Calculate center of all faces
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        return self._crop_around_center(image, center_x, center_y, target_width, target_height)
    
    def _crop_around_center(self, image: Image.Image, center_x: int, center_y: int,
                           target_width: int, target_height: int) -> Image.Image:
        """Crop around a specified center point"""
        source_ar = image.width / image.height
        target_ar = target_width / target_height
        
        # Calculate crop dimensions maintaining target aspect ratio
        if target_ar > source_ar:
            # Target is wider - crop top/bottom
            crop_width = image.width
            crop_height = int(crop_width / target_ar)
        else:
            # Target is taller - crop left/right
            crop_height = image.height
            crop_width = int(crop_height * target_ar)
        
        # Calculate crop box centered on target point
        left = max(0, center_x - crop_width // 2)
        top = max(0, center_y - crop_height // 2)
        
        # Adjust if crop extends beyond image
        if left + crop_width > image.width:
            left = image.width - crop_width
        if top + crop_height > image.height:
            top = image.height - crop_height
        
        # Ensure non-negative
        left = max(0, left)
        top = max(0, top)
        
        # Crop and resize
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    def _center_crop(self, image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """Simple center crop"""
        source_ar = image.width / image.height
        target_ar = target_width / target_height
        
        if target_ar > source_ar:
            # Target is wider - crop top/bottom
            crop_width = image.width
            crop_height = int(crop_width / target_ar)
            crop_top = (image.height - crop_height) // 2
            cropped = image.crop((0, crop_top, crop_width, crop_top + crop_height))
        else:
            # Target is taller - crop left/right
            crop_height = image.height
            crop_width = int(crop_height * target_ar)
            crop_left = (image.width - crop_width) // 2
            cropped = image.crop((crop_left, 0, crop_left + crop_width, crop_height))
        
        return cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)


class SDXLDatasetCopier:
    """Main class for copying datasets to SDXL resolutions"""
    
    def __init__(self, source_dir: str, output_dir: str, 
                 cropping_config: CroppingConfig,
                 caption_extension: str = ".txt",
                 skip_existing: bool = False,
                 recursive: bool = True):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.cropping_config = cropping_config
        self.caption_extension = caption_extension
        self.skip_existing = skip_existing
        self.recursive = recursive
        
        self.cropper = SmartCropper(cropping_config)
        self.stats = {
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'resolutions': {}
        }
    
    def copy_dataset(self):
        """Main execution method"""
        logger.info(f"Starting SDXL dataset copy from {self.source_dir} to {self.output_dir}")
        logger.info(f"Using YOLOv9 face detection for intelligent cropping")
        logger.info(f"Target resolutions: {len(SDXL_TRAINING_RESOLUTIONS)} SDXL training resolutions")
        
        # Find all images
        image_files = self._find_images()
        logger.info(f"Found {len(image_files)} images to process")
        
        if not image_files:
            logger.error("No images found in source directory!")
            return
        
        # Create output directories for each resolution
        self._create_resolution_dirs()
        
        # Process each image
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                self._process_image(image_path)
                self.stats['processed'] += 1
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                self.stats['errors'] += 1
        
        # Save statistics
        self._save_stats()
        
        logger.info(f"\nProcessing complete!")
        logger.info(f"Processed: {self.stats['processed']}")
        logger.info(f"Skipped: {self.stats['skipped']}")
        logger.info(f"Errors: {self.stats['errors']}")
    
    def _find_images(self) -> List[Path]:
        """Find all images in source directory"""
        image_files = []
        
        if self.recursive:
            for ext in train_util.IMAGE_EXTENSIONS:
                image_files.extend(self.source_dir.rglob(f"*{ext}"))
        else:
            for ext in train_util.IMAGE_EXTENSIONS:
                image_files.extend(self.source_dir.glob(f"*{ext}"))
        
        return sorted(image_files)
    
    def _create_resolution_dirs(self):
        """Create output directories for each resolution"""
        for width, height in SDXL_TRAINING_RESOLUTIONS:
            res_dir = self.output_dir / f"{width}x{height}"
            res_dir.mkdir(parents=True, exist_ok=True)
    
    def _process_image(self, image_path: Path):
        """Process a single image to all resolutions"""
        # Load image
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            logger.error(f"Could not load image {image_path}: {e}")
            raise
        
        # Get caption path
        caption_path = image_path.with_suffix(self.caption_extension)
        has_caption = caption_path.exists()
        
        # Process for each resolution
        for width, height in SDXL_TRAINING_RESOLUTIONS:
            res_key = f"{width}x{height}"
            res_dir = self.output_dir / res_key
            
            # Output paths
            output_image_path = res_dir / image_path.name
            output_caption_path = res_dir / caption_path.name
            
            # Skip if exists and skip_existing is True
            if self.skip_existing and output_image_path.exists():
                self.stats['skipped'] += 1
                continue
            
            # Crop and resize image
            try:
                resized_image = self.cropper.crop_to_resolution(image, width, height)
                
                # Save image with maximum quality
                resized_image.save(output_image_path, quality=95, optimize=False)
                
                # Copy caption if exists
                if has_caption:
                    shutil.copy2(caption_path, output_caption_path)
                
                # Update stats
                self.stats['resolutions'][res_key] = self.stats['resolutions'].get(res_key, 0) + 1
                
            except Exception as e:
                logger.error(f"Error processing {image_path} at {res_key}: {e}")
                raise
    
    def _save_stats(self):
        """Save processing statistics"""
        stats_path = self.output_dir / "copy_stats.json"
        
        stats_output = {
            'source_dir': str(self.source_dir),
            'output_dir': str(self.output_dir),
            'cropping_config': {
                'yolo_model': self.cropping_config.yolo_model,
                'yolo_conf_threshold': self.cropping_config.yolo_conf_threshold,
                'padding_ratio': self.cropping_config.padding_ratio,
            },
            'statistics': self.stats,
            'resolutions': SDXL_TRAINING_RESOLUTIONS
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_output, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_path}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy dataset to all SDXL training resolutions with YOLOv9 face-centered cropping"
    )
    
    parser.add_argument(
        "source_dir",
        type=str,
        help="Source directory containing images"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for resolution-organized copies"
    )
    parser.add_argument(
        "--padding_ratio",
        type=float,
        default=0.2,
        help="Padding ratio around detected faces (default: 0.2)"
    )
    parser.add_argument(
        "--yolo_conf_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for YOLOv9 face detections (default: 0.5)"
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="tools/face_yolov9c.pt",
        help="Path to YOLOv9 face detection model (default: tools/face_yolov9c.pt)"
    )
    parser.add_argument(
        "--aspect_ratio_tolerance",
        type=float,
        default=0.1,
        help="Aspect ratio difference tolerance for simple resize (default: 0.1)"
    )
    parser.add_argument(
        "--caption_extension",
        type=str,
        default=".txt",
        help="Caption file extension (default: .txt)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip images that already exist in output"
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Do not search recursively for images"
    )
    
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    # Create cropping config
    cropping_config = CroppingConfig(
        padding_ratio=args.padding_ratio,
        aspect_ratio_tolerance=args.aspect_ratio_tolerance,
        yolo_conf_threshold=args.yolo_conf_threshold,
        yolo_model=args.yolo_model
    )
    
    # Create copier and execute
    copier = SDXLDatasetCopier(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        cropping_config=cropping_config,
        caption_extension=args.caption_extension,
        skip_existing=args.skip_existing,
        recursive=not args.no_recursive
    )
    
    copier.copy_dataset()


if __name__ == "__main__":
    main()