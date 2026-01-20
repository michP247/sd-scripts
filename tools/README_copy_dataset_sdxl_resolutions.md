# SDXL Dataset Resolution Copier

A powerful tool for copying your dataset to all 40 SDXL training resolutions with intelligent content-aware cropping.

## Overview

This tool creates copies of your dataset at all resolutions that SDXL was originally trained on, enabling comprehensive multi-resolution training coverage. It features intelligent cropping strategies that preserve important content when converting between different aspect ratios.

## Features

- **40 SDXL Training Resolutions**: Automatically generates dataset copies for all 40 resolutions SDXL was trained on
- **Intelligent Cropping**: Multiple cropping strategies to preserve important subjects
  - Content-aware cropping with face detection
  - Saliency-based subject detection
  - Rule-based composition-aware cropping
  - Simple center cropping
- **Caption Preservation**: Automatically copies caption files to all resolution variants
- **High Quality Output**: Maximum quality image saving (quality=95)
- **Flexible Configuration**: Extensive command-line options for customization
- **Progress Tracking**: Comprehensive statistics and progress reporting

## SDXL Training Resolutions

The tool creates 40 resolution folders with the following resolutions.

Folders are named using standard WIDTHxHEIGHT convention:

```
2048x512, 1984x512, 1920x512, 1856x512,
1792x576, 1728x576, 1664x576,
1600x640, 1536x640,
1472x704, 1408x704, 1344x704,
1344x768, 1280x768,
1216x832, 1152x832,
1152x896, 1088x896,
1088x960, 1024x960,
1024x1024,
960x1024, 960x1088, 896x1088,
896x1152, 832x1152,
832x1216, 768x1280,
768x1344, 704x1408, 704x1472,
640x1536, 640x1600,
576x1664, 576x1728, 576x1792,
512x1856, 512x1920, 512x1984, 512x2048
```

These match SDXL's original 40 training resolutions, ranging from ultra-wide landscapes (2048x512) to ultra-tall portraits (512x2048).

## Installation

No additional dependencies beyond standard sd-scripts requirements. Ensure you have:
- Python 3.10+
- OpenCV (cv2) for advanced cropping features
- PIL/Pillow for image processing

## Basic Usage

```bash
python tools/copy_dataset_sdxl_resolutions.py /path/to/source/dataset /path/to/output/dataset
```

This will:
1. Find all images in the source dataset
2. Create 40 resolution folders in the output directory
3. Copy and intelligently crop each image to all 40 resolutions
4. Copy caption files (.txt by default) to each resolution folder
5. Generate statistics about the processing

## Advanced Usage

### Content-Aware Cropping (Default)

```bash
python tools/copy_dataset_sdxl_resolutions.py \
    /path/to/source/dataset \
    /path/to/output/dataset \
    --cropping_strategy content_aware
```

This strategy:
- Detects faces and crops around them
- Uses saliency detection to find important regions
- Falls back to rule-based cropping if detection fails
- Best for preserving subjects and important content

### Rule-Based Cropping

```bash
python tools/copy_dataset_sdxl_resolutions.py \
    /path/to/source/dataset \
    /path/to/output/dataset \
    --cropping_strategy rule_based
```

This strategy:
- Uses composition rules (rule of thirds)
- Positions crops intelligently based on aspect ratios
- Faster than content-aware
- Good for landscapes and general content

### Center Cropping

```bash
python tools/copy_dataset_sdxl_resolutions.py \
    /path/to/source/dataset \
    /path/to/output/dataset \
    --cropping_strategy center
```

This strategy:
- Simple center crop and resize
- Fastest processing
- Good for centered subjects

### Disable Face Detection

If you don't want face detection (faster processing):

```bash
python tools/copy_dataset_sdxl_resolutions.py \
    /path/to/source/dataset \
    /path/to/output/dataset \
    --no_face_detection
```

### Skip Existing Files

Resume interrupted processing:

```bash
python tools/copy_dataset_sdxl_resolutions.py \
    /path/to/source/dataset \
    /path/to/output/dataset \
    --skip_existing
```

### Custom Caption Extension

If your captions use a different extension:

```bash
python tools/copy_dataset_sdxl_resolutions.py \
    /path/to/source/dataset \
    /path/to/output/dataset \
    --caption_extension .caption
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `source_dir` | str | Required | Source directory containing images |
| `output_dir` | str | Required | Output directory for resolution copies |
| `--cropping_strategy` | str | content_aware | Cropping strategy: content_aware, rule_based, center |
| `--no_face_detection` | flag | False | Disable face detection |
| `--no_saliency_detection` | flag | False | Disable saliency detection |
| `--padding_ratio` | float | 0.1 | Padding around detected subjects (0.0-1.0) |
| `--fallback_strategy` | str | rule_based | Fallback when content-aware fails |
| `--aspect_ratio_tolerance` | float | 0.1 | Tolerance for simple resize vs crop |
| `--caption_extension` | str | .txt | Caption file extension |
| `--skip_existing` | flag | False | Skip existing output files |
| `--no_recursive` | flag | False | Don't search subdirectories |

## Output Structure

```
output_directory/
├── 512x2048/
│   ├── image1.jpg
│   ├── image1.txt
│   ├── image2.jpg
│   ├── image2.txt
│   └── ...
├── 1024x1024/
│   ├── image1.jpg
│   ├── image1.txt
│   └── ...
├── 2048x512/
│   └── ...
└── copy_stats.json
```

## Statistics File

After processing, a `copy_stats.json` file is generated with:
- Source and output directories
- Cropping configuration used
- Number of images processed, skipped, and errors
- Per-resolution processing counts
- List of all target resolutions

Example:
```json
{
  "source_dir": "/path/to/source",
  "output_dir": "/path/to/output",
  "cropping_config": {
    "strategy": "content_aware",
    "face_detection": true,
    "saliency_detection": true
  },
  "statistics": {
    "processed": 100,
    "skipped": 0,
    "errors": 0,
    "resolutions": {
      "2048x512": 100,
      "1024x1024": 100,
      ...
    }
  }
}
```

## Cropping Strategies Explained

### Content-Aware (Recommended)

**When to use**: Datasets with people, faces, or specific subjects

**How it works**:
1. Detects faces using Haar Cascades
2. If faces found, crops around all detected faces with padding
3. Falls back to saliency detection (edge detection + contours)
4. If saliency found, crops around the salient region
5. Final fallback to rule-based cropping

**Pros**:
- Preserves important subjects
- Handles off-center compositions
- Adapts to content

**Cons**:
- Slower processing
- Requires OpenCV

### Rule-Based (Balanced)

**When to use**: General datasets, landscapes, varied content

**How it works**:
1. Uses composition rules (rule of thirds)
2. For wide targets: crops from upper-middle third (30% from top)
3. For tall targets: crops from center horizontally
4. Maintains compositional balance

**Pros**:
- Fast processing
- Good for most content
- No external dependencies

**Cons**:
- May miss off-center subjects
- Fixed positioning rules

### Center (Fast)

**When to use**: Centered subjects, quick processing needed

**How it works**:
1. Simple center crop calculation
2. Crops equal amounts from edges
3. Resizes to target

**Pros**:
- Fastest processing
- Simple and predictable

**Cons**:
- May cut off off-center subjects
- No content awareness

## Training with Multiple Resolutions

After copying your dataset to all SDXL resolutions, you can train using any resolution folder, or create a mixed-resolution training configuration:

```toml
[[datasets]]
resolution = [1024, 1024]
batch_size = 4

  [[datasets.subsets]]
  image_dir = "/path/to/output/1024x1024"
  num_repeats = 1

[[datasets]]
resolution = [2048, 512]  # ultra-wide landscape
batch_size = 2

  [[datasets.subsets]]
  image_dir = "/path/to/output/2048x512"
  num_repeats = 1
```

This enables training on multiple aspect ratios simultaneously, matching SDXL's original training data distribution.

## Performance Tips

1. **Use SSD**: Output to SSD for faster write speeds
2. **Skip Existing**: Use `--skip_existing` to resume interrupted processing
3. **Choose Strategy**: Use `rule_based` for faster processing if content-aware isn't needed
4. **Disable Detection**: Use `--no_face_detection` and `--no_saliency_detection` for maximum speed

## Troubleshooting

### Face Detection Not Working

If face detection fails to load:
```
WARNING: Face detector failed to load, disabling face detection
```

This is usually fine - the tool will automatically fall back to saliency detection and rule-based cropping.

### Out of Memory

For very large images or limited RAM:
- Process in batches by splitting your source dataset
- Use `--skip_existing` to resume if interrupted

### Poor Crop Quality

If crops are cutting off subjects:
1. Try `--cropping_strategy content_aware` (if not already using it)
2. Increase `--padding_ratio` to 0.2 or higher
3. Check if face detection is working (remove `--no_face_detection`)

## Examples

### Example 1: Character Dataset
```bash
python tools/copy_dataset_sdxl_resolutions.py \
    ~/datasets/characters \
    ~/datasets/characters_sdxl \
    --cropping_strategy content_aware \
    --padding_ratio 0.15
```

### Example 2: Landscape Dataset
```bash
python tools/copy_dataset_sdxl_resolutions.py \
    ~/datasets/landscapes \
    ~/datasets/landscapes_sdxl \
    --cropping_strategy rule_based
```

### Example 3: Quick Processing
```bash
python tools/copy_dataset_sdxl_resolutions.py \
    ~/datasets/misc \
    ~/datasets/misc_sdxl \
    --cropping_strategy center \
    --no_face_detection \
    --no_saliency_detection
```

### Example 4: Resume Processing
```bash
python tools/copy_dataset_sdxl_resolutions.py \
    ~/datasets/large \
    ~/datasets/large_sdxl \
    --skip_existing
```

## License

This tool is part of sd-scripts and follows the same license.