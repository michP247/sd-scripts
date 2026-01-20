#!/usr/bin/env python3

# SDXL's 40 training resolutions in (width, height) format
SDXL_TRAINING_RESOLUTIONS = [
    (2048, 512), (1984, 512), (1920, 512), (1856, 512),
    (1792, 576), (1728, 576), (1664, 576),
    (1600, 640), (1536, 640),
    (1472, 704), (1408, 704), (1344, 704),
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

# Convert to folder names format (WIDTHxHEIGHT)
expected_folders = [f"{width}x{height}" for width, height in SDXL_TRAINING_RESOLUTIONS]

# Actual folders found in the directory
actual_folders = [
    "512x1856", "512x1920", "512x1984", "512x2048",
    "576x1664", "576x1728", "576x1792",
    "640x1536", "640x1600",
    "704x1344", "704x1408", "704x1472",
    "768x1280", "768x1344",
    "832x1152", "832x1216",
    "896x1088", "896x1152",
    "960x1024", "960x1088",
    "1024x960", "1024x1024",
    "1088x896", "1088x960",
    "1152x832", "1152x896",
    "1216x832",
    "1280x768",
    "1344x704", "1344x768",
    "1408x704",
    "1472x704",
    "1536x640",
    "1600x640",
    "1664x576",
    "1728x576",
    "1792x576",
    "1856x512",
    "1920x512",
    "1984x512",
    "2048x512"
]

print(f"Expected folders: {len(expected_folders)}")
print(f"Actual folders: {len(actual_folders)}")

# Find folders in expected but not in actual
missing_folders = [folder for folder in expected_folders if folder not in actual_folders]
print(f"\nMissing folders: {len(missing_folders)}")
for folder in missing_folders:
    print(f"  {folder}")

# Find folders in actual but not in expected
extra_folders = [folder for folder in actual_folders if folder not in expected_folders]
print(f"\nExtra folders: {len(extra_folders)}")
for folder in extra_folders:
    print(f"  {folder}")

# Check for any typos or similar names
for actual in actual_folders:
    if actual not in expected_folders:
        # Try to find similar expected folders
        similar = [exp for exp in expected_folders if exp.replace('x', '_') == actual or actual.replace('x', '_') == exp]
        if similar:
            print(f"\nPossible typo: '{actual}' should be '{similar[0]}'")