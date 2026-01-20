import os
import shutil
import glob

source_dir = "dataset/HSdata"
dest_dir = "dataset/HSdata_val"

prefixes = [
    "hmi124", "hmi35", "hmi13", "hmi22", "hmi3", "hmi138", "hmi53", "hmi195", "hmi2", "hmi27",
    "hmi11", "hmi102", "hmi26", "hmi61", "hmi169", "hmi99", "hmi56", "hmi145", "hmi165", "hmi196",
    "hmi84", "hmi74", "hmi172", "hmi174", "hmi80", "hmi173", "hmi179", "hmi68", "hmi85", "hmi133",
    "hmi39", "hmi187", "hmi60", "hmi82", "hmi41", "hmi42", "hmi36", "hmi167", "hmi111", "hmi37"
]

# Ensure precise matching to avoid moving hmi111 when targeting hmi11
files_moved = 0
for prefix in prefixes:
    # Match exact filename with extension
    pattern_dot = os.path.join(source_dir, f"{prefix}.*")
    # Match filename with underscore (e.g. hmi1_te_outputs)
    pattern_us = os.path.join(source_dir, f"{prefix}_*")
    
    files = glob.glob(pattern_dot) + glob.glob(pattern_us)
    
    for f in files:
        # Double check that we aren't matching hmi111 when looking for hmi11
        # The basename should start with prefix + "." or prefix + "_"
        base = os.path.basename(f)
        if base.startswith(prefix + ".") or base.startswith(prefix + "_"):
            shutil.move(f, os.path.join(dest_dir, base))
            files_moved += 1
            print(f"Moved {base}")

print(f"Total files moved: {files_moved}")
