import toml
import sys
import os

# Add the parent directory to the path so we can import lycoris
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lycoris.kohya import LycorisNetworkKohya

def test_preset_loading(preset_path):
    """Loads a preset and checks for key Qwen-Image patterns."""
    print(f"Loading preset from: {preset_path}")
    try:
        preset = toml.load(preset_path)
        print("Preset loaded successfully.")
        
        # Check for key patterns that should exist
        key_patterns = [
            "transformer_blocks.*.txt_mlp_net.*",
            "transformer_blocks.*.img_mlp_net.*",
            "transformer_blocks.*.attn.*",
            "transformer_blocks.*.img_mod.*",
            "transformer_blocks.*.txt_mod.*"
        ]
        
        found_patterns = []
        for pattern in key_patterns:
            # Check in name_algo_map
            if any(pattern in key for key in preset.get("name_algo_map", {}).keys()):
                found_patterns.append(f"Found '{pattern}' in name_algo_map")
            # Also check in the main dict
            if any(pattern in key for key in preset.keys()):
                found_patterns.append(f"Found '{pattern}' in main preset keys")
        
        if found_patterns:
            print("\n--- SUCCESS: Found expected patterns ---")
            for p in found_patterns:
                print(f"  - {p}")
        else:
            print("\n--- FAILURE: Could not find expected patterns ---")
            
        return len(found_patterns) == len(key_patterns)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        preset_path = sys.argv[1]
    else:
        # Default to the user's preset path
        preset_path = "/home/en1u2/AI-Stuff/lyco_presets/qwen_nfn_lora.toml"
    
    success = test_preset_loading(preset_path)
    
    if success:
        print("\n✅ Preset test PASSED. The fix should be working.")
    else:
        print("\n❌ Preset test FAILED. Check the preset file.")