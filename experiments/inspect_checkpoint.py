#!/usr/bin/env python3
"""
Inspect a checkpoint file to see its contents.
"""

import torch
import os
import sys


def inspect_checkpoint(filepath):
    if not os.path.exists(filepath):
        return f"File {filepath} does not exist"

    try:
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        result = f"Checkpoint: {filepath}\n"
        result += f"Type: {type(data)}\n"
        if isinstance(data, dict):
            result += f"Keys: {list(data.keys())}\n"
            for k, v in data.items():
                result += f"  {k}: {type(v)}\n"
                if isinstance(v, dict):
                    result += f"    sub-keys: {list(v.keys())}\n"
                elif hasattr(v, 'shape'):
                    result += f"    shape: {v.shape}\n"
                elif isinstance(v, list):
                    result += f"    length: {len(v)}\n"
                    if v:
                        result += f"    first item type: {type(v[0])}\n"
        else:
            result += f"Data: {str(data)[:500]}...\n"
        return result
    except Exception as e:
        return f"Error loading {filepath}: {e}"


def main():
    checkpoint_dir = "../checkpoints/processed_graphs"
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint dir {checkpoint_dir} does not exist")
        return

    # List some checkpoint files
    files = os.listdir(checkpoint_dir)
    pt_files = [f for f in files if f.endswith('.pt')]
    if not pt_files:
        print("No .pt files in checkpoint dir")
        return

    # Inspect the first few
    output_file = "debug_checkpoints.txt"
    with open(output_file, 'w') as f:
        f.write("Checkpoint Files Inspection\n")
        f.write("=" * 50 + "\n")

    for i, pt_file in enumerate(pt_files[:3]):  # First 3
        filepath = os.path.join(checkpoint_dir, pt_file)
        print(f"Inspecting {pt_file}...")
        info = inspect_checkpoint(filepath)
        with open(output_file, 'a') as f:
            f.write(f"\n=== Checkpoint {i+1}: {pt_file} ===\n")
            f.write(info + "\n")

    print(f"\nInspection complete. Check {output_file}")


if __name__ == "__main__":
    main()
