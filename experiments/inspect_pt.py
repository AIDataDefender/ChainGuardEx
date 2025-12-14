#!/usr/bin/env python3
"""
Simple debug script to inspect saved PT files without importing DGL.
"""

import torch
import os
import sys


def inspect_pt_file(filepath):
    if not os.path.exists(filepath):
        return f"File {filepath} does not exist"

    try:
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        result = f"File: {filepath}\n"
        result += f"Type: {type(data)}\n"
        if isinstance(data, list):
            result += f"Length: {len(data)}\n"
            if data:
                item = data[0]
                result += f"First item type: {type(item)}\n"
                if isinstance(item, tuple) and len(item) == 2:
                    name, graph_data = item
                    result += f"Name: {name} (type: {type(name)})\n"
                    result += f"Graph data type: {type(graph_data)}\n"
                    if isinstance(graph_data, dict):
                        result += f"Graph data keys: {list(graph_data.keys())}\n"
                        for k, v in graph_data.items():
                            result += f"  {k}: {type(v)}\n"
                            if isinstance(v, list):
                                result += f"    length: {len(v)}\n"
                                if v and len(v) > 0:
                                    result += f"    first item: {v[0]} (type: {type(v[0])})\n"
                                    if isinstance(v[0], list):
                                        result += f"      sub-length: {len(v[0])}\n"
                                        if v[0]:
                                            result += f"      first sub-item: {v[0][0]} (type: {type(v[0][0])})\n"
                            elif isinstance(v, dict):
                                result += f"    keys: {list(v.keys())}\n"
                    elif isinstance(graph_data, int):
                        result += f"Int value: {graph_data}\n"
                    elif hasattr(graph_data, 'num_nodes'):
                        result += f"DGL Graph: {graph_data.num_nodes()} nodes\n"
                    else:
                        result += f"Other: {str(graph_data)[:200]}...\n"
        else:
            result += f"Data: {str(data)[:500]}...\n"
        return result
    except Exception as e:
        return f"Error loading {filepath}: {e}"


def main():
    save_dir = "../save_data"
    stages = [1, 2, 3]

    output_file = "debug_pt_files.txt"
    with open(output_file, 'w') as f:
        f.write("PT Files Inspection\n")
        f.write("=" * 50 + "\n")

    for stage in stages:
        graph_file = os.path.join(
            save_dir, f"DAppSCAN_dataset_stage{stage}_graph.pt")
        label_file = os.path.join(
            save_dir, f"DAppSCAN_dataset_stage{stage}_label.pt")

        print(f"\nInspecting Stage {stage}...")

        with open(output_file, 'a') as f:
            f.write(f"\n=== Stage {stage} ===\n")

            graph_info = inspect_pt_file(graph_file)
            f.write("Graph File:\n")
            f.write(graph_info + "\n")

            label_info = inspect_pt_file(label_file)
            f.write("Label File:\n")
            f.write(label_info + "\n")

    print(f"\nInspection complete. Check {output_file}")


if __name__ == "__main__":
    main()
