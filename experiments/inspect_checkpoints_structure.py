import torch
import argparse
from pathlib import Path
import sys
import dgl


def recursive_dump(obj, indent=0):
    indent_str = '  ' * indent
    if isinstance(obj, dgl.DGLGraph):
        return "dgl"  # Skip networkx graphs for brevity
    if isinstance(obj, dict):
        print(f"{indent_str}dict:")
        for k, v in obj.items():
            print(f"{indent_str}  {k}:")
            recursive_dump(v, indent + 2)
    elif isinstance(obj, (list, tuple)):
        type_name = 'list' if isinstance(obj, list) else 'tuple'
        print(f"{indent_str}{type_name}:")
        for i, v in enumerate(obj):
            print(f"{indent_str}  [{i}]:")
            recursive_dump(v, indent + 2)
    elif hasattr(obj, '__dict__'):
        print(f"{indent_str}{type(obj).__name__}:")
        for k, v in vars(obj).items():
            print(f"{indent_str}  {k}:")
            recursive_dump(v, indent + 2)
    else:
        print(f"{indent_str}{type(obj).__name__}: {str(obj)}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect checkpoints2 structure")
    parser.add_argument('--out', required=True, help="Output text file")
    args = parser.parse_args()
    _max = 1
    x = 0

    with open(args.out, 'w') as f:
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            dir_path = Path('checkpoints/processed_graphs/')
            for file in dir_path.glob('*.pt'):
                if x >= _max:
                    break
                print(f"File: {file}")
                data = torch.load(file, map_location='cpu')
                recursive_dump(data)
                print("\n")
                x += 1
        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    main()
