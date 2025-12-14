from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import argparse
import traceback

import torch

from the_utils import graph_utils


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def _default_multilabel_row() -> List[float]:
    # Keep this aligned with training code: use the canonical helper.
    return list(graph_utils.vuln_to_label([]))


def _is_row_vuln(row: Any) -> bool:
    try:
        return bool(sum(float(x) for x in row) > 0.0)
    except Exception:
        return False


def _ensure_list_of_rows(value: Any) -> List[List[float]]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, list):
        return []
    if not value:
        return []
    # If it's a flat list, treat it as a single row.
    if value and not isinstance(value[0], list):
        return [list(value)]
    return [list(r) for r in value]


def _fix_stage3_labels_for_graph(g, lbl: Any) -> Dict[str, List[List[float]]]:
    """Checkpoint-only repair for stage3 label schema and lengths.

    Returns dict with keys: cfg_node, ast_node. Each is list[row], aligned by node index.
    If label data is missing or incompatible, fills zeros with correct lengths.
    """
    base_row = _default_multilabel_row()
    out: Dict[str, List[List[float]]] = {"cfg_node": [], "ast_node": []}

    if isinstance(lbl, dict):
        out["cfg_node"] = _ensure_list_of_rows(lbl.get("cfg_node"))
        out["ast_node"] = _ensure_list_of_rows(lbl.get("ast_node"))

    for ntype in ("cfg_node", "ast_node"):
        if not hasattr(g, "ntypes") or ntype not in g.ntypes:
            out[ntype] = []
            continue

        target_n = int(g.num_nodes(ntype))
        rows = out.get(ntype) or []

        # Normalize each row to the expected label dimension.
        fixed_rows: List[List[float]] = []
        for r in rows:
            r = list(r)
            if len(r) < len(base_row):
                r = r + [0.0] * (len(base_row) - len(r))
            elif len(r) > len(base_row):
                r = r[: len(base_row)]
            fixed_rows.append(r)

        # Fix row count to match node count.
        if len(fixed_rows) < target_n:
            fixed_rows.extend([list(base_row)
                              for _ in range(target_n - len(fixed_rows))])
        elif len(fixed_rows) > target_n:
            fixed_rows = fixed_rows[:target_n]

        out[ntype] = fixed_rows

    return out


def _fix_binary_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int,)):
        return 1 if value != 0 else 0
    if isinstance(value, float):
        return 1 if value != 0.0 else 0
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return 1 if float(value.detach().cpu().item()) != 0.0 else 0
    return 0


@dataclass
class FixerConfig:
    input_checkpoint_dir: Path
    output_checkpoint_dir: Path
    save_dir: Path
    output_base: str


class CPG_Checkpoint_Fixer:
    """Checkpoint-only repair/export utility.

    Uses ONLY the contents of checkpoint files:
    - Reuses DGL graphs.
    - Validates/repairs per-stage label dicts (keys + lengths).
    - Exports fixed per-project checkpoints + dataset.py-compatible per-stage .pt triples.
    """

    def __init__(self, cfg: FixerConfig):
        self.cfg = cfg
        self.cfg.output_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.save_dir.mkdir(parents=True, exist_ok=True)

    def fix_all(self) -> None:
        ckpt_files = sorted(self.cfg.input_checkpoint_dir.glob("*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(
                f"No checkpoints found in {self.cfg.input_checkpoint_dir}")

        flattened_graphs: Dict[int, List[Tuple[str, str, Any]]] = {
            1: [], 2: [], 3: []}
        flattened_labels: Dict[int, List[Tuple[str, str, Any]]] = {
            1: [], 2: [], 3: []}

        print("=" * 60)
        print("Fixing checkpoints (reuse DGL graphs, recompute labels)")
        print("=" * 60)
        print(f"Input checkpoints:  {self.cfg.input_checkpoint_dir}")
        print(f"Output checkpoints: {self.cfg.output_checkpoint_dir}")
        print(f"Save dir:           {self.cfg.save_dir}")
        print(f"Output base:        {self.cfg.output_base}")
        print(f"Total checkpoints:  {len(ckpt_files)}")

        fixed_count = 0
        skipped = 0

        for ckpt_path in ckpt_files:
            try:
                checkpoint = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False)
                p_name = checkpoint.get("project_name") or ckpt_path.stem

                required = [
                    "stage1_graph",
                    "stage2_graph",
                    "stage3_graph",
                ]
                for k in required:
                    if k not in checkpoint:
                        raise KeyError(f"Checkpoint missing {k}")

                dgl_s1 = checkpoint["stage1_graph"]
                dgl_s2 = checkpoint["stage2_graph"]
                dgl_s3 = checkpoint["stage3_graph"]

                if not isinstance(dgl_s1, dict) or not isinstance(dgl_s2, dict) or not isinstance(dgl_s3, dict):
                    raise TypeError(
                        "stageX_graph must be dict[subkey->DGLGraph]")

                # Load existing labels if present; repair to be compatible with graphs.
                raw_s1 = checkpoint.get("stage1_labels") or {}
                raw_s2 = checkpoint.get("stage2_labels") or {}
                raw_s3 = checkpoint.get("stage3_labels") or {}

                s1_labels_by_key: Dict[str, int] = {}
                for subkey in dgl_s1.keys():
                    s1_labels_by_key[subkey] = _fix_binary_label(
                        raw_s1.get(subkey, 0) if isinstance(raw_s1, dict) else 0)

                s2_labels_by_key: Dict[str, int] = {}
                for subkey in dgl_s2.keys():
                    s2_labels_by_key[subkey] = _fix_binary_label(
                        raw_s2.get(subkey, 0) if isinstance(raw_s2, dict) else 0)

                s3_labels_by_key: Dict[str, Dict[str, Any]] = {}
                for subkey, g in dgl_s3.items():
                    existing = raw_s3.get(subkey) if isinstance(
                        raw_s3, dict) else None
                    s3_labels_by_key[subkey] = _fix_stage3_labels_for_graph(
                        g, existing)

                # Optional: if stage3 labels indicate a vuln but stage2 label is 0, do NOT override.
                # This tool is for schema repair/export; it keeps existing stage1/2 semantics.

                fixed_checkpoint = {
                    "project_name": p_name,
                    "stage1_graph": dgl_s1,
                    "stage2_graph": dgl_s2,
                    "stage3_graph": dgl_s3,
                    "stage1_labels": s1_labels_by_key,
                    "stage2_labels": s2_labels_by_key,
                    "stage3_labels": s3_labels_by_key,
                }

                out_path = self.cfg.output_checkpoint_dir / \
                    f"{_safe_name(p_name)}.pt"
                torch.save(fixed_checkpoint, out_path)

                # Flatten per-stage for dataset.py compatibility
                for subkey, g in dgl_s1.items():
                    flattened_graphs[1].append((p_name, subkey, g))
                    flattened_labels[1].append(
                        (p_name, subkey, s1_labels_by_key[subkey]))
                for subkey, g in dgl_s2.items():
                    flattened_graphs[2].append((p_name, subkey, g))
                    flattened_labels[2].append(
                        (p_name, subkey, s2_labels_by_key[subkey]))
                for subkey, g in dgl_s3.items():
                    flattened_graphs[3].append((p_name, subkey, g))
                    flattened_labels[3].append(
                        (p_name, subkey, s3_labels_by_key[subkey]))

                fixed_count += 1
                if fixed_count % 25 == 0:
                    print(
                        f"Fixed {fixed_count}/{len(ckpt_files)} checkpoints...")

            except Exception as e:
                skipped += 1
                print(f"[SKIP] {ckpt_path.name}: {e}")
                traceback.print_exc()

        # Save per-stage .pt files (same triple format as dataset.py)
        for stage in (1, 2, 3):
            g_path = self.cfg.save_dir / \
                f"{self.cfg.output_base}_stage{stage}_graph.pt"
            l_path = self.cfg.save_dir / \
                f"{self.cfg.output_base}_stage{stage}_label.pt"
            torch.save(flattened_graphs[stage], g_path)
            torch.save(flattened_labels[stage], l_path)
            print(
                f"Saved stage{stage}: graphs={len(flattened_graphs[stage])} labels={len(flattened_labels[stage])}"
            )

        print("=" * 60)
        print(f"Done. Fixed={fixed_count}, Skipped={skipped}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix checkpoint labels using existing DGL graphs")
    parser.add_argument(
        "--input_checkpoint_dir",
        default="./checkpoints2/processed_graphs",
        help="Directory containing original per-project checkpoints (*.pt)",
    )
    parser.add_argument(
        "--output_checkpoint_dir",
        default="./checkpoints_fixed/processed_graphs",
        help="Directory to write fixed per-project checkpoints (*.pt)",
    )
    parser.add_argument(
        "--save_dir",
        default="./save_data",
        help="Directory to write per-stage flattened .pt files",
    )
    parser.add_argument(
        "--output_base",
        default="DAppSCAN_dataset_fixed",
        help="Base filename prefix for per-stage outputs (no extension)",
    )

    args = parser.parse_args()

    cfg = FixerConfig(
        input_checkpoint_dir=Path(args.input_checkpoint_dir),
        output_checkpoint_dir=Path(args.output_checkpoint_dir),
        save_dir=Path(args.save_dir),
        output_base=str(args.output_base),
    )
    fixer = CPG_Checkpoint_Fixer(cfg)
    fixer.fix_all()
