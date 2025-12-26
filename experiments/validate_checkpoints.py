import argparse
import sys
from pathlib import Path
import torch

# Local imports
from cpg_processor import CPG_Processor


def _make_processor(checkpoint_dir: Path):
    class _DummyConf:
        def __init__(self):
            self.embedding_dim = 768
            self.name = "microsoft/codebert-base"

    # Minimal processor on CPU; no model/tokenizer loaded
    return CPG_Processor(
        tokenizer=None,
        model=None,
        device="cpu",
        batch_size=1,
        checkpoint_dir=str(checkpoint_dir),
        embedding_model_conf=_DummyConf(),
    )


def validate_checkpoint(processor: CPG_Processor, ckpt_path: Path):
    name = ckpt_path.stem
    try:
        data = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        return name, "load_failed", f"torch.load error: {e}"

    if not processor._checkpoint_is_compatible(data):
        return name, "incompatible", "checkpoint schema mismatch"

    try:
        processor._validate_project_stage_outputs(
            name, 1, data.get("stage1_graph", {}), data.get(
                "stage1_labels", {})
        )
        processor._validate_project_stage_outputs(
            name, 2, data.get("stage2_graph", {}), data.get(
                "stage2_labels", {})
        )
        processor._validate_project_stage_outputs(
            name, 3, data.get("stage3_graph", {}), data.get(
                "stage3_labels", {})
        )
    except Exception as e:
        return name, "invalid", str(e)

    return name, "ok", ""


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate saved CPG checkpoints")
    parser.add_argument(
        "--dir",
        default="./checkpoints/processed_graphs",
        help="Directory containing .pt checkpoints",
    )
    args = parser.parse_args(argv)

    ckpt_dir = Path(args.dir)
    if not ckpt_dir.exists():
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return 1

    processor = _make_processor(ckpt_dir)

    statuses = {"ok": 0, "invalid": 0, "incompatible": 0, "load_failed": 0}
    failures = []

    for ckpt_path in sorted(ckpt_dir.glob("*.pt")):
        name, status, detail = validate_checkpoint(processor, ckpt_path)
        statuses[status] = statuses.get(status, 0) + 1
        if status != "ok":
            failures.append((name, status, detail))

    total = sum(statuses.values())
    print(f"Checked {total} checkpoints in {ckpt_dir}")
    print(f"OK: {statuses['ok']}  Invalid: {statuses['invalid']}  Incompatible: {statuses['incompatible']}  LoadFailed: {statuses['load_failed']}")

    if failures:
        print("\nFailures:")
        for name, status, detail in failures:
            print(f"- {name}: {status} -> {detail}")

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
