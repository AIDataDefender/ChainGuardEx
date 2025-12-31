"""Split EtherScanIO 3-stage dataset shards into 3 smaller datasets by project.

Goal
- Load existing stage1/2/3 (graph,label) shard files.
- Find projects common to ALL 3 stages.
- Split those projects into 3 roughly equal groups.
- Save each group back as shard files into separate folders.

This script is intentionally standalone: it does NOT import CustomDataset.
It operates directly on the on-disk shard format:
  - each shard is a torch-saved list of triples (project_name, subkey, obj)
  - graphs and labels are stored in separate parallel shard files

Example (default workspace):
  python experiments/split_etherscanio_into_3.py \
    --load-dir ./save_data \
    --out-root ./save_data_splits

It will create:
  save_data_splits/split_0/
  save_data_splits/split_1/
  save_data_splits/split_2/

and write files like:
  EtherScanIO_dataset_stage3_microsoft_codebert-base_graph_0.pt
  EtherScanIO_dataset_stage3_microsoft_codebert-base_label_0.pt

Notes
- The split key is project_name (not item count). This guarantees that all 3 stages
    for a given project end up in the same split folder.
- Only projects present across all 3 stages are included.
- Memory: the script never loads all shards into RAM at once. Peak RAM is roughly:
        (one input shard) + (one label shard) + (output buffers up to --max-items-per-shard)
    You can reduce peak memory by lowering --max-items-per-shard.
- Project-ID scan can read from labels (default) to avoid loading DGL graphs just
    to extract project names.
"""

from __future__ import annotations

import argparse
import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from tqdm import tqdm

# DGL is usually required to unpickle heterograph objects stored inside torch shards.
# If DGL isn't installed, torch.load may fail.
try:
    import dgl  # noqa: F401
except Exception:
    dgl = None


Triple = Tuple[str, str, object]


@dataclass(frozen=True)
class StagePaths:
    graph_base: Path
    label_base: Path
    is_single_file: bool


def _stage_base_filenames(*, source: str, stage: int, embedding_id: str, is_test: bool) -> Tuple[str, str]:
    test_tag = "_test" if is_test else ""
    g_name = f"{source}_dataset_stage{stage}{test_tag}_{embedding_id}_graph.pt"
    l_name = f"{source}_dataset_stage{stage}{test_tag}_{embedding_id}_label.pt"
    return g_name, l_name


def _stage_base_candidates(*, load_dir: Path, source: str, stage: int, embedding_id: str, is_test: bool) -> List[Tuple[Path, Path]]:
    """Return ordered (graph_base,label_base) candidates.

    Supports two historical patterns:
    - preferred: `_test_` in the middle (dataset.py preferred)
    - legacy: `_test` without underscore before embedding

    Your current save_data uses `_test_...`? Actually shows `_stage1_test_microsoft...`
    i.e. `_test_` after stage; we match that.
    """
    # Preferred already handled by _stage_base_filenames.
    g_name, l_name = _stage_base_filenames(
        source=source, stage=stage, embedding_id=embedding_id, is_test=is_test
    )
    preferred = (load_dir / g_name, load_dir / l_name)

    # Legacy variant: stage{n}_test{embedding} (no underscore after test)
    # Example: EtherScanIO_dataset_stage1_testmicrosoft_codebert-base_graph.pt
    if is_test:
        legacy_g = f"{source}_dataset_stage{stage}_test{embedding_id}_graph.pt"
        legacy_l = f"{source}_dataset_stage{stage}_test{embedding_id}_label.pt"
        legacy = (load_dir / legacy_g, load_dir / legacy_l)
        if legacy != preferred:
            return [preferred, legacy]

    return [preferred]


def _find_stage_paths(*, load_dir: Path, source: str, stage: int, embedding_id: str, is_test: bool) -> StagePaths:
    """Resolve stage base paths for (graph,label) that exist on disk.

    Accepts either:
    - single-file: graph_base and label_base exist
    - sharded: graph_base_0.pt and label_base_0.pt exist

    Returns the base paths; caller can iterate shard indices.
    """
    candidates = _stage_base_candidates(
        load_dir=load_dir,
        source=source,
        stage=stage,
        embedding_id=embedding_id,
        is_test=is_test,
    )

    for g_base, l_base in candidates:
        if g_base.exists() and l_base.exists():
            return StagePaths(graph_base=g_base, label_base=l_base, is_single_file=True)

        g0 = Path(str(g_base).replace(".pt", "_0.pt"))
        l0 = Path(str(l_base).replace(".pt", "_0.pt"))
        if g0.exists() and l0.exists():
            return StagePaths(graph_base=g_base, label_base=l_base, is_single_file=False)

    raise FileNotFoundError(
        f"Could not find stage{stage} files for source={source} embedding_id={embedding_id} is_test={is_test} under {load_dir}"
    )


def _iter_shard_indices(*, graph_base: Path, label_base: Path) -> List[int]:
    """Return consecutive shard indices present for both graph and label."""
    indices: List[int] = []
    shard_idx = 0
    while True:
        g_shard = Path(str(graph_base).replace(".pt", f"_{shard_idx}.pt"))
        l_shard = Path(str(label_base).replace(".pt", f"_{shard_idx}.pt"))
        if not (g_shard.exists() and l_shard.exists()):
            break
        indices.append(shard_idx)
        shard_idx += 1
    return indices


def _load_list(path: Path) -> List[Triple]:
    """torch.load wrapper with mmap fallback handling."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _scan_projects_for_stage(*, stage_paths: StagePaths, scan_from: str = "label") -> Set[str]:
    """Scan only project names for one stage, streaming shard-by-shard.

    Args:
        scan_from: 'label' (default) or 'graph'. Scanning from labels usually avoids
            DGL unpickling overhead and is typically lighter.
    """
    projects: Set[str] = set()

    scan_from = (scan_from or "label").strip().lower()
    if scan_from not in {"graph", "label"}:
        raise ValueError("scan_from must be 'graph' or 'label'")

    base_path = stage_paths.graph_base if scan_from == "graph" else stage_paths.label_base

    if stage_paths.is_single_file:
        items = _load_list(base_path)
        for it in items:
            if isinstance(it, (list, tuple)) and len(it) == 3:
                p, _sub, _obj = it
                projects.add(str(p))
        del items
        gc.collect()
        return projects

    indices = _iter_shard_indices(
        graph_base=stage_paths.graph_base, label_base=stage_paths.label_base)
    if not indices:
        raise FileNotFoundError(
            f"No shards found for {stage_paths.graph_base}")

    for shard_idx in tqdm(indices, desc=f"Scan projects ({scan_from}) stage shards: {base_path.name}"):
        shard_path = Path(str(base_path).replace(".pt", f"_{shard_idx}.pt"))
        chunk = _load_list(shard_path)
        if not isinstance(chunk, list):
            raise TypeError(
                f"Expected list in {shard_path}, got {type(chunk)}")
        for it in chunk:
            if isinstance(it, (list, tuple)) and len(it) == 3:
                p, _sub, _obj = it
                projects.add(str(p))
        del chunk
        if shard_idx % 25 == 0:
            gc.collect()

    gc.collect()
    return projects


def _chunk_projects(projects: Sequence[str], n_splits: int) -> List[List[str]]:
    """Deterministically split sorted projects into n_splits near-equal chunks."""
    projects = sorted([str(p) for p in projects])
    total = len(projects)
    if total == 0:
        return [[] for _ in range(n_splits)]

    base = total // n_splits
    rem = total % n_splits

    chunks: List[List[str]] = []
    start = 0
    for i in range(n_splits):
        size = base + (1 if i < rem else 0)
        end = start + size
        chunks.append(projects[start:end])
        start = end
    return chunks


@dataclass
class _StageSplitWriters:
    split_dirs: List[Path]
    graph_out_bases: List[Path]
    label_out_bases: List[Path]
    max_items_per_shard: int

    shard_idx: List[int]
    buf_g: List[List[Triple]]
    buf_l: List[List[Triple]]
    total_items: List[int]

    @classmethod
    def create(
        cls,
        *,
        split_dirs: List[Path],
        out_graph_name: str,
        out_label_name: str,
        max_items_per_shard: int,
    ) -> "_StageSplitWriters":
        graph_out_bases = [d / out_graph_name for d in split_dirs]
        label_out_bases = [d / out_label_name for d in split_dirs]
        return cls(
            split_dirs=split_dirs,
            graph_out_bases=graph_out_bases,
            label_out_bases=label_out_bases,
            max_items_per_shard=max_items_per_shard,
            shard_idx=[0 for _ in split_dirs],
            buf_g=[[] for _ in split_dirs],
            buf_l=[[] for _ in split_dirs],
            total_items=[0 for _ in split_dirs],
        )

    def _flush_one(self, split_i: int) -> None:
        if not self.buf_g[split_i]:
            return
        if len(self.buf_g[split_i]) != len(self.buf_l[split_i]):
            raise ValueError(
                f"Flush split={split_i}: graphs={len(self.buf_g[split_i])} labels={len(self.buf_l[split_i])} mismatch"
            )
        idx = self.shard_idx[split_i]
        g_out = Path(str(self.graph_out_bases[split_i]).replace(
            ".pt", f"_{idx}.pt"))
        l_out = Path(str(self.label_out_bases[split_i]).replace(
            ".pt", f"_{idx}.pt"))

        torch.save(self.buf_g[split_i], g_out)
        torch.save(self.buf_l[split_i], l_out)

        self.total_items[split_i] += len(self.buf_g[split_i])
        self.shard_idx[split_i] += 1
        self.buf_g[split_i].clear()
        self.buf_l[split_i].clear()

    def maybe_flush(self, split_i: int) -> None:
        if len(self.buf_g[split_i]) >= self.max_items_per_shard:
            self._flush_one(split_i)

    def flush_all(self) -> None:
        for i in range(len(self.split_dirs)):
            self._flush_one(i)


def _stream_split_stage(
    *,
    stage_paths: StagePaths,
    split_of_project: Dict[str, int],
    writers: _StageSplitWriters,
) -> None:
    """Stream-read a stage (graphs+labels) and write filtered items into split writers."""

    def _process_chunk(chunk_g: List[Triple], chunk_l: List[Triple]) -> None:
        if len(chunk_g) != len(chunk_l):
            raise ValueError(
                f"Chunk size mismatch: graphs={len(chunk_g)} labels={len(chunk_l)}")
        for gi, li in zip(chunk_g, chunk_l):
            if not (isinstance(gi, (list, tuple)) and len(gi) == 3):
                raise TypeError(f"Bad graph item type: {type(gi)}")
            if not (isinstance(li, (list, tuple)) and len(li) == 3):
                raise TypeError(f"Bad label item type: {type(li)}")
            gp, gsub, gobj = gi
            lp, lsub, lobj = li
            gp = str(gp)
            lp = str(lp)
            gsub = str(gsub)
            lsub = str(lsub)
            if gp != lp or gsub != lsub:
                raise ValueError(
                    f"Graph/label misaligned: graph=({gp},{gsub}) label=({lp},{lsub})")

            split_i = split_of_project.get(gp)
            if split_i is None:
                continue

            writers.buf_g[split_i].append((gp, gsub, gobj))
            writers.buf_l[split_i].append((lp, lsub, lobj))
            writers.maybe_flush(split_i)

    if stage_paths.is_single_file:
        chunk_g = _load_list(stage_paths.graph_base)
        chunk_l = _load_list(stage_paths.label_base)
        if not isinstance(chunk_g, list) or not isinstance(chunk_l, list):
            raise TypeError(
                f"Expected list in single files stage: graphs={type(chunk_g)} labels={type(chunk_l)}"
            )
        _process_chunk(chunk_g, chunk_l)
        del chunk_g, chunk_l
        writers.flush_all()
        gc.collect()
        return

    indices = _iter_shard_indices(
        graph_base=stage_paths.graph_base, label_base=stage_paths.label_base)
    if not indices:
        raise FileNotFoundError(
            f"No shards found for {stage_paths.graph_base}")

    for shard_idx in tqdm(indices, desc=f"Split stage shards: {stage_paths.graph_base.name}"):
        g_shard = Path(str(stage_paths.graph_base).replace(
            ".pt", f"_{shard_idx}.pt"))
        l_shard = Path(str(stage_paths.label_base).replace(
            ".pt", f"_{shard_idx}.pt"))
        chunk_g = _load_list(g_shard)
        chunk_l = _load_list(l_shard)
        if not isinstance(chunk_g, list) or not isinstance(chunk_l, list):
            raise TypeError(
                f"Expected list in shard files: {g_shard} -> {type(chunk_g)}, {l_shard} -> {type(chunk_l)}"
            )
        _process_chunk(chunk_g, chunk_l)
        del chunk_g, chunk_l

        if shard_idx % 25 == 0:
            gc.collect()

    writers.flush_all()
    gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split EtherScanIO 3-stage dataset into 3 project-aligned datasets")
    parser.add_argument("--source", type=str, default="EtherScanIO",
                        help="Dataset source (default EtherScanIO)")
    parser.add_argument("--load-dir", type=str, default="./save_data",
                        help="Folder containing existing shard files")
    parser.add_argument("--out-root", type=str,
                        default="./save_data_splits", help="Output root folder")
    parser.add_argument(
        "--embedding-id",
        type=str,
        default="microsoft_codebert-base",
        help="Embedding id used in filenames (default microsoft_codebert-base)",
    )
    parser.add_argument("--is-test", action="store_true",
                        help="Use *_test_* files")
    parser.add_argument(
        "--splits",
        type=int,
        default=3,
        help="Number of splits (default 3)",
    )
    parser.add_argument(
        "--max-items-per-shard",
        type=int,
        default=int(os.getenv("MAX_ITEMS_PER_SHARD", "500")),
        help="Flush after N items per output shard (default env MAX_ITEMS_PER_SHARD or 500)",
    )
    parser.add_argument(
        "--scan-from",
        type=str,
        default="label",
        choices=["label", "graph"],
        help="Where to scan project IDs from (default: label). 'label' avoids DGL graph unpickling during scan.",
    )

    args = parser.parse_args()

    load_dir = Path(args.load_dir)
    out_root = Path(args.out_root)
    source = str(args.source)
    embedding_id = str(args.embedding_id)
    is_test = bool(args.is_test)
    n_splits = int(args.splits)
    max_items_per_shard = int(args.max_items_per_shard)
    scan_from = str(args.scan_from)

    if n_splits <= 0:
        raise ValueError("--splits must be > 0")
    if max_items_per_shard <= 0:
        raise ValueError("--max-items-per-shard must be > 0")

    if dgl is None:
        print(
            "[WARN] dgl is not importable; torch.load of graphs may fail if graphs are DGL objects.")

    print("=" * 80)
    print("Split EtherScanIO dataset")
    print(f"  load_dir:      {load_dir.resolve()}")
    print(f"  out_root:      {out_root.resolve()}")
    print(f"  source:        {source}")
    print(f"  embedding_id:  {embedding_id}")
    print(f"  is_test:       {is_test}")
    print(f"  splits:        {n_splits}")
    print(f"  shard_size:    {max_items_per_shard}")
    print(f"  scan_from:     {scan_from}")
    print("=" * 80)

    # Resolve existing stage paths
    stage_paths = {}
    for s in (1, 2, 3):
        stage_paths[s] = _find_stage_paths(
            load_dir=load_dir,
            source=source,
            stage=s,
            embedding_id=embedding_id,
            is_test=is_test,
        )
        print(
            f"Resolved stage{s}: graph_base={stage_paths[s].graph_base.name} label_base={stage_paths[s].label_base.name} "
            f"mode={'single' if stage_paths[s].is_single_file else 'sharded'}"
        )

    # Pass 1: scan project ids per stage (streaming)
    projects_by_stage: Dict[int, Set[str]] = {}
    for s in (1, 2, 3):
        projects_by_stage[s] = _scan_projects_for_stage(
            stage_paths=stage_paths[s], scan_from=scan_from)
        print(f"Stage{s}: projects={len(projects_by_stage[s])}")

    common_projects = set.intersection(
        projects_by_stage[1], projects_by_stage[2], projects_by_stage[3]
    )
    print(f"Common projects across stage1/2/3: {len(common_projects)}")

    if not common_projects:
        raise RuntimeError(
            "No common projects across stage1/2/3; cannot split consistently.")

    # Split projects
    chunks = _chunk_projects(sorted(common_projects), n_splits)
    for i, ch in enumerate(chunks):
        print(f"Split {i}: projects={len(ch)}")

    split_of_project: Dict[str, int] = {}
    for i, ch in enumerate(chunks):
        for p in ch:
            split_of_project[str(p)] = i

    # Prepare output directories
    split_dirs = [out_root / f"split_{i}" for i in range(n_splits)]
    for d in split_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Pass 2: stream each stage, and write into split folders
    for s in (1, 2, 3):
        out_g_name, out_l_name = _stage_base_filenames(
            source=source, stage=s, embedding_id=embedding_id, is_test=is_test
        )
        writers = _StageSplitWriters.create(
            split_dirs=split_dirs,
            out_graph_name=out_g_name,
            out_label_name=out_l_name,
            max_items_per_shard=max_items_per_shard,
        )
        print("-" * 80)
        print(f"Writing stage{s} -> {out_g_name} / {out_l_name}")

        _stream_split_stage(
            stage_paths=stage_paths[s],
            split_of_project=split_of_project,
            writers=writers,
        )

        for i in range(n_splits):
            print(
                f"stage{s} split_{i}: shards_written={writers.shard_idx[i]} items_written={writers.total_items[i]}"
            )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
