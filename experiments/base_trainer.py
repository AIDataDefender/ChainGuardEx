from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import os
import random
import time

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from contextlib import nullcontext
from tqdm import tqdm
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Subset, ConcatDataset
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    classification_report,
    hamming_loss,
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from experiments.dataset import CustomDataset, custom_collate
    from experiments.the_utils.logger import setup_logger
    from experiments.the_utils.graph_utils import OWASP_VULN
    from experiments.the_utils.FocalLoss_and_PCGrad import FocalLoss, PCGrad
except ImportError:
    from dataset import CustomDataset, custom_collate
    from the_utils.logger import setup_logger
    from the_utils.graph_utils import OWASP_VULN
    from the_utils.FocalLoss_and_PCGrad import FocalLoss, PCGrad

os.environ["DGLBACKEND"] = "pytorch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Config:
    class Stage1:
        batch_size = 256
        num_epochs = 30
        learning_rate = 1e-3
        patience = 3
        hidden_dim = 128
        fixed_thresholds = 0.65  # Fixed threshold / use None for dynamic

        oversample_ratio = 0.0
        out_dim = 1
        weight_decay = 0.001

        scheduler_factor = 0.5
        scheduler_patience = 2
        scheduler_min_lr = 1e-7

        focal_alpha = 0.65
        focal_gamma = 1.55

        grad_clip_max_norm = 1.0
        eval_threshold_min = 0.05
        eval_threshold_max = 0.95
        eval_threshold_steps = 91

        class_weight_min = 1.0
        class_weight_max = 10.0

    class Stage2:
        batch_size = 512
        num_epochs = 30
        learning_rate = 1e-4
        patience = 3
        hidden_dim = 128
        fixed_thresholds = 0.65

        oversample_ratio = 4.0
        undersample_ratio = 0.0
        out_dim = 1
        weight_decay = 0.001

        scheduler_factor = 0.5
        scheduler_patience = 2
        scheduler_min_lr = 1e-7

        focal_alpha = 0.65
        focal_gamma = 1.55

        grad_clip_max_norm = 1.0
        eval_threshold_min = 0.05
        eval_threshold_max = 0.95
        eval_threshold_steps = 91

        class_weight_min = 1.0
        class_weight_max = 10.0

    class Stage3:
        batch_size = 512
        num_epochs = 30
        learning_rate = 5e-3
        patience = 3
        hidden_dim = 128
        fixed_thresholds = 0.65

        oversample_ratio = 5
        out_dim = 8
        weight_decay = 0.001

        scheduler_factor = 0.5
        scheduler_patience = 1
        scheduler_min_lr = 1e-8

        focal_alpha = 0.45
        focal_gamma = 1.65

        label_smoothing = 0.05
        grad_clip_max_norm = 1.0

        eval_threshold_min = 0.1
        eval_threshold_max = 0.9
        eval_threshold_steps = 17

        eval_score_f1_weight = 0.6
        eval_score_prec_weight = 0.4
        eval_min_threshold = 0.2

        class_weight_min = 1.0
        class_weight_max = 10.0

    class Base:
        train_ratio = 0.6
        val_ratio = 0.2  # test_ratio = 0.2
        rand_seed = 42
        use_pcgrad = False

        # mando_class_weight = [10.0] * 8
        # dappscan_class_weight = [10.0] * 8
        # etherscanio_class_weight = [10, 10, 10, 6.438515, 10, 10, 10, 10]


class BaseTrainer:
    def __init__(
        self,
        stage=3,
        batch_size=None,
        num_epochs=None,
        learning_rate=None,
        patience=None,
        hidden_dim=None,
        rand_seed=None,
        log_folder=None,
        oversample_ratio=None,
        undersample_ratio=None,
        model_type=None,
    ):
        self.stage = int(stage)
        if self.stage not in [1, 2, 3]:
            raise ValueError("Stage must be 1, 2, or 3.")
        self.rand_seed = rand_seed if rand_seed is not None else Config.Base.rand_seed
        self.set_rand_seed(self.rand_seed)

        # Get stage-specific config
        if self.stage == 1:
            config = Config.Stage1
        elif self.stage == 2:
            config = Config.Stage2
        else:
            config = Config.Stage3

        # Use provided values or defaults from config
        self.batch_size = batch_size if batch_size is not None else config.batch_size
        self.num_epochs = num_epochs if num_epochs is not None else config.num_epochs
        self.learning_rate = learning_rate if learning_rate is not None else config.learning_rate
        self.patience = patience if patience is not None else config.patience
        self.hidden_dim = hidden_dim if hidden_dim is not None else config.hidden_dim
        self.oversample_ratio = oversample_ratio if oversample_ratio is not None else config.oversample_ratio
        self.undersample_ratio = undersample_ratio if undersample_ratio is not None else getattr(config, 'undersample_ratio', 1.0)
        self.fixed_thresholds = config.fixed_thresholds  # 0.65
        # Model selection (baseline_X variants). None preserves baseline_X default.
        self.model_type = model_type

        # Store config for later use
        self.config = config

        # Logging
        self.log_folder = log_folder or Path(
            f'Logs/Stage{self.stage}_{datetime.now().strftime("%Y%m%d_%H%M")}')
        os.makedirs(self.log_folder, exist_ok=True)
        self.logger = setup_logger(
            f"{self.log_folder}/trainer.log", logging.INFO)

        self.logger.info(f"[GREEN] Using Model Type: {self.model_type}")
        self.logger.info(f"[GREEN] Oversample Ratio: {self.oversample_ratio}")
        self.logger.info(f"[GREEN] Undersample Ratio: {self.undersample_ratio}")
        self.logger.info(f"[GREEN] Fixed Thresholds: {self.fixed_thresholds}")
        self.logger.info(f"[GREEN] Patience: {self.patience}")
        self.logger.info(f"[GREEN] Learning Rate: {self.learning_rate}")
        self.logger.info(f"[GREEN] Batch Size: {self.batch_size}")
        self.logger.info(f"[GREEN] Num Epochs: {self.num_epochs}")
        self.logger.info(f"[GREEN] Hidden Dim: {self.hidden_dim}")
        self.logger.info(f"[GREEN] Stage: {self.stage}")
        self.logger.info(f"[GREEN] Random Seed: {self.rand_seed}")
        print("Initializing BaseTrainer...")

        # GPU Detection and Diagnostics
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA Available: {gpu_count} GPU(s) detected")
            print(f"✓ GPU 0: {gpu_name}")
            self.device = torch.device("cuda")

            # Performance knobs (safe defaults for RTX 4060 Laptop / Ampere+)
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
                self.logger.info(
                    "[PERF] Enabled TF32 + cudnn.benchmark + matmul_precision=high")
            except Exception as e:
                self.logger.warning(
                    f"[PERF] Failed to set TF32/cudnn knobs: {e}")
        else:
            print("✗ CUDA NOT Available - Running on CPU")
            print(f"  PyTorch version: {torch.__version__}")
            print(f"  CUDA built version: {torch.version.cuda}")
            self.device = torch.device("cpu")

        # AMP (mixed precision)
        self.use_amp = bool(cuda_available)
        self.amp_dtype = torch.float16
        try:
            self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        except Exception:
            self.scaler = None
            self.use_amp = False

        # CUDA memory tracking (helps spot leaks/regressions)
        self.track_cuda_memory = bool(cuda_available)
        self._last_cuda_reserved_bytes = None

        # Mode Switching
        self.is_graph_level = (self.stage in [1, 2])
        self.logger.info(
            f"Initialized Trainer for Stage {self.stage} ({'Graph' if self.is_graph_level else 'Node'} Classification)")
        self.logger.info(f"Using device: {self.device}")

        # History tracking
        self.history = {"train_loss": [],
                        "val_loss": [], "val_f1": [], "val_auc": [], "val_hamming": [], "epoch_times": []}

        self.embedding_dims = None
        # Initialize Dataset
        self.load_data(self.rand_seed)

        # Initialize Model
        self.setup_model()

    def set_rand_seed(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_data(self, rand_seed):
        self.logger.info("Loading dataset...")
        # Pass stage to CustomDataset
        dataset1 = CustomDataset(
            source="DAppSCAN", load_dir="/mnt/d/KLTN2/save_data2", force_reload=False, rand_seed=rand_seed, stage=self.stage)
        self.embedding_dims = dataset1.embedding_dims
        dataset2 = CustomDataset(
            source="MANDO", load_dir="/mnt/d/KLTN2/save_data3", force_reload=False, rand_seed=rand_seed, stage=self.stage)
        dataset3 = CustomDataset(
            source="EtherScanIO", load_dir="./save_data_splits/split_0", force_reload=False, rand_seed=rand_seed, stage=self.stage)

        # Fuse into one dataset
        self.dataset = ConcatDataset([dataset1, dataset2, dataset3])
        self.dataset1 = dataset3  # Keep reference for attributes

        # IMPORTANT: model setup depends on embedding dims from the dataset processor.
        self.embedding_dims = getattr(self.dataset1, "embedding_dims", None)
        if not isinstance(self.embedding_dims, dict) or not self.embedding_dims:
            raise ValueError(
                "Dataset is missing `embedding_dims` (expected dict like {cfg_node, ast_node, edge}). "
                "Check `experiments/dataset.py` / `CPG_Processor.embedding_dims`."
            )

        # Stratified Split (80/10/10) to keep label ratios consistent across splits.
        # For Stage 1/2: binary graph labels.
        # For Stage 3: binary proxy label = any vulnerable node in (cfg|ast).
        n_total = len(self.dataset)
        if n_total <= 0:
            raise ValueError("Empty dataset")

        def _to_binary(v):
            try:
                if isinstance(v, torch.Tensor):
                    if v.numel() != 1:
                        return 0
                    v = v.item()
                if isinstance(v, bool):
                    v = int(v)
                if isinstance(v, (int, float)):
                    return 1 if float(v) > 0.5 else 0
            except Exception:
                return 0
            return 0

        labels = []
        if self.stage in [1, 2]:
            # Extract labels from all datasets
            for ds in [dataset1, dataset2, dataset3]:
                for _, lbl in getattr(ds, "dataset_label", []) or []:
                    labels.append(_to_binary(lbl))
        else:
            # For Stage 3, use manual split
            labels = None

        if labels is not None and len(labels) != n_total:
            self.logger.warning(
                f"Label extraction mismatch: got {len(labels)} labels for dataset size {n_total}. Falling back to random split."
            )
            labels = None

        def _stratified_indices(y, seed, train_ratio=0.8, val_ratio=0.1):
            idx0 = [i for i, v in enumerate(y) if int(v) == 0]
            idx1 = [i for i, v in enumerate(y) if int(v) == 1]
            rng = random.Random(seed)
            rng.shuffle(idx0)
            rng.shuffle(idx1)

            # Log overall balance (useful for Stage 1/2 where positives may be extremely rare).
            try:
                self.logger.info(
                    f"[GREEN][SPLIT DEBUG] Total={len(y)} | pos={len(idx1)} ({(len(idx1)/len(y)*100):.3f}%) | neg={len(idx0)}"
                )
            except Exception:
                pass

            # If a class is missing, stratification is impossible.
            if len(idx0) == 0 or len(idx1) == 0:
                all_idx = list(range(len(y)))
                rng.shuffle(all_idx)
                tr = int(len(all_idx) * train_ratio)
                va = int(len(all_idx) * val_ratio)
                return all_idx[:tr], all_idx[tr: tr + va], all_idx[tr + va:]

            def split_class(idxs):
                """Split a single class list into train/val/test.

                Key property: if the class has >=3 samples, guarantee at least
                1 in val and 1 in test so val metrics (e.g., F1) are meaningful.
                """
                n = len(idxs)
                if n <= 0:
                    return [], [], []
                if n == 1:
                    return idxs[:1], [], []
                if n == 2:
                    # Prefer train+val so validation sees the rare class.
                    return idxs[:1], idxs[1:2], []

                # n >= 3
                tr = int(n * train_ratio)
                va = int(n * val_ratio)
                te = n - tr - va

                # Ensure at least 1 in each split where possible.
                va = max(1, va)
                te = max(1, te)
                tr = n - va - te
                if tr <= 0:
                    tr = 1
                    # take from the larger of (va, te)
                    if va >= te and va > 1:
                        va -= 1
                    elif te > 1:
                        te -= 1
                    else:
                        # Worst-case: push everything into train+val
                        te = 0
                        va = n - tr

                return idxs[:tr], idxs[tr: tr + va], idxs[tr + va: tr + va + te]

            tr0, va0, te0 = split_class(idx0)
            tr1, va1, te1 = split_class(idx1)

            train_idx = tr0 + tr1
            val_idx = va0 + va1
            test_idx = te0 + te1
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)
            rng.shuffle(test_idx)
            return train_idx, val_idx, test_idx

        if labels is None:
            all_idx = list(range(n_total))
            rng = random.Random(rand_seed)
            rng.shuffle(all_idx)
            train_size = int(Config.Base.train_ratio * n_total)
            val_size = int(Config.Base.val_ratio * n_total)
            train_idx = all_idx[:train_size]
            val_idx = all_idx[train_size: train_size + val_size]
            test_idx = all_idx[train_size + val_size:]

        else:
            train_idx, val_idx, test_idx = _stratified_indices(
                labels, rand_seed)

        # Persist class balance for Stage 1/2 loss weighting.
        self._split_stats = {
            "train": {"pos": 0, "neg": 0},
            "val": {"pos": 0, "neg": 0},
            "test": {"pos": 0, "neg": 0},
        }
        if labels is not None:
            def _count(idxs):
                pos = sum(1 for i in idxs if int(labels[i]) == 1)
                neg = len(idxs) - pos
                return pos, neg

            tr_pos, tr_neg = _count(train_idx)
            va_pos, va_neg = _count(val_idx)
            te_pos, te_neg = _count(test_idx)
            self._split_stats["train"] = {"pos": tr_pos, "neg": tr_neg}
            self._split_stats["val"] = {"pos": va_pos, "neg": va_neg}
            self._split_stats["test"] = {"pos": te_pos, "neg": te_neg}
            try:
                self.logger.info(
                    f"[GREEN][SPLIT DEBUG] Train={len(train_idx)} (pos={tr_pos}, neg={tr_neg}) | "
                    f"Val={len(val_idx)} (pos={va_pos}, neg={va_neg}) | "
                    f"Test={len(test_idx)} (pos={te_pos}, neg={te_neg})"
                )
            except Exception:
                pass

        if self.stage in [1, 2] and self.oversample_ratio > 1.0 and labels is not None:
            positive_indices = [idx for idx in train_idx if labels[idx] == 1]
            if positive_indices:
                original_len = len(train_idx)
                num_duplicates = int(self.oversample_ratio - 1)
                for _ in range(num_duplicates):
                    train_idx.extend(positive_indices)
                self.logger.info(
                    f"[DATA DEBUG] Oversampled {len(positive_indices)} positive graphs {num_duplicates} times, train set from {original_len} to {len(train_idx)}")

        if self.stage == 2 and self.undersample_ratio < 1.0 and self.undersample_ratio > 0.0 and labels is not None:
            positive_indices = [idx for idx in train_idx if labels[idx] == 1]
            negative_indices = [idx for idx in train_idx if labels[idx] == 0]
            positive_count = len(positive_indices)
            target_neg = int(self.undersample_ratio * positive_count)
            if len(negative_indices) > target_neg:
                original_len = len(train_idx)
                random.shuffle(negative_indices)
                negative_indices = negative_indices[:target_neg]
                train_idx = positive_indices + negative_indices
                random.shuffle(train_idx)
                self.logger.info(
                    f"[DATA DEBUG] Undersampled negatives from {original_len - positive_count} to {target_neg}, train set from {original_len} to {len(train_idx)}")
                # Update split stats after undersampling
                tr_pos, tr_neg = _count(train_idx)
                self._split_stats["train"] = {"pos": tr_pos, "neg": tr_neg}
                self.logger.info(
                    f"[GREEN][SPLIT DEBUG] After undersampling: Train={len(train_idx)} (pos={tr_pos}, neg={tr_neg})"
                )

        if self.stage == 3 and self.oversample_ratio > 1.0:
            # Compute class counts for training set
            class_pos_counts = torch.zeros(8, dtype=torch.float32)
            for idx in train_idx:
                sample = self.dataset[idx]
                if 'cfg_labels' in sample and sample['cfg_labels'] is not None:
                    class_pos_counts += sample['cfg_labels'].sum(dim=0).cpu()
                if 'ast_labels' in sample and sample['ast_labels'] is not None:
                    class_pos_counts += sample['ast_labels'].sum(dim=0).cpu()

            # Define rare classes as those with count < mean count (excluding benign)
            mean_count = class_pos_counts.mean()
            rare_classes = [i for i in range(
                8) if class_pos_counts[i] < mean_count]
            self.logger.info(
                f"[DATA DEBUG] Rare classes (count < {mean_count:.0f}): {rare_classes} with counts {class_pos_counts[rare_classes].numpy()}")

            # Identify graphs with rare labels
            positive_indices = []
            for idx in train_idx:
                sample = self.dataset[idx]
                has_rare = False
                for labels in [sample.get('cfg_labels'), sample.get('ast_labels')]:
                    if labels is not None:
                        for class_idx in rare_classes:
                            if labels[:, class_idx].sum() > 0:
                                has_rare = True
                                break
                        if has_rare:
                            break
                if has_rare:
                    positive_indices.append(idx)

            if positive_indices:
                original_len = len(train_idx)
                num_duplicates = int(self.oversample_ratio - 1)
                for _ in range(num_duplicates):
                    train_idx.extend(positive_indices)
                self.logger.info(
                    f"[DATA DEBUG] Oversampled {len(positive_indices)} graphs with rare labels {num_duplicates} times, train set from {original_len} to {len(train_idx)}")
            else:
                self.logger.warning(
                    "[DATA DEBUG] No graphs with rare labels found for oversampling")

        train_ds = Subset(self.dataset, train_idx)
        val_ds = Subset(self.dataset, val_idx)
        test_ds = Subset(self.dataset, test_idx)

        # Create Loaders
        def collate_fn(b): return custom_collate(b, stage=self.stage)

        num_workers = 0
        # try:
        #     cpu_cnt = os.cpu_count() or 0
        #     # Conservative default that helps without over-subscribing.
        #     num_workers = 0 #int(max(0, min(6, cpu_cnt // 2)))
        #     print(
        #         f"Using {num_workers} DataLoader workers (CPU count: {cpu_cnt})")
        # except Exception:
        #     num_workers = 0

        pin_memory = bool(self.device.type == "cuda")
        persistent_workers = bool(num_workers > 0)

        def _make_loader(ds, *, shuffle: bool):
            base_kwargs = dict(
                batch_size=self.batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
            )
            perf_kwargs = dict(
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=4 if num_workers > 0 else None,
            )
            # Remove None values (torch DataLoader rejects None for prefetch_factor)
            perf_kwargs = {k: v for k, v in perf_kwargs.items()
                           if v is not None}

            try:
                return GraphDataLoader(ds, **base_kwargs, **perf_kwargs)
            except TypeError as e:
                self.logger.warning(
                    f"[PERF][DATALOADER] GraphDataLoader rejected perf kwargs ({e}); falling back to defaults.")
                return GraphDataLoader(ds, **base_kwargs)

        self.train_loader = _make_loader(train_ds, shuffle=True)
        self.val_loader = _make_loader(val_ds, shuffle=False)
        self.test_loader = _make_loader(test_ds, shuffle=False)

    def setup_model(self):
        # 1. Infer Dimensions from a sample
        sample = None
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            if sample is not None:
                break
        if sample is None:
            raise ValueError(
                "No valid sample found in dataset for model setup.")

        # Get rel_names from processor
        self.rel_names = self.dataset1.rel_names
        print(self.rel_names)
        # [('cfg_node', 'cf_false', 'cfg_node'),
        # ('cfg_node', 'return_call', 'cfg_node'),
        # ('cfg_node', 'df', 'cfg_node'),
        # ('ast_node', 'ast_to_cfg', 'cfg_node'),
        # ('cfg_node', 'cf', 'cfg_node'),
        # ('ast_node', 'ast_child', 'ast_node'),
        # ('cfg_node', 'call', 'cfg_node')]
        # Use embedding_dims from CPG_Processor
        node_dims = {
            'cfg_node': self.embedding_dims['cfg_node'],
            'ast_node': self.embedding_dims['ast_node'],
        }
        edge_dims = {}
        rel_names = []
        # baseline_X (and most hetero backbones) require canonical etypes for all stages.
        # Stage 3 already populates these; Stage 1/2 must as well to avoid empty rel_names
        # which would result in no message passing and degenerate predictions.
        g = sample['graph']
        rel_names = [_et for _et in (self.rel_names or [])]
        if not rel_names:
            rel_names = list(getattr(g, 'canonical_etypes', []))
        if not rel_names:
            raise ValueError(
                f"No rel_names available for Stage {self.stage}. "
                "Ensure dataset processor sets dataset.rel_names or graphs have canonical_etypes."
            )
        edge_dims = {et: self.embedding_dims['edge'] for et in rel_names}

        # 2. Define Output Dim
        out_dim = self.config.out_dim

        # 3. Instantiate Model
        self.model = None
        if self.model_type:
            try:
                from experiments.models.baseline_X import CascadedHeteroModel
            except Exception:
                from models.baseline_X import CascadedHeteroModel

            self.model = CascadedHeteroModel(
                node_dims=node_dims,
                edge_dims=edge_dims,
                hidden_dim=self.hidden_dim,
                out_dim=out_dim,
                rel_names=rel_names,
                stage=self.stage,
                model_type=self.model_type,
            ).to(self.device)
        else:
            try:
                from experiments.models.proto_1 import CascadedHeteroModel
            except Exception:
                from models.proto_1 import CascadedHeteroModel

            self.model = CascadedHeteroModel(
                node_dims=node_dims,
                edge_dims=edge_dims,
                hidden_dim=self.hidden_dim,
                out_dim=out_dim,
                rel_names=rel_names,
                stage=self.stage,
            ).to(self.device)

            if hasattr(self.model, "model_type"):
                self.model_type = self.model.model_type
        print(f"Model instantiated: {self.model_type}")
        # 4. Optimizer & Loss & Scheduler
        base_optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.config.weight_decay)

        if Config.Base.use_pcgrad:
            self.logger.info(
                "[YELLOW][OPTIMIZER] Using PCGrad optimizer wrapper.")
            self.optimizer = PCGrad(base_optimizer)
        else:

            self.optimizer = base_optimizer

        if self.stage in [1, 2]:
            self.scheduler = None
            self.reduce_on_plateau = optim.lr_scheduler.ReduceLROnPlateau(
                base_optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.scheduler_min_lr
            )
            self.logger.info(
                f"[SCHEDULER] ReduceLROnPlateau: factor={self.config.scheduler_factor}, patience={self.config.scheduler_patience}, min_lr={self.config.scheduler_min_lr}")
        else:
            # For Stage 3, use ReduceLROnPlateau to handle overfitting
            self.scheduler = None
            self.reduce_on_plateau = optim.lr_scheduler.ReduceLROnPlateau(
                base_optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.scheduler_min_lr
            )
            self.logger.info(
                f"[SCHEDULER] ReduceLROnPlateau: factor={self.config.scheduler_factor}, patience={self.config.scheduler_patience}")

        if self.stage in [1, 2]:
            pos = int(getattr(self, "_split_stats", {}).get(
                "train", {}).get("pos", 0))
            neg = int(getattr(self, "_split_stats", {}).get(
                "train", {}).get("neg", 0))

            if pos > 0 and neg > 0:
                pos_weight = torch.clamp(
                    torch.tensor([neg / max(1, pos)],
                                 dtype=torch.float32, device=self.device),
                    min=self.config.class_weight_min, max=self.config.class_weight_max
                )
                self.logger.info(
                    f"[GREEN][LOSS DEBUG] Stage{self.stage} pos_weight={float(pos_weight.item()):.4f}")
                self.criterion = FocalLoss(
                    alpha=self.config.focal_alpha, gamma=self.config.focal_gamma, pos_weight=pos_weight, reduction="none")
                # nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.logger.warning(
                    f"[RED][LOSS DEBUG] Stage{self.stage} has pos={pos}, neg={neg} in train; using unweighted BCEWithLogitsLoss"
                )
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            # Compute per-class weights for Stage 3 multilabel
            pos_weight = self._compute_stage3_class_weights()
            self.criterion = FocalLoss(
                alpha=self.config.focal_alpha, gamma=self.config.focal_gamma, pos_weight=pos_weight, reduction="none")

            if pos_weight is not None:
                self.logger.info(
                    f"[GREEN][LOSS DEBUG] Per-class weights: {pos_weight.cpu().numpy()}")

        # Store best thresholds for Stage 3 (initialized to 0.5)
        self.best_thresholds = torch.full(
            (self.config.out_dim,), 0.5, dtype=torch.float32) if self.stage == 3 else None

    def _compute_stage3_class_weights(self):
        """Compute per-class weights for Stage 3 multilabel classification using 50% random sample."""
        if self.stage != 3:
            return None

        self.logger.info(
            "Computing per-class weights for Stage 3 (50% random sample)...")
        class_pos_counts = torch.zeros(
            self.config.out_dim, dtype=torch.float32)
        class_neg_counts = torch.zeros(
            self.config.out_dim, dtype=torch.float32)

        # Sample 50% of the training dataset randomly
        full_dataset = self.train_loader.dataset  # This is a Subset
        full_indices = list(range(len(full_dataset)))
        random.shuffle(full_indices)
        sample_size = len(full_indices) // 4
        sample_indices = full_indices[:sample_size]
        sample_dataset = Subset(full_dataset.dataset, [
                                full_dataset.indices[i] for i in sample_indices])

        # Use larger batch size for faster counting
        count_batch_size = min(4096, len(sample_dataset))

        def temp_collate_fn(b):
            return custom_collate(b, stage=self.stage)
        temp_loader = GraphDataLoader(
            sample_dataset,
            batch_size=count_batch_size,
            shuffle=False,
            collate_fn=temp_collate_fn,
            num_workers=0,
            pin_memory=False
        )

        # Count positive/negative samples per class across sampled training data
        for batch in temp_loader:
            if batch is None:
                continue

            cfg_labels = batch.get('cfg_labels')
            ast_labels = batch.get('ast_labels')

            if cfg_labels is not None:
                class_pos_counts += cfg_labels.sum(dim=0)
                class_neg_counts += (1 - cfg_labels).sum(dim=0)

            if ast_labels is not None:
                class_pos_counts += ast_labels.sum(dim=0)
                class_neg_counts += (1 - ast_labels).sum(dim=0)

        # Compute pos_weight = neg_count / max(pos_count, 1) for each class
        pos_weight = class_neg_counts / torch.clamp(class_pos_counts, min=1.0)

        # Cap maximum weight to prevent extreme values
        pos_weight = torch.clamp(
            pos_weight, min=self.config.class_weight_min, max=self.config.class_weight_max)

        self.logger.info(
            f"[GREEN]Class positive counts (sampled): {class_pos_counts.numpy()}")
        self.logger.info(
            f"[GREEN]Class negative counts (sampled): {class_neg_counts.numpy()}")

        return pos_weight.to(self.device)

    def _prepare_batch(self, batch):
        """Moves batch to device and extracts labels based on stage."""
        if self.stage in [1, 2]:
            g = batch['graph'].to(self.device)

            # Labels are dicts: {"ContractA": 0, "ContractB": 1, ...}
            labels_raw = batch['graph_labels']  # List of dicts
            project_names = batch['project_names']

            # Build a mapping of (batch_idx, entity_name) -> label
            label_map = {}
            for batch_idx, (p_name, lbl_data) in enumerate(zip(project_names, labels_raw)):
                if isinstance(lbl_data, dict):
                    for entity_name, label_val in lbl_data.items():
                        # Store with batch index for matching
                        label_map[(batch_idx, entity_name)] = float(label_val)

            return g, label_map

        else:
            # Stage 3: Labels are Tensors [Total_Nodes, 8]
            g = batch['graph'].to(self.device)
            targets = {
                'cfg': batch['cfg_labels'].to(self.device, non_blocking=True) if batch['cfg_labels'] is not None else None,
                'ast': batch['ast_labels'].to(self.device, non_blocking=True) if batch['ast_labels'] is not None else None
            }
            return g, targets

    def _maybe_log_cuda_memory(self, *, epoch: int, tag: str):
        if not self.track_cuda_memory or not torch.cuda.is_available():
            return
        try:
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
            max_alloc = int(torch.cuda.max_memory_allocated())
            max_reserved = int(torch.cuda.max_memory_reserved())

            msg = (
                f"[CUDA MEM][{tag}] epoch={epoch} "
                f"alloc={allocated/1024**2:.1f}MB resv={reserved/1024**2:.1f}MB "
                f"max_alloc={max_alloc/1024**2:.1f}MB max_resv={max_reserved/1024**2:.1f}MB"
            )
            self.logger.info(msg)

            # Heuristic warning: reserved grows >256MB between epochs (after warmup).
            if self._last_cuda_reserved_bytes is not None and epoch >= 3:
                delta = reserved - int(self._last_cuda_reserved_bytes)
                if delta > 256 * 1024**2:
                    self.logger.warning(
                        f"[CUDA MEM] Reserved increased by {delta/1024**2:.1f}MB since last epoch; possible caching/leak or batch-size pressure.")
            self._last_cuda_reserved_bytes = reserved
        except Exception:
            pass

    def train(self):
        best_f1 = 0.0
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            epoch_loss = 0
            grad_norms = []

            if self.track_cuda_memory and torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

            batch_count = 0
            for batch in tqdm(self.train_loader, desc=f"Ep {epoch+1} Train"):
                if batch is None:
                    continue

                inputs, label_data = self._prepare_batch(batch)

                # Faster zero_grad (set_to_none) when supported
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    self.optimizer.zero_grad()

                autocast_ctx = (
                    torch.amp.autocast('cuda', dtype=self.amp_dtype)
                    if (self.use_amp and self.device.type == "cuda")
                    else nullcontext()
                )

                with autocast_ctx:
                    preds = self.model({'graph': inputs})

                    loss = 0
                    if self.is_graph_level:
                        loss = self._train_step_graph(preds, label_data)
                    else:
                        loss = self._train_step_node(preds, label_data)

                if loss == 0:
                    continue

                # Backward + step (AMP-aware)
                if isinstance(loss, list):
                    objectives = [loss_item.float() for loss_item in loss]
                    epoch_loss += float(sum(loss_item.item()
                                        for loss_item in objectives))
                    if self.use_amp and self.scaler is not None:
                        scaled = [self.scaler.scale(loss_item)
                                  for loss_item in objectives]
                        if isinstance(self.optimizer, PCGrad):
                            self.optimizer.pc_backward(scaled)
                        else:
                            self.scaler.scale(sum(objectives)).backward()
                    else:
                        if isinstance(self.optimizer, PCGrad):
                            self.optimizer.pc_backward(objectives)
                        else:
                            sum(objectives).backward()
                else:
                    single = loss.float()
                    epoch_loss += float(single.item())
                    if self.use_amp and self.scaler is not None:
                        if isinstance(self.optimizer, PCGrad):
                            self.optimizer.pc_backward(
                                [self.scaler.scale(single)])
                        else:
                            self.scaler.scale(single).backward()
                    else:
                        if isinstance(self.optimizer, PCGrad):
                            self.optimizer.pc_backward([single])
                        else:
                            single.backward()

                # Unscale before clipping (AMP)
                if self.use_amp and self.scaler is not None:
                    try:
                        self.scaler.unscale_(self.optimizer)
                    except Exception:
                        pass

                # Gradient clipping for stability
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.grad_clip_max_norm)
                grad_norms.append(grad_norm.item())

                # CRITICAL: Actually update the weights
                if self.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                batch_count += 1
                # Log first batch to verify training
                if batch_count == 1:
                    if isinstance(loss, list):
                        self.logger.info(
                            f"[GREEN][TRAIN DEBUG] Epoch {epoch+1} Batch 1 Loss: {[loss_item.item() for loss_item in loss]} | Grad Norm: {grad_norm:.4f}")
                    else:
                        self.logger.info(
                            f"[GREEN][TRAIN DEBUG] Epoch {epoch+1} Batch 1 Loss: {loss.item():.4f} | Grad Norm: {grad_norm:.4f}")

                # Step scheduler if using OneCycleLR
                if self.scheduler is not None and hasattr(self.scheduler, 'step') and len(self.scheduler.__class__.__name__) == 'OneCycleLR':
                    self.scheduler.step()

            # Step epoch-based schedulers
            if self.scheduler is not None and self.scheduler.__class__.__name__ == 'CosineAnnealingLR':
                self.scheduler.step()

            # Validation
            val_start_time = time.time()
            val_metrics = self.evaluate(
                self.val_loader, self.fixed_thresholds)
            val_time = time.time() - val_start_time

            train_time = time.time() - epoch_start_time - val_time
            epoch_total_time = time.time() - epoch_start_time

            avg_loss = epoch_loss / max(len(self.train_loader), 1)
            avg_grad_norm = sum(grad_norms) / \
                len(grad_norms) if grad_norms else 0
            self.logger.info(
                f"[GREEN]Epoch {epoch+1}: Loss={avg_loss:.4f} | Val Loss={val_metrics['loss']:.4f} | Val F1={val_metrics['f1']:.4f} | Val AUC={val_metrics['auc']:.4f}" + (f" | Val Hamming={val_metrics['hamming']:.4f}" if self.stage == 3 else "") + f" | Avg Grad Norm={avg_grad_norm:.4f} | Train Time: {train_time:.2f}s | Val Time: {val_time:.2f}s | Total: {epoch_total_time:.2f}s")

            # Step scheduler if using ReduceLROnPlateau
            if self.reduce_on_plateau is not None:
                prev_lr = self.optimizer.param_groups[0]['lr']
                self.reduce_on_plateau.step(val_metrics['loss'])
                current_lr = self.optimizer.param_groups[0]['lr']
                if current_lr != prev_lr:
                    self.logger.info(
                        f"[PURPLE][TRAIN DEBUG] Learning Rate reduced from {prev_lr:.6f} to {current_lr:.6f}")
                else:
                    self.logger.info(
                        f"[PURPLE][TRAIN DEBUG] Current Learning Rate: {current_lr:.6f}")

            # Update history
            self.history["train_loss"].append(
                epoch_loss / len(self.train_loader))
            self.history["val_loss"].append(val_metrics['loss'])
            self.history["val_f1"].append(val_metrics['f1'])
            self.history["val_auc"].append(val_metrics['auc'])
            self.history["val_hamming"].append(val_metrics['hamming'])
            self.history["epoch_times"].append(epoch_total_time)

            # Save history to JSON
            with open(f"{self.log_folder}/history.json", 'w') as f:
                json.dump(self.history, f)

            # Save history plot if method exists
            if hasattr(self, 'save_history_plot'):
                self.save_history_plot()

            # Checkpoint
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                patience_counter = 0
                torch.save(self.model.state_dict(),
                           f"{self.log_folder}/best_model.pth")
                self.logger.info(
                    f"[PURPLE]New best model saved with Val F1={best_f1:.4f} - Patience reset")
                # still update loss bound
                if val_metrics['loss'] < best_loss:
                    best_loss = val_metrics['loss']
                elif val_metrics['loss'] > best_loss:
                    # overfitting case
                    self.logger.info(
                        "Seems begin overfitting - best F1 improved but loss did not.")
                    patience_counter += 1
            elif val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                patience_counter = 0
                self.logger.info(
                    f"[PURPLE] Best Val Loss={best_loss:.4f} - Patience reset")
            else:
                patience_counter += 1
                self.logger.info(
                    f"{patience_counter} / {self.patience} patience used")
                if patience_counter >= self.patience:
                    self.logger.info("Early Stopping")
                    break

            self._maybe_log_cuda_memory(epoch=epoch + 1, tag="end_epoch")

    def _train_step_graph(self, preds, label_map):
        """Training step for stages 1-2: Graph-level classification with one entity per graph."""
        # preds['graph_logits']: [batch_size, 1]
        # label_map: {(batch_idx, entity_name): label} - one per graph

        # Sort by batch_idx to get targets in order
        sorted_keys = sorted(label_map.keys(), key=lambda x: x[0])
        targets = torch.tensor(
            [label_map[k] for k in sorted_keys], dtype=torch.float32).to(self.device)

        logits = preds['graph_logits'].squeeze(-1)  # [batch_size]

        # Support either FocalLoss (legacy) or BCEWithLogitsLoss (preferred for Stage 1/2).
        if isinstance(self.criterion, FocalLoss):
            loss = self.criterion(
                logits.unsqueeze(-1), targets.unsqueeze(-1)
            ).mean()
        else:
            loss = self.criterion(logits, targets)
        return loss

    def _train_step_node(self, preds, label_data):
        """Training step for stage 3: Node-level multilabel classification."""
        targets = label_data
        losses = []

        # Apply label smoothing to prevent overconfidence
        label_smoothing = self.config.label_smoothing

        if targets['cfg'] is not None:
            smoothed_cfg = targets['cfg'] * \
                (1 - label_smoothing) + 0.5 * label_smoothing
            losses.append(self.criterion(
                preds['cfg_logits'], smoothed_cfg).mean())
        if targets['ast'] is not None:
            smoothed_ast = targets['ast'] * \
                (1 - label_smoothing) + 0.5 * label_smoothing
            # Balance AST and CFG losses equally (not 0.5x)
            losses.append(
                self.criterion(preds['ast_logits'], smoothed_ast).mean())
        return losses if losses else 0

    def evaluate(self, loader, fixed_thresholds=None, optimize_thresholds=True):
        self.model.eval()
        all_preds = []
        all_targets = []
        epoch_loss = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                if batch is None:
                    continue
                inputs, label_data = self._prepare_batch(batch)

                autocast_ctx = (
                    torch.amp.autocast('cuda', dtype=self.amp_dtype)
                    if (self.use_amp and self.device.type == "cuda")
                    else nullcontext()
                )

                with autocast_ctx:
                    preds = self.model({'graph': inputs})

                # Compute loss
                if self.is_graph_level:
                    loss = self._train_step_graph(preds, label_data)
                else:
                    losses = self._train_step_node(preds, label_data)
                    loss = sum(losses) if isinstance(losses, list) else losses
                epoch_loss += loss.item() if loss != 0 else 0

                if self.is_graph_level:
                    probs, targets = self._eval_step_graph(preds, label_data)
                else:
                    probs, targets = self._eval_step_node(preds, label_data)

                if probs is None or targets is None:
                    continue

                all_preds.append(probs)
                all_targets.append(targets)

        if not all_preds:
            return {'f1': 0, 'auc': 0, 'loss': 0, 'y_pred': np.array([]), 'y_true': np.array([])}

        y_prob = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()

        # Debug: Log label distribution
        pos_count = y_true.sum()
        neg_count = len(y_true) - pos_count
        self.logger.info(
            f"[YELLOW][EVAL DEBUG] Samples: {len(y_true)} | Positive: {pos_count} ({pos_count/len(y_true)*100:.1f}%) | Negative: {neg_count}")

        # Metrics
        thr = 0.5
        thresholds_used = None
        if self.stage in [1, 2]:
            # Tune threshold for macro-F1 on this eval split (skips if single-class).
            try:
                unique = np.unique(y_true)
                if unique.size >= 2:
                    best_f1 = -1.0
                    best_thr = 0.5
                    for t in np.linspace(self.config.eval_threshold_min, self.config.eval_threshold_max, self.config.eval_threshold_steps):
                        yp = (y_prob > t).astype(int)
                        f1_t = f1_score(
                            y_true, yp, average='macro', zero_division=0)
                        if f1_t > best_f1:
                            best_f1 = f1_t
                            best_thr = float(t)
                    thr = best_thr
                    self.logger.info(
                        f"[YELLOW][EVAL DEBUG] Best threshold={thr:.3f} (macro-F1={best_f1:.4f})")
            except Exception as e:
                self.logger.warning(
                    f"[EVAL DEBUG] Threshold tuning failed: {e}")

            y_pred = (y_prob > thr).astype(int)
        else:
            # Stage 3: per-class thresholds
            num_classes = y_prob.shape[1]
            if fixed_thresholds is not None:
                thresholds_used = np.array(fixed_thresholds, dtype=float)
                self.logger.info(
                    f"[EVAL DEBUG] Using fixed thresholds: {thresholds_used}")
            elif optimize_thresholds:
                thresholds_used = np.zeros(num_classes)
                for class_idx in range(num_classes):
                    y_true_class = y_true[:, class_idx]
                    y_prob_class = y_prob[:, class_idx]
                    if y_true_class.sum() == 0:
                        # conservative when no positives
                        thresholds_used[class_idx] = 0.9
                        continue
                    best_score = -1.0
                    best_thr = 0.5
                    # Wider range for better tuning
                    for t in np.linspace(self.config.eval_threshold_min, self.config.eval_threshold_max, self.config.eval_threshold_steps):
                        yp = (y_prob_class >= t).astype(int)
                        if yp.sum() == 0:
                            continue
                        f1_t = f1_score(y_true_class, yp,
                                        average='binary', zero_division=0)
                        prec_t = precision_score(
                            y_true_class, yp, zero_division=0)
                        score = self.config.eval_score_f1_weight * f1_t + \
                            self.config.eval_score_prec_weight * prec_t  # Favor F1 more
                        if score > best_score:
                            best_score = score
                            best_thr = float(t)
                    # Lower min threshold for more predictions
                    thresholds_used[class_idx] = max(
                        best_thr, self.config.eval_min_threshold)
                self.logger.info(
                    f"[EVAL DEBUG] Per-class thresholds (optimized): {thresholds_used}")
                self.best_thresholds = torch.tensor(
                    thresholds_used, dtype=torch.float32)
            else:
                thresholds_used = np.full(num_classes, 0.5)

            thresholds_used = np.array(thresholds_used, dtype=float)
            y_pred = (y_prob >= thresholds_used).astype(int)

        # Debug: Log prediction distribution
        pred_pos = y_pred.sum()
        pred_neg = len(y_pred) - pred_pos
        self.logger.info(
            f"[YELLOW][EVAL DEBUG] Predictions: Positive: {pred_pos} ({pred_pos/len(y_pred)*100:.1f}%) | Negative: {pred_neg}")
        self.logger.info(
            f"[EVAL DEBUG] Prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}] | Mean: {y_prob.mean():.4f}")

        # Use weighted F1 to handle extreme imbalance better (weights by class support)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        # Compute AUC safely for multilabel
        if self.stage == 3:
            auc_scores = []
            for class_idx in range(y_prob.shape[1]):
                y_true_class = y_true[:, class_idx]
                y_prob_class = y_prob[:, class_idx]
                if len(np.unique(y_true_class)) == 2:  # Both classes present
                    try:
                        auc_class = roc_auc_score(y_true_class, y_prob_class)
                        auc_scores.append(auc_class)
                    except Exception:
                        pass
            auc = np.mean(auc_scores) if auc_scores else 0.5
            self.logger.info(f"[EVAL DEBUG] Per-Class AUC: {auc_scores}")
        else:
            try:
                auc = roc_auc_score(y_true, y_prob, average='macro')
            except Exception:
                auc = 0.5  # Fail gracefully if only one class present

        # Hamming score for Stage 3
        hamming = 1 - hamming_loss(y_true, y_pred) if self.stage == 3 else 0

        # Classification report
        if self.stage == 3:
            # Per-label report for multilabel Stage 3
            report = classification_report(
                y_true, y_pred, target_names=OWASP_VULN, zero_division=0, digits=6)
            self.logger.info(
                f"Per-Label Classification Report (Stage 3):\n{report}")
            self.logger.info(f"Hamming Loss: {hamming:.6f}")
        else:
            # Binary report for Stages 1/2
            report = classification_report(y_true, y_pred, target_names=[
                'Safe', 'Vuln'], zero_division=0, digits=6)
            self.logger.info(f"Classification Report:\n{report}")

        return {
            'f1': f1,
            'auc': auc,
            'hamming': hamming,
            'loss': epoch_loss / len(loader),
            'y_pred': y_pred,
            'y_true': y_true,
            'thresholds': thresholds_used,
            'y_prob': y_prob
        }

    def _eval_step_graph(self, preds, label_map):
        """Evaluation step for stages 1-2: Graph-level classification."""
        sorted_keys = sorted(label_map.keys(), key=lambda x: x[0])
        targets = torch.tensor(
            [label_map[k] for k in sorted_keys], dtype=torch.float32).to(self.device)

        probs = torch.sigmoid(
            preds['graph_logits'].squeeze(-1))  # [batch_size]
        return probs.cpu(), targets.cpu()

    def _eval_step_node(self, preds, label_data):
        """Evaluation step for stage 3: Node-level classification."""
        targets = label_data
        if targets['cfg'] is None and targets['ast'] is None:
            return None, None

        all_probs = []
        all_targets = []

        if targets['cfg'] is not None:
            all_probs.append(torch.sigmoid(preds['cfg_logits']))
            all_targets.append(targets['cfg'])

        if targets['ast'] is not None:
            all_probs.append(torch.sigmoid(preds['ast_logits']))
            all_targets.append(targets['ast'])

        if all_probs:
            combined_probs = torch.cat(all_probs, dim=0)
            combined_targets = torch.cat(all_targets, dim=0)
            return combined_probs.cpu(), combined_targets.cpu()

        return None, None

    def test(self):
        """Evaluate on test set."""
        return self.evaluate(self.test_loader, self.fixed_thresholds)
