"""
Deep Label Leak Detection Script
==================================
Performs comprehensive checks for potential label leaks in the ChainGuard dataset.
"""

import logging
import sys
import os
import hashlib
import numpy as np
import torch
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from experiments.trainer import Trainer
from experiments.dataset import CustomDataset
from experiments.utils.logger import setup_logger

logger = setup_logger("Logs/label_leak_detection.log", logging.INFO)


class LabelLeakDetector:
    """Comprehensive label leak detection for vulnerability detection models."""
    
    def __init__(self):
        self.trainer = None
        self.dataset = None
        
    def initialize(self):
        """Initialize trainer and dataset."""
        logger.info("=" * 80)
        logger.info("LABEL LEAK DETECTION - INITIALIZATION")
        logger.info("=" * 80)
        
        self.trainer = Trainer(rand_seed=42)
        self.dataset = self.trainer.dataset
        
        logger.info(f"Dataset size: {len(self.dataset)}")
        logger.info(f"Train size: {len(self.trainer.train_loader.dataset)}")
        logger.info(f"Val size: {len(self.trainer.val_loader.dataset)}")
        logger.info(f"Test size: {len(self.trainer.test_loader.dataset)}")
        
    # =========================================================================
    # CHECK 1: Label Distribution Similarity
    # =========================================================================
    
    def check_label_distribution(self):
        """Check if train/val/test have suspiciously similar label distributions."""
        logger.info("\n" + "=" * 80)
        logger.info("CHECK 1: Label Distribution Similarity Analysis")
        logger.info("=" * 80)
        
        distributions = {}
        
        for split_name, loader in [
            ('train', self.trainer.train_loader),
            ('val', self.trainer.val_loader),
            ('test', self.trainer.test_loader)
        ]:
            logger.info(f"\nAnalyzing {split_name} split...")
            
            all_code_labels = []
            all_cg_labels = []
            all_cfg_labels = []
            
            for batch in tqdm(loader, desc=f"Processing {split_name}"):
                if batch['labels'] is not None:
                    all_code_labels.append(batch['labels'].cpu().numpy())
                if batch['cg_labels'] is not None:
                    all_cg_labels.append(batch['cg_labels'].cpu().numpy())
                if batch['cfg_labels'] is not None:
                    all_cfg_labels.append(batch['cfg_labels'].cpu().numpy())
            
            # Compute distributions
            code_dist = self._compute_distribution(all_code_labels) if all_code_labels else None
            cg_dist = self._compute_distribution(all_cg_labels) if all_cg_labels else None
            cfg_dist = self._compute_distribution(all_cfg_labels) if all_cfg_labels else None
            
            distributions[split_name] = {
                'code': code_dist,
                'cg': cg_dist,
                'cfg': cfg_dist
            }
            
            # Log distributions
            if code_dist:
                logger.info(f"  Code labels - Positive rate: {code_dist['pos_rate']:.4f}")
                logger.info(f"  Code labels - Per-class rates: {code_dist['per_class_rates']}")
            if cg_dist:
                logger.info(f"  CG labels - Positive rate: {cg_dist['pos_rate']:.4f}")
            if cfg_dist:
                logger.info(f"  CFG labels - Positive rate: {cfg_dist['pos_rate']:.4f}")
        
        # Compare distributions
        logger.info("\n" + "-" * 80)
        logger.info("DISTRIBUTION COMPARISON:")
        logger.info("-" * 80)
        
        # Check code label similarity
        train_code_pos = distributions['train']['code']['pos_rate']
        val_code_pos = distributions['val']['code']['pos_rate']
        test_code_pos = distributions['test']['code']['pos_rate']
        
        train_val_diff = abs(train_code_pos - val_code_pos)
        train_test_diff = abs(train_code_pos - test_code_pos)
        
        logger.info(f"Code labels:")
        logger.info(f"  Train positive rate: {train_code_pos:.4f}")
        logger.info(f"  Val positive rate: {val_code_pos:.4f}")
        logger.info(f"  Test positive rate: {test_code_pos:.4f}")
        logger.info(f"  Train-Val difference: {train_val_diff:.4f}")
        logger.info(f"  Train-Test difference: {train_test_diff:.4f}")
        
        # Warning thresholds
        if train_val_diff < 0.02:
            logger.warning("⚠️  Train/Val distributions are VERY similar (diff < 0.02)")
            logger.warning("   This could indicate label leakage or insufficient stratification")
        
        if train_test_diff < 0.02:
            logger.warning("⚠️  Train/Test distributions are VERY similar (diff < 0.02)")
            logger.warning("   This could indicate label leakage")
        
        return distributions
    
    def _compute_distribution(self, label_list):
        """Compute label distribution statistics."""
        if not label_list:
            return None
        
        all_labels = np.concatenate([l.reshape(-1, l.shape[-1]) for l in label_list], axis=0)
        
        # Overall positive rate
        pos_rate = (all_labels > 0).mean()
        
        # Per-class positive rates
        per_class_rates = (all_labels > 0).mean(axis=0)
        
        return {
            'pos_rate': float(pos_rate),
            'per_class_rates': per_class_rates.tolist(),
            'total_samples': all_labels.shape[0]
        }
    
    # =========================================================================
    # CHECK 2: Sample Overlap Detection
    # =========================================================================
    
    def check_sample_overlap(self):
        """Detect if same samples appear in multiple splits."""
        logger.info("\n" + "=" * 80)
        logger.info("CHECK 2: Sample Overlap Detection")
        logger.info("=" * 80)
        
        def get_sample_hashes(loader, split_name):
            """Generate hashes for samples in a loader."""
            hashes = []
            for i, batch in enumerate(tqdm(loader, desc=f"Hashing {split_name}")):
                # Hash code features
                if batch['code'] is not None:
                    code_bytes = batch['code'].cpu().numpy().tobytes()
                    code_hash = hashlib.md5(code_bytes).hexdigest()
                    hashes.append(('code', i, code_hash))
                
                # Hash graph structure
                if batch['graph'] is not None:
                    graph = batch['graph']
                    # Create hash from node counts and edge counts
                    graph_sig = f"{graph.num_nodes()}_{graph.num_edges()}"
                    for ntype in graph.ntypes:
                        graph_sig += f"_{ntype}:{graph.num_nodes(ntype)}"
                    graph_hash = hashlib.md5(graph_sig.encode()).hexdigest()
                    hashes.append(('graph', i, graph_hash))
            
            return hashes
        
        train_hashes = get_sample_hashes(self.trainer.train_loader, 'train')
        val_hashes = get_sample_hashes(self.trainer.val_loader, 'val')
        test_hashes = get_sample_hashes(self.trainer.test_loader, 'test')
        
        # Extract just the hash values for comparison
        train_code_hashes = set(h[2] for h in train_hashes if h[0] == 'code')
        val_code_hashes = set(h[2] for h in val_hashes if h[0] == 'code')
        test_code_hashes = set(h[2] for h in test_hashes if h[0] == 'code')
        
        train_graph_hashes = set(h[2] for h in train_hashes if h[0] == 'graph')
        val_graph_hashes = set(h[2] for h in val_hashes if h[0] == 'graph')
        test_graph_hashes = set(h[2] for h in test_hashes if h[0] == 'graph')
        
        # Check overlaps
        code_train_val = len(train_code_hashes & val_code_hashes)
        code_train_test = len(train_code_hashes & test_code_hashes)
        code_val_test = len(val_code_hashes & test_code_hashes)
        
        graph_train_val = len(train_graph_hashes & val_graph_hashes)
        graph_train_test = len(train_graph_hashes & test_graph_hashes)
        graph_val_test = len(val_graph_hashes & test_graph_hashes)
        
        logger.info(f"\nCode feature overlap:")
        logger.info(f"  Train-Val: {code_train_val} samples")
        logger.info(f"  Train-Test: {code_train_test} samples")
        logger.info(f"  Val-Test: {code_val_test} samples")
        
        logger.info(f"\nGraph structure overlap:")
        logger.info(f"  Train-Val: {graph_train_val} samples")
        logger.info(f"  Train-Test: {graph_train_test} samples")
        logger.info(f"  Val-Test: {graph_val_test} samples")
        
        if code_train_val > 0 or code_train_test > 0:
            logger.error("🔴 CRITICAL: Code feature overlap detected!")
            logger.error("   Same code samples appear in multiple splits")
        
        if graph_train_val > 0 or graph_train_test > 0:
            logger.error("🔴 CRITICAL: Graph structure overlap detected!")
            logger.error("   Same graphs appear in multiple splits")
        
        if code_train_val == 0 and code_train_test == 0 and graph_train_val == 0 and graph_train_test == 0:
            logger.info("✅ No sample overlap detected")
    
    # =========================================================================
    # CHECK 3: Graph-Label Alignment Validation
    # =========================================================================
    
    def check_graph_label_alignment(self):
        """Verify that graph structures match label dimensions."""
        logger.info("\n" + "=" * 80)
        logger.info("CHECK 3: Graph-Label Alignment Validation")
        logger.info("=" * 80)
        
        misalignment_count = 0
        total_checked = 0
        
        for split_name, loader in [
            ('train', self.trainer.train_loader),
            ('val', self.trainer.val_loader),
            ('test', self.trainer.test_loader)
        ]:
            logger.info(f"\nChecking {split_name} split...")
            
            for i, batch in enumerate(tqdm(loader, desc=f"Validating {split_name}")):
                if i >= 10:  # Check first 10 batches
                    break
                
                graph = batch['graph']
                cg_labels = batch['cg_labels']
                cfg_labels = batch['cfg_labels']
                cg_lengths = batch['cg_lengths']
                cfg_lengths = batch['cfg_lengths']
                
                # Get actual node counts per graph in batch
                batch_size = graph.batch_size
                func_nodes_per_graph = graph.batch_num_nodes('function').tolist()
                block_nodes_per_graph = graph.batch_num_nodes('block').tolist()
                
                # Check alignment
                for b in range(batch_size):
                    actual_funcs = func_nodes_per_graph[b]
                    expected_funcs = cg_lengths[b].item()
                    
                    actual_blocks = block_nodes_per_graph[b]
                    expected_blocks = cfg_lengths[b].item()
                    
                    if actual_funcs != expected_funcs:
                        logger.warning(f"  Batch {i}, Sample {b}: CG mismatch - graph has {actual_funcs} functions, labels have {expected_funcs}")
                        misalignment_count += 1
                    
                    if actual_blocks != expected_blocks:
                        logger.warning(f"  Batch {i}, Sample {b}: CFG mismatch - graph has {actual_blocks} blocks, labels have {expected_blocks}")
                        misalignment_count += 1
                    
                    total_checked += 2
        
        logger.info(f"\nAlignment check complete:")
        logger.info(f"  Total checks: {total_checked}")
        logger.info(f"  Misalignments: {misalignment_count}")
        
        if misalignment_count > 0:
            logger.error(f"🔴 CRITICAL: {misalignment_count} graph-label misalignments detected!")
            logger.error("   Model may be learning from wrong labels")
        else:
            logger.info("✅ All graph-label alignments correct")
    
    # =========================================================================
    # CHECK 4: Node Feature Inspection for Label Leakage
    # =========================================================================
    
    def check_node_features_for_labels(self):
        """Check if node features inadvertently encode vulnerability labels."""
        logger.info("\n" + "=" * 80)
        logger.info("CHECK 4: Node Feature Inspection for Label Leakage")
        logger.info("=" * 80)
        
        # Sample a few graphs and check node features
        logger.info("Inspecting node features in first batch...")
        
        batch = next(iter(self.trainer.train_loader))
        graph = batch['graph']
        
        # Check each node type
        for ntype in graph.ntypes:
            if 'feat' in graph.nodes[ntype].data:
                features = graph.nodes[ntype].data['feat']
                logger.info(f"\nNode type '{ntype}':")
                logger.info(f"  Feature shape: {features.shape}")
                logger.info(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")
                logger.info(f"  Feature mean: {features.mean():.4f}")
                logger.info(f"  Feature std: {features.std():.4f}")
                
                # Check for suspicious patterns
                # If features are all 0 or 1, might be one-hot encoded labels
                unique_vals = torch.unique(features)
                if len(unique_vals) <= 10:
                    logger.info(f"  Unique values: {unique_vals.tolist()[:10]}")
                
                # Check if feature dimensions match vulnerability count (9)
                if features.shape[-1] == 9:
                    logger.warning(f"  ⚠️  Feature dimension ({features.shape[-1]}) matches vulnerability count!")
                    logger.warning(f"     This could indicate labels are embedded in features")
        
        # Check labels
        cg_labels = batch['cg_labels']
        cfg_labels = batch['cfg_labels']
        code_labels = batch['labels']
        
        logger.info(f"\nLabel shapes:")
        logger.info(f"  Code labels: {code_labels.shape}")
        logger.info(f"  CG labels: {cg_labels.shape}")
        logger.info(f"  CFG labels: {cfg_labels.shape}")
    
    # =========================================================================
    # CHECK 5: Project Name Correspondence Validation
    # =========================================================================
    
    def check_project_correspondence(self):
        """Verify that graph and code data are from the same projects."""
        logger.info("\n" + "=" * 80)
        logger.info("CHECK 5: Project Name Correspondence Validation")
        logger.info("=" * 80)
        
        logger.info("Checking first 50 samples for project name consistency...")
        
        mismatches = []
        for i in range(min(50, len(self.dataset))):
            try:
                # Access raw dataset items
                graph_item = self.dataset.dataset_graph[i]
                code_item = self.dataset.dataset_code[i]
                
                if isinstance(graph_item, tuple) and isinstance(code_item, tuple):
                    g_project_name = graph_item[0]
                    c_project_name = code_item[0]
                    
                    if g_project_name != c_project_name:
                        mismatches.append((i, g_project_name, c_project_name))
                        logger.warning(f"  Sample {i}: graph='{g_project_name}', code='{c_project_name}'")
            
            except Exception as e:
                logger.warning(f"  Could not check sample {i}: {e}")
        
        if mismatches:
            logger.error(f"🔴 CRITICAL: {len(mismatches)} project name mismatches detected!")
            logger.error("   Graph and code data are not synchronized")
        else:
            logger.info("✅ All project names match between graph and code")
    
    # =========================================================================
    # CHECK 6: Label Statistics and Anomalies
    # =========================================================================
    
    def check_label_statistics(self):
        """Analyze label statistics for anomalies."""
        logger.info("\n" + "=" * 80)
        logger.info("CHECK 6: Label Statistics and Anomaly Detection")
        logger.info("=" * 80)
        
        logger.info("Collecting label statistics from entire dataset...")
        
        all_code_labels = []
        all_cg_labels = []
        all_cfg_labels = []
        
        for i in tqdm(range(len(self.dataset)), desc="Scanning dataset"):
            try:
                item = self.dataset[i]
                
                if item['labels'] is not None:
                    all_code_labels.append(item['labels'].numpy())
                if item['cg_labels'] is not None:
                    all_cg_labels.append(item['cg_labels'].numpy())
                if item['cfg_labels'] is not None:
                    all_cfg_labels.append(item['cfg_labels'].numpy())
            except Exception as e:
                logger.warning(f"Error reading sample {i}: {e}")
        
        # Analyze code labels
        if all_code_labels:
            logger.info("\nCode labels analysis:")
            all_code = np.concatenate([l.reshape(-1, l.shape[-1]) for l in all_code_labels], axis=0)
            
            # Check for each vulnerability class
            for vuln_idx in range(all_code.shape[1]):
                pos_count = (all_code[:, vuln_idx] > 0).sum()
                pos_rate = pos_count / all_code.shape[0]
                logger.info(f"  Vuln class {vuln_idx}: {pos_count} positive ({pos_rate:.4%})")
                
                # Flag suspicious patterns
                if pos_rate > 0.95:
                    logger.warning(f"    ⚠️  Class {vuln_idx} is almost always positive (>{pos_rate:.4%})")
                elif pos_rate < 0.05 and pos_rate > 0:
                    logger.warning(f"    ⚠️  Class {vuln_idx} is very rare (<{pos_rate:.4%})")
            
            # Check for all-zero or all-one samples
            all_zero = (all_code.sum(axis=1) == 0).sum()
            all_one = (all_code == 1).all(axis=1).sum()
            
            logger.info(f"\n  All-zero samples: {all_zero} ({all_zero/len(all_code):.4%})")
            logger.info(f"  All-one samples: {all_one} ({all_one/len(all_code):.4%})")
            
            if all_zero > len(all_code) * 0.9:
                logger.error("🔴 CRITICAL: >90% of labels are all-zero!")
                logger.error("   Labels may not be properly loaded")
    
    # =========================================================================
    # Main Execution
    # =========================================================================
    
    def run_all_checks(self):
        """Run all label leak detection checks."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPREHENSIVE LABEL LEAK DETECTION")
        logger.info("=" * 80)
        
        self.initialize()
        
        # Run all checks
        self.check_label_distribution()
        self.check_sample_overlap()
        self.check_graph_label_alignment()
        self.check_node_features_for_labels()
        self.check_project_correspondence()
        self.check_label_statistics()
        
        logger.info("\n" + "=" * 80)
        logger.info("LABEL LEAK DETECTION COMPLETE")
        logger.info("=" * 80)
        logger.info("Check the log file for detailed results: Logs/label_leak_detection.log")


def main():
    """Main entry point."""
    detector = LabelLeakDetector()
    detector.run_all_checks()


if __name__ == "__main__":
    main()
