import os
import time
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from base_trainer import BaseTrainer

class TrainerWithAnalysis(BaseTrainer):
    """
    Extended trainer that adapts analysis and visualization based on the 
    Stage (Graph vs Node Classification).
    """

    def __init__(self, do_test_data=True, **kwargs):
        self.do_test_data = do_test_data
        super().__init__(**kwargs)
        
        # Store training start time for efficiency tracking
        self.start_time = time.time()
        self.training_time = 0

    def run_pre_train_analysis(self):
        """Run comprehensive checks before training starts."""
        if self.do_test_data:
            self.test_data_loader()
        self._log_model_summary()

    def test_data_loader(self):
        """Analyze batch composition for Train/Val/Test."""
        self.logger.info("=" * 40)
        self.logger.info(f"Testing DataLoader (Stage {self.stage})")
        self.logger.info("=" * 40)

        for name, loader in [("TRAIN", self.train_loader), ("VAL", self.val_loader)]:
            self.logger.info(f"\n# {name} DATASET")
            self._analyze_dataloader(loader)

    def _analyze_dataloader(self, loader):
        """Robust analysis handling both Tensor labels (S3) and Dict labels (S1/2)."""
        try:
            pos_samples = 0
            total_samples = 0
            
            # Check first batch only for brevity
            for batch_idx, batch in enumerate(loader):
                if batch is None: continue
                
                # 1. Graph Stats
                if self.stage in [1, 2]:
                    # No graph, use embeddings
                    batch_size = batch['embeddings'].shape[0]
                    
                    if batch_idx == 0:
                        self.logger.info(f"  Batch 0 Stats:")
                        self.logger.info(f"    Embeddings: {batch_size}")
                        self.logger.info(f"    Embedding Dim: {batch['embeddings'].shape[1]}")
                        self.logger.info(f"    Device: {batch['embeddings'].device}")
                else:
                    g = batch['graph']
                    batch_size = g.batch_size
                    
                    if batch_idx == 0:
                        self.logger.info(f"  Batch 0 Stats:")
                        self.logger.info(f"    Graphs: {batch_size}")
                        self.logger.info(f"    Nodes: {g.num_nodes()} (Avg {g.num_nodes()/batch_size:.1f})")
                        self.logger.info(f"    Edges: {g.num_edges()}")
                        self.logger.info(f"    Device: {g.device}")

                # 2. Label Stats
                # Extract logic similar to BaseTrainer._prepare_batch
                if self.stage in [1, 2]:
                    # Graph Level: Label is in batch['graph_labels'] as list/dict
                    raw = batch['graph_labels']
                    
                    # DEBUG: Log first batch labels to understand structure
                    if batch_idx == 0:
                        self.logger.info(f"  DEBUG - Label dicts for first 3 GRAPHS (each graph has multiple contracts):")
                        for i, graph_labels in enumerate(raw[:3] if len(raw) >= 3 else raw):
                            num_contracts = len(graph_labels) if isinstance(graph_labels, dict) else 1
                            num_vuln = sum(1 for v in graph_labels.values() if v == 1) if isinstance(graph_labels, dict) else (1 if graph_labels == 1 else 0)
                            self.logger.info(f"    Graph {i+1}: {num_contracts} contracts, {num_vuln} vulnerable")
                            if isinstance(graph_labels, dict):
                                vuln_contracts = [k for k, v in graph_labels.items() if v == 1]
                                if vuln_contracts:
                                    self.logger.info(f"      Vulnerable: {vuln_contracts}")
                    
                    # Convert to binary list
                    curr_pos = 0
                    for item in raw:
                        if int(item) == 1:
                            curr_pos += 1
                    
                    pos_samples += curr_pos
                    total_samples += len(raw)
                    
                else:
                    # Node Level: Label is Tensor [N, 8]
                    cfg_lbls = batch.get('cfg_labels')
                    ast_lbls = batch.get('ast_labels')

                    # Count vulnerable nodes across both views
                    if cfg_lbls is not None:
                        pos_samples += (cfg_lbls.sum(dim=1) > 0).sum().item()
                        total_samples += cfg_lbls.shape[0]

                    if ast_lbls is not None:
                        pos_samples += (ast_lbls.sum(dim=1) > 0).sum().item()
                        total_samples += ast_lbls.shape[0]

            # Summary
            if total_samples > 0:
                pos_pct = (pos_samples / total_samples) * 100
                entity = "GRAPHS" if self.stage in [1, 2] else "NODES"
                self.logger.info(f"  Total {entity}: {total_samples}")
                self.logger.info(f"  Vulnerable {entity}: {pos_samples} ({pos_pct:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            traceback.print_exc()

    def _log_model_summary(self):
        """Log model parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model Summary: {total_params:,} Total Params ({trainable_params:,} Trainable)")

    def visualize_results(self, test_metrics=None):
        """Generate Stage-specific visualizations."""
        try:
            sns.set_theme(style="whitegrid")
            os.makedirs(f"{self.log_folder}/viz", exist_ok=True)

            # 1. Training History
            if self.history["train_loss"]:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                # Loss
                axes[0].plot(self.history["train_loss"], label="Train")
                axes[0].plot(self.history["val_loss"], label="Val")
                axes[0].set_title(f"Stage {self.stage} Loss")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Loss")
                axes[0].legend()

                # Metrics
                axes[1].plot(self.history["val_f1"], label="Macro F1", color="green")
                axes[1].plot(self.history["val_auc"], label="AUC", color="purple")
                axes[1].set_title("Validation Metrics")
                axes[1].set_xlabel("Epoch")
                axes[1].legend()
                
                plt.savefig(f"{self.log_folder}/viz/history_stage{self.stage}.png")
                plt.close()

            # 2. Confusion Matrix (Binary View)
            if test_metrics and "y_pred" in test_metrics:
                y_true = test_metrics["y_true"]
                y_pred = test_metrics["y_pred"]
                
                # Flatten Multi-label to Binary (Vuln vs Benign) for visualization
                if y_true.ndim > 1:
                    y_true = (y_true.sum(axis=1) > 0).astype(int)
                    y_pred = (y_pred.sum(axis=1) > 0).astype(int)

                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Vuln"])
                
                fig, ax = plt.subplots(figsize=(6, 6))
                disp.plot(cmap="Blues", ax=ax, colorbar=False)
                plt.title(f"Stage {self.stage} Confusion Matrix")
                plt.savefig(f"{self.log_folder}/viz/cm_stage{self.stage}.png")
                plt.close()

        except Exception as e:
            self.logger.warning(f"Viz error: {e}")

    def train(self):
        """Wrapper to track time."""
        self.run_pre_train_analysis()
        super().train()
        self.training_time = time.time() - self.start_time

    def test(self):
        """Wrapper to visualize after test."""
        metrics = super().test()
        self.visualize_results(metrics)
        return metrics


class CascadeOrchestrator:
    """Manages the sequential training of the 3-Stage Pipeline."""
    
    def __init__(self):
        self.results = {}
        self.stages = [1, 2, 3] # Gatekeeper -> Locator -> Specialist
        
    def run(self):
        print("\n" + "#"*60)
        print("STARTING 3-STAGE CASCADE TRAINING PIPELINE")
        print("#"*60 + "\n")

        for stage in self.stages:
            print(f"\n{'='*40}")
            print(f"LAUNCHING STAGE {stage}")
            print(f"{'='*40}")
            
            try:
                # 1. Initialize Trainer for this stage
                # Note: Adjust batch_size or params per stage if needed
                bs = 16 if stage < 3 else 8 # Stage 3 graphs are smaller but more nodes? Adjust as needed.
                trainer = TrainerWithAnalysis(
                    stage=stage,
                    batch_size=bs,
                    num_epochs=20, # Adjust epochs
                    do_test_data=True
                )
                
                # 2. Train
                trainer.train()
                
                # 3. Test
                metrics = trainer.test()
                
                # 4. Store Results
                self.results[stage] = {
                    "metrics": metrics,
                    "time": trainer.training_time,
                    "params": sum(p.numel() for p in trainer.model.parameters())
                }
                
                # Free memory
                del trainer
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"!!! CRITICAL FAILURE IN STAGE {stage} !!!")
                traceback.print_exc()
                break

        self.print_summary()

    def print_summary(self):
        """Prints a comparison table of the cascade efficiency."""
        print("\n" + "="*80)
        print("CASCADE EFFICIENCY SUMMARY")
        print("="*80)
        
        # Header
        print(f"{'Stage':<10} | {'Role':<15} | {'F1 Score':<10} | {'AUC':<10} | {'Recall':<10} | {'Time (m)':<10}")
        print("-" * 75)
        
        roles = {1: "Gatekeeper", 2: "Locator", 3: "Specialist"}
        
        for stage in self.stages:
            if stage not in self.results: continue
            
            data = self.results[stage]
            m = data["metrics"]
            
            # Recall approximation (True Pos Pct captures calibration, but let's assume we want pure recall if available)
            # BaseTrainer calculates macro metrics. For Stage 1 (Gatekeeper), Recall is the most critical metric.
            # We can infer it from the CM or assume High F1 implies good recall.
            # Here we just list F1/AUC.
            
            time_mins = data["time"] / 60
            
            print(f"{stage:<10} | {roles[stage]:<15} | {m['f1']:.4f}     | {m['auc']:.4f}     | {'--':<10} | {time_mins:.1f}")

        print("-" * 75)
        print("Note: In a deployed cascade, Stage 1 filters ~90% of traffic,")
        print("      preventing Stage 3 (Heavy) from running on benign contracts.")
        print("="*80 + "\n")


if __name__ == "__main__":
    orchestrator = CascadeOrchestrator()
    orchestrator.run()