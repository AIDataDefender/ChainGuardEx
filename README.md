# ChainGuard (ChainGuardEx)

[![python](https://img.shields.io/badge/python-3.12.3-blue)](https://www.python.org/)
[![slither](https://img.shields.io/badge/slither-0.11.3-orange)](https://github.com/crytic/slither)
[![dgl](https://img.shields.io/badge/dgl-2.4.0+cu12.1-green)](https://www.dgl.ai/)
[![pytorch](https://img.shields.io/badge/pytorch-2.4.0_cu12.1-orange)](https://pytorch.org/get-started/previous-versions/)

<p align="center">
<img src="./resources/logo.png" alt="ChainGuardEx Logo" width="225" height="225" class="center">
</p

ChainGuard is a graph-learning framework for automated vulnerability detection in Ethereum smart contracts. It trains Graph Neural Networks (GNNs) over Code Property Graphs (CPGs) extracted from smart contract code and supports both graph-level classification and fine-grained (node-level) vulnerability localization.

## What’s in this repo

- **3-stage pipeline** (implemented in `experiments/`):
  - **Stage 1 (Graph-level)**: binary classification (vulnerable vs benign).
  - **Stage 2 (Graph-level)**: binary classification with heavier imbalance handling (oversampling/undersampling).
  - **Stage 3 (Node-level)**: multi-label classification (8 OWASP categories).
- **Models**:
  - **Default model**: `proto_3` (when `model_type=None`).
  - **Baselines**: `GCN`, `GraphSAGE`, `GIN`, `GATv2` (see `experiments/models/baseline_X.py`).
- **Analysis + plots**: training logs + metrics visualizations saved under each run folder in `Logs/.../viz/`.

## Tested env

CPU Ryzen7 8745H 8C-18T 3.8GHz, 16GB RAM, RTX4060 Laptop 8GB VRAM

WSL2 (Ubuntu24.04 LTS)
- [Python 3.12.3](https://www.python.org/downloads/release/python-3123/)
- [Pytorch](https://pytorch.org/get-started/previous-versions/)/ [DGL](https://www.dgl.ai/pages/start.html) 2.4.0 + CUDA 12.1
- Transformers 4.44.2
- Slither 0.11.3

## Data / paths (important)

Training loads **3 sources** by default in `experiments/base_trainer.py`:
**Due to large data when processing upto 100GB+ it is infeasible to save in same drive**
- `DAppSCAN` (currently hard-coded to `/mnt/d/KLTN2/save_data2`) - est 86.4 GB after processed
- `MANDO` (currently hard-coded to `/mnt/d/KLTN2/save_data3`) - est 5.50 GB after processed
- `EtherScanIO` (defaults to `./save_data_splits/split_0`) - est 40.6 GB after processed

Will likely need to edit `BaseTrainer.load_data()` in `experiments/base_trainer.py` so `load_dir=...` points to your local folders.


***You may find processed data for fast testing [here](https://drive.google.com/drive/folders/1xdQXXF1RrpZBhBTqVUG-sXGBriwcJBNR?usp=sharing)***
## Quickstart

### 1) Train (with analysis + plots)

Run from the repo root:

```bash
python experiments\main_trainer_analysis.py --stage 1
```

Run multiple stages:

```bash
python experiments\main_trainer_analysis.py --stage 1 2 3
```

Train a specific baseline model (or multiple):

```bash
python experiments\main_trainer_analysis.py --stage 3 --model-types GCN
python experiments\main_trainer_analysis.py --stage 3 --model-types GCN GIN
```

Run all baseline variants defined in `baseline_X.py`:

```bash
python experiments\main_trainer_analysis.py --stage 1 --run-all-models
```

Run default model + all baselines:

```bash
python experiments\main_trainer_analysis.py --run-full
```

### 2) Ablations / relation pruning

```bash
# Drop one node type (keeps schema but sets node count to 0)
python experiments\main_trainer_analysis.py --stage 3 --ablate-node-type cfg
python experiments\main_trainer_analysis.py --stage 3 --ablate-node-type ast

# Drop only AST<->CFG cross edges
python experiments\main_trainer_analysis.py --stage 3 --drop-cross-edges

# Keep or drop specific relation names (comma-separated)
python experiments\main_trainer_analysis.py --stage 3 --keep-relations "cf,df"
python experiments\main_trainer_analysis.py --stage 3 --drop-relations "call,return_call"
```

### 3) Evaluate saved checkpoints

`experiments/evaluate.py` is a convenience script that loads checkpoints from `Logs/.../best_model.pth`, runs evaluation, and writes figures into the corresponding `viz/` folders.

Edit the `model_paths` dict in `experiments/evaluate.py` to point at the checkpoints you want, then run:

```bash
python experiments\evaluate.py
```

## DAppSCAN preprocessing (optional)

The `Data/DAppSCAN/` folder contains scripts used to extract and preprocess graphs.

```bash
python Data\DAppSCAN\b1_GraphExtractor.py
python Data\DAppSCAN\e5_preprocess_data.py
```

## Outputs

- Training runs are stored under `Logs/Stage{N}_YYYYMMDD_HHMM/`.
- Each run typically includes:
  - `best_model.pth` (model weights)
  - `history.json` (loss/metrics curves)
  - `viz/` (confusion matrices, ROC curves, history plots, JSON stats)

## Project layout

```
.
├── experiments/
│   ├── base_trainer.py            # training loop + dataset wiring
│   ├── main_trainer_analysis.py   # CLI runner (stages/baselines/ablations) # MAIN ENTRY
│   ├── evaluate.py                # checkpoint evaluation helper
│   ├── dataset.py                 # dataset loading + graph standardization
│   ├── cpg_processor.py           # CPG processing utilities
│   ├── models/
│   │   ├── baseFrameModel.py
│   │   ├── baseline_X.py            # baseline model implementations       
│   │   └── proto_3.py               # default/proposed model
│   └── the_utils/
│       ├── graph_utils.py
│       ├── logger.py
│       └── FocalLoss_and_PCGrad.py
├── Data/
│   ├── DAppSCAN/                  # upstream dataset scripts + adapters/
│   │   ├── the_utils/               # utils
│   │   ├── b1_modules/              # module for auto-compiling projects/file, may not cover all case, but efficient/
│   │   │   └── b_3solc_and_npm.py     # Decide on versions and dependency 
│   │   ├── c2_build_CPG_modules/    # module to build HCPG/
│   │   │   ├── c_1AST.py              # Turn/combine AST json to HAST Graphviz
│   │   │   ├── c_2constants.py        # Contain constant
│   │   │   ├── c_3IR.py               # Analyze SlithIR
│   │   │   ├── c_9CFG.py              # Combine CFG to HCFG
│   │   │   └── c_10Linking.py         # Main linking script to create HCPG
│   │   ├── b1_modules.py            # RUN THIS 1st to auto compile for CFG/AST/SlithIR
│   │   └── e5_preprocess_data.py    # Then RUN THIS 2nd to generate Graphviz HCPG
│   └── ...                          # Other similar dataset, may in different dirve/folder
├── Logs/                        # saved runs
└── save_data_splits/            # pre-split dataset folders (e.g., split_0/)
```

## References

- EtherScan: https://etherscan.io/
  - Collected using [Google Big Query](https://cloud.google.com/bigquery?hl=en)
    ```SQL
    SELECT 
    contracts.address, 
    COUNT(1) AS tx_count 
    FROM `bigquery-public-data.crypto_ethereum.contracts` AS contracts
    JOIN `bigquery-public-data.crypto_ethereum.transactions` AS transactions
      ON transactions.to_address = contracts.address
    GROUP BY contracts.address
    ORDER BY tx_count DESC;
    ```
- DAppSCAN: https://github.com/InPlusLab/DAppSCAN/tree/main

```bibtex
@article{dappscan,
  title={DAppSCAN: Building Large-Scale Datasets for Smart Contract Weaknesses in DApp Projects},
  volume={50},
  doi={10.1109/TSE.2024.3383422},
  number={6},
  journal={IEEE Transactions on Software Engineering},
  publisher={IEEE Computer Society},
  author={Zheng, Zibin and Su, Jianzhong and Chen, Jiachi and Lo, David and Zhong, Zhijie and Ye, Mingxi},
  year={2024},
  month={Mar},
  pages={1360--1373}
}
```

- MANDO-HGT: https://github.com/MANDO-Project/ge-sc-transformer

```bibtex
@inproceedings{nguyen2023msr,
  author = {Nguyen, Hoang H. and Nguyen, Nhat-Minh and Xie, Chunyao and Ahmadi, Zahra and Kudenko, Daniel and Doan, Thanh-Nam and Jiang, Lingxiao},
  title = {MANDO-HGT: Heterogeneous Graph Transformers for Smart Contract Vulnerability Detection},
  year = {2023},
  month = {5},
  booktitle = {Proceedings of the 20th International Conference on Mining Software Repositories},
  numpages = {13},
  keywords = {vulnerability detection, smart contracts, source code, bytecode, heterogeneous graph learning, graph transformer},
  location = {Melbourne, Australia},
  series = {MSR '23}
}
```
