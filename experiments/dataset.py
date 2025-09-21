"""
Dataset classes for ChainGuard.
Handles data loading and preprocessing for training, validation, and testing.
"""

import json
import logging
import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from experiments.data_processing import convert_examples_to_features, preprocess_bytecode

logger = logging.getLogger(__name__)


## DEFINE INPUTS


## SOURCE CODE
# Plain text - clean
# Call Graph
# Control Flow Graph
# Dependency Graph
## OP CODE
# 
## BYTE CODE (BIN)
# 


class ModelDataset:
    """Dataset class for loading and processing data."""

    def __init__(self, tokenizer, args, file_path="train"):
        self.examples = []
        self.args = args
        index_filename = file_path

        logger.info("Creating features from index file at %s ", index_filename)
        index_to_sourcecode = {}
        index_to_bytecode = {}

        # Load data from JSONL file
        with open("Data/Mando-test/test-byte-set.jsonl") as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                index_to_sourcecode[js["idx"]] = js["source"]
                index_to_bytecode[js["idx"]] = js["byte"]

        # Convert bytecode to DataFrame
        df_bytecode = pd.DataFrame.from_dict(
            index_to_bytecode, orient="index", columns=["bytecode"]
        )
        df_bytecode.index.name = "index"
        df_bytecode.reset_index(inplace=True)

        # Preprocess bytecode embeddings
        bytecode_embedding, bytecode_index = preprocess_bytecode(
            df_bytecode, max_length=20
        )
        print("Processed bytecode")
        embedding_dict = {
            index: embedding
            for index, embedding in zip(bytecode_index, bytecode_embedding)
        }

        # Load code function according to index
        data_source = []
        cache = {}
        with open(index_filename) as f:
            for line in f:
                line = line.strip()
                url1, labels = line.split(" ", 1)
                labels = [int(label) for label in labels.split()]
                if url1 not in index_to_sourcecode or url1 not in index_to_bytecode:
                    continue
                embedding_code = embedding_dict[url1]
                data_source.append(
                    (
                        url1,
                        labels,
                        tokenizer,
                        args,
                        cache,
                        index_to_sourcecode,
                        embedding_code,
                    )
                )

        # Only use 10% valid data_source to keep best model
        if "valid" in file_path:
            data_source = random.sample(data_source, int(len(data_source) * 0.1))

        # Convert examples to input features
        self.examples = [
            convert_examples_to_features(x)
            for x in tqdm(data_source, total=len(data_source))
        ]

        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        """Get item with attention mask calculation."""
        # Calculate graph-guided masked function
        attn_mask_1 = np.zeros(
            (
                self.args.code_length + self.args.data_flow_length,
                self.args.code_length + self.args.data_flow_length,
            ),
            dtype=bool,
        )

        # Calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])

        # Sequence can attend to sequence
        attn_mask_1[:node_index, :node_index] = True

        # Special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids):
            if i in [0, 2]:
                attn_mask_1[idx, :max_length] = True

        # Nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask_1[idx + node_index, a:b] = True
                attn_mask_1[a:b, idx + node_index] = True

        # Nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask_1[idx + node_index, a + node_index] = True

        out_bytecode_embedding = torch.tensor(self.examples[item].bytecode_embedding)

        return (
            torch.tensor(self.examples[item].input_ids),
            torch.tensor(self.examples[item].position_idx),
            torch.tensor(attn_mask_1),
            out_bytecode_embedding,
            torch.tensor(self.examples[item].label),
        )
