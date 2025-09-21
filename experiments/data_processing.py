"""
Data processing utilities for ChainGuard.
Handles data flow extraction, bytecode preprocessing, and feature conversion.
"""

import logging
from experiments.logger import setup_logger
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from tree_sitter import Language, Parser

from parser import DFG_javascript
from parser import tree_to_token_index, index_to_code_token

logger = setup_logger("data_processing.log")




###########################################################################################
#  █████████                                                                           █████         
# ███░░░░░███                                                                         ░░███          
# ███    ░░░   ██████  █████ ████ ████████   ██████   ██████      ██████   ██████   ███████   ██████ 
# ░█████████  ███░░███░░███ ░███ ░░███░░███ ███░░███ ███░░███    ███░░███ ███░░███ ███░░███  ███░░███
# ░░░░░░░░███░███ ░███ ░███ ░███  ░███ ░░░ ░███ ░░░ ░███████    ░███ ░░░ ░███ ░███░███ ░███ ░███████ 
# ███    ░███░███ ░███ ░███ ░███  ░███     ░███  ███░███░░░     ░███  ███░███ ░███░███ ░███ ░███░░░  
# ░░█████████ ░░██████  ░░████████ █████    ░░██████ ░░██████    ░░██████ ░░██████ ░░████████░░██████ 
# ░░░░░░░░░   ░░░░░░    ░░░░░░░░ ░░░░░      ░░░░░░   ░░░░░░      ░░░░░░   ░░░░░░   ░░░░░░░░  ░░░░░░  
    ###########################################################################################                                                                            


# Define DFG functions mapping
dfg_function = {
    "javascript": DFG_javascript,
}

# Initialize parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language("../parser/my-languages.so", lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


def extract_dataflow(code, parser):
    """
    Extract dataflow from source code.

    Args:
        code: Source code string
        parser: Parser from parsers dictionary
    Returns:
        tuple: (code_tokens, dfg)
    """
    try:
        tree = parser[0].parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split("\n")
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except Exception:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except Exception:
        dfg = []

    return code_tokens, dfg


class InputFeatures(object):
    """A single training/test features of a smart contract/project."""

    def __init__(
        self,
        input_tokens,
        input_ids,
        position_idx,
        dfg_to_code,
        dfg_to_dfg,
        bytecode_embedding,
        label,
        url,
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.bytecode_embedding = bytecode_embedding
        self.label = label
        self.url = url


def convert_examples_to_features(item):
    """
    Convert raw examples to InputFeatures.

    Args:
        item: Tuple containing (url, label, tokenizer, args, cache, url_to_sourcecode, bytecode_embedding)

    Returns:
        InputFeatures: Processed features
    """
    url, label, tokenizer, args, cache, url_to_sourcecode, bytecode_embedding = item
    parser = parsers["javascript"]

    if url not in cache:
        source = url_to_sourcecode[url]
        code_tokens, dfg = extract_dataflow(source, parser, "javascript")

        code_tokens = [
            tokenizer.tokenize("@ " + x)[1:] if idx != 0 else tokenizer.tokenize(x)
            for idx, x in enumerate(code_tokens)
        ]

        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (
                ori2cur_pos[i - 1][1],
                ori2cur_pos[i - 1][1] + len(code_tokens[i]),
            )
        code_tokens = [y for x in code_tokens for y in x]

        code_tokens = code_tokens[: args.code_length + args.data_flow_length - 3][
            : 512 - 3
        ]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [
            i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))
        ]
        dfg = dfg[: args.code_length + args.data_flow_length - len(source_tokens)]
        source_tokens += [x[0] for x in dfg]
        position_idx += [0 for x in dfg]
        source_ids += [tokenizer.unk_token_id for x in dfg]
        padding_length = args.code_length + args.data_flow_length - len(source_ids)
        position_idx += [tokenizer.pad_token_id] * padding_length
        source_ids += [tokenizer.pad_token_id] * padding_length

        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + (
                [reverse_index[i] for i in x[-1] if i in reverse_index],
            )
        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
        length = len([tokenizer.cls_token])
        dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
        cache[url] = source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg

    source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg = cache[url]
    return InputFeatures(
        source_tokens,
        source_ids,
        position_idx,
        dfg_to_code,
        dfg_to_dfg,
        bytecode_embedding,
        label,
        url,
    )


###########################################################################################
#  ███████████              █████                                      █████         
# ░░███░░░░░███            ░░███                                      ░░███          
#  ░███    ░███ █████ ████ ███████    ██████      ██████   ██████   ███████   ██████ 
#  ░██████████ ░░███ ░███ ░░░███░    ███░░███    ███░░███ ███░░███ ███░░███  ███░░███
#  ░███░░░░░███ ░███ ░███   ░███    ░███████    ░███ ░░░ ░███ ░███░███ ░███ ░███████ 
#  ░███    ░███ ░███ ░███   ░███ ███░███░░░     ░███  ███░███ ░███░███ ░███ ░███░░░  
#  ███████████  ░░███████   ░░█████ ░░██████    ░░██████ ░░██████ ░░████████░░██████ 
# ░░░░░░░░░░░    ░░░░░███    ░░░░░   ░░░░░░      ░░░░░░   ░░░░░░   ░░░░░░░░  ░░░░░░  
#                ███ ░███                                                            
#               ░░██████                                                             
#                ░░░░░░                                                              
#VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV                                                                                           

def preprocess_bytecode(bytecode, max_length=512, batch_size=32):
    """
    Preprocess bytecode into embeddings.

    Args:
        bytecode: DataFrame containing bytecode data
        max_length: Maximum sequence length
        batch_size: Batch size for processing
    Returns:
        tuple: (embedding, indices)
    """
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    modelBert = AutoModel.from_pretrained("microsoft/codebert-base")
    embedding = []
    indices = []
    num_sample = len(bytecode)
    num_batches = (num_sample + batch_size - 1) // batch_size
    logger.info(f"Number of batch size {num_batches}")

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_sample)
        batch_bytecodes = bytecode["bytecode"].iloc[start_idx:end_idx].fillna("")
        batch_indices = bytecode["index"].iloc[start_idx:end_idx]

        tokenized_texts = tokenizer(
            batch_bytecodes.tolist(),
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = modelBert(**tokenized_texts)
        batch_embeddings = outputs.last_hidden_state.numpy()

        embedding.append(batch_embeddings)
        indices.append(batch_indices)

    embedding = np.concatenate(embedding, axis=0)
    indices = np.concatenate(indices, axis=0)
    return embedding, indices

###########################################################################################
#     ███████                                       █████         
#   ███░░░░░███                                    ░░███          
#  ███     ░░███ ████████      ██████   ██████   ███████   ██████ 
# ░███      ░███░░███░░███    ███░░███ ███░░███ ███░░███  ███░░███
# ░███      ░███ ░███ ░███   ░███ ░░░ ░███ ░███░███ ░███ ░███████ 
# ░░███     ███  ░███ ░███   ░███  ███░███ ░███░███ ░███ ░███░░░  
#  ░░░███████░   ░███████    ░░██████ ░░██████ ░░████████░░██████ 
#    ░░░░░░░     ░███░░░      ░░░░░░   ░░░░░░   ░░░░░░░░  ░░░░░░  
#                ░███                                             
#                █████                                            
#               ░░░░░                                             
###########################################################################################