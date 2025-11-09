import os
import re
import sys
import traceback
import ast
import torch
import numpy as np
import networkx as nx
from transformers import AutoTokenizer
import dgl
from typing import List, Dict, Tuple, Optional, Any
import logging
from tqdm import tqdm
import warnings
from collections import defaultdict
try:
    from experiments.the_utils.logger import setup_logger
    from experiments.the_utils.graph_utils import *  ## MOST IMPORTANT AS IT CONTAINS ALL THE VALUES
except ImportError:
    from the_utils.logger import setup_logger
    from the_utils.graph_utils import *  ## MOST IMPORTANT AS IT CONTAINS ALL THE VALUES
import hashlib 
# Configure logging
log_folder = os.getenv('LOG_FOLDER', 'Logs')
os.makedirs(log_folder, exist_ok=True)
logger = setup_logger(f"{log_folder}/GraphProcessing.log")

class GraphFeatureExtractor:
    def __init__(self, tokenizer=None , device=None):
        # Set global random seeds for reproducibility
        
        # region FDimensions
        self.max_modifiers = 10  # maximum number of modifiers to consider
        self.max_structs = 10  # maximum number of structs to consider

        self.vuln_node_cfg_dim =  len(OWASP_VULN) # 8 vulnerability types (no Benign)
        self.vuln_node_cg_dim =  len(OWASP_VULN) # 8 vulnerability types (no Benign)
        self.vuln_node_dfg_dim = len(OWASP_VULN)  # 8 vulnerability types (no Benign)
        # CFG NODE FEATURES DIMENSIONS
        # self.cfg_ir_token_dim = 128
        # self.cfg_code_token_dim = 128
        self.cfg_node_feat_dim =  len(CFG_NODE_TYPE_LIST) + 1 + 1 + self.max_structs + len(STATE_MUTABILITY) + len(FUNC_VISIBILITY) + 3 + len(OPCODE_HIST)  # one-hot + vuln (2) + contract hash + func hash + 7 additional
        
        ###################################################
        #  CG NODE FEATURES DIMENSIONS
        
        self.cg_node_feat_dim = len(CG_NODE_TYPE_LIST) + 1 + len(CONTRACT_KIND) + 1 + self.max_modifiers + self.max_structs + len(STATE_MUTABILITY) + len(FUNC_VISIBILITY) + 3 + len(OPCODE_HIST)  # one-hot + vuln (2) + contract hash + func hash + 7 additional

        ###################################################
        #  AST NODE FEATURES DIMENSIONS
        self.ast_node_feat_dim = len(AST_NODE_TYPE_LIST) + 1 + 1 + 1 + 1 + len(OPCODE_HIST)  + 1  # 1 length value , 2 hash features , no vuln features

        ###################################################
        #  DFG NODE FEATURES DIMENSIONS
        # Enhanced DFG features include:
        # - node_type one-hot (AST_NODE_TYPE_LIST)
        # - contract hash + function hash + var/identifier hash
        # - src length (positional encoding)
        # - type string hash (variable/expression type)
        # - storage location one-hot (VAR_STORAGE)
        # - visibility one-hot (FUNC_VISIBILITY for functions, VAR_VISIBILITY for variables)
        # - mutability flags (constant, stateVariable, isConstant, isPure, isLValue)
        # - state mutability one-hot (STATE_MUTABILITY)
        # - function flags (isConstructor, isFallback, isReceive, virtual, implemented)
        # - operator hash (for binary/unary operations)
        # - memberName hash (for member access)
        # - literal value hash (for literal nodes)
        # - opcode histogram
        self.max_dfg_modifiers = 5  # maximum modifiers for DFG nodes
        self.dfg_node_feat_dim = (
            len(AST_NODE_TYPE_LIST) +  # node_type one-hot
            1 +  # contract hash
            1 +  # function hash  
            1 +  # var/identifier hash
            1 +  # src length
            1 +  # type hash
            len(VAR_STORAGE) +  # storage location one-hot
            len(FUNC_VISIBILITY) +  # visibility one-hot (using FUNC_VISIBILITY as it covers VAR_VISIBILITY)
            5 +  # mutability flags (constant, stateVariable, isConstant, isPure, isLValue)
            len(STATE_MUTABILITY) +  # state mutability one-hot
            5 +  # function flags (isConstructor, isFallback, isReceive, virtual, implemented)
            1 +  # operator hash
            1 +  # memberName hash
            1 +  # literal value hash
            self.max_dfg_modifiers +  # modifiers (hashed, fixed length)
            1 +  # scope hash
            len(OPCODE_HIST)  # opcode histogram
        )
        
        ###################################################
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
        ###################################################

        ###################################################
        # CFG EDGES FEATURES DIMENSIONS 

        self.cfg_edge_feat_dim = len(CFG_EDGE_TYPES) + 1  # one-hot edge type + frequency feature

        ###################################################
        # CG EDGES FEATURES DIMENSIONS

        self.cg_edge_feat_dim = len(CG_EDGE_TYPES) + 1 + len(CALL_OPCODE)  # one-hot + frequency

        ###################################################
        # AST EDGES FEATURES DIMENSIONS

        self.ast_edge_feat_dim = len(AST_EDGE_TYPES)  # one-hot

        ###################################################
        # DFG EDGES FEATURES DIMENSIONS
        # Enhanced DFG edge features include:
        # - edge_type one-hot (DFG_EDGE_TYPES: 9 types)
        # - flow label one-hot (DFG_FLOW_LABELS: 13 types)
        # - frequency (1 float)
        # - src_var hash (1 float)
        # - dst_var hash (1 float)
        # - operator hash (1 float) - for assignment edges
        # - contract hash (1 float)
        # - function hash (1 float)
        # - src_offset (1 float) - source location offset
        # - src_length (1 float) - source location length
        # - is_external_call (1 boolean) - for call-arg edges
        self.dfg_edge_feat_dim = (
            len(DFG_EDGE_TYPES) +  # edge_type one-hot
            len(DFG_FLOW_LABELS) +  # flow label one-hot
            1 +  # frequency
            1 +  # src_var hash
            1 +  # dst_var hash
            1 +  # operator hash
            1 +  # contract hash
            1 +  # function hash
            1    # is_external_call
        )


        ###################################################
        self.tokenizer = tokenizer
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.int_dtype = torch.int32 
        self.mode = "train"
        # endregion

############################################################
############################################################
    # region UTILS
    def fetch_all_feat_dim(self):
        return {
            # Graph input dimensions (node features)
            "GRAPH_IN_DIMS": {
                "ast_node": self.ast_node_feat_dim + 1, # 1 node ID hash feature
                "block": self.cfg_node_feat_dim + 1,     # 1 node ID hash feature
                "function": self.cg_node_feat_dim + 1,   # 1 node ID hash feature
                "dfg_node": self.dfg_node_feat_dim + 1    # 1 node ID hash feature
            },
            
            # Additional dimensions for reference
            "EDGE_DIMS": {
                "cfg_edge": self.cfg_edge_feat_dim,
                "cg_edge": self.cg_edge_feat_dim,
                "ast_edge": self.ast_edge_feat_dim,
                "dfg_edge": self.dfg_edge_feat_dim
            },
            
            "LABEL": {
                "cfg": self.vuln_node_cfg_dim,
                "cg": self.vuln_node_cg_dim,
                "dfg": self.vuln_node_dfg_dim
            },
        }
    @staticmethod
    def _hash_md5_f32(text: str) -> np.float32:
        if not text or text.strip() == "":
            return np.float32(0.0)
        h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
        val = (h & 0xFFFFFFFF) / 0xFFFFFFFF  # normalize to [0,1]
        return np.float32(val)

    @staticmethod
    def _safe_literal_eval(value: Any, default: Any = None) -> Any:
        """
        Safely evaluate a string representation of a Python literal.
        Handles JSON-style booleans (lowercase true/false), nulls, and Python literals.
        
        Args:
            value: The value to evaluate (can be already evaluated or string)
            default: Default value if evaluation fails (defaults to empty list or dict based on str representation)
        
        Returns:
            The evaluated value or default if evaluation fails
        """
        # If already evaluated (list, dict, bool, etc.), return as-is
        if isinstance(value, (list, dict, bool, int, float)):
            return value
        
        # If None or empty, return default
        if value is None or (isinstance(value, str) and not value.strip()):
            if default is None:
                # Infer default from context - if looks like list, return []
                return [] if default is None else default
            return default
        
        # Try to evaluate string representations
        if isinstance(value, str):
            # Normalize JSON-style values to Python style
            normalized_value = value.strip()
            
            # Handle JSON booleans (lowercase)
            if normalized_value.lower() == 'true':
                return True
            elif normalized_value.lower() == 'false':
                return False
            elif normalized_value.lower() == 'null':
                return None
            
            # Replace JSON null with Python None in complex structures like [null], {null: ...}
            # This allows ast.literal_eval to parse them correctly
            normalized_value = normalized_value.replace('null', 'None')
            
            # Try ast.literal_eval for lists, dicts, and other Python literals
            try:
                return ast.literal_eval(normalized_value)
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Failed to literal_eval '{value}': {e}. Using default: {default}")
                if default is None:
                    # Try to infer based on content
                    if normalized_value.startswith('['):
                        return []
                    elif normalized_value.startswith('{'):
                        return {}
                    else:
                        return normalized_value  # Return as string if can't parse
                return default
        
        return value if default is None else default
############################################################
############################################################

    # region NODES

    def _cfg_node_features(self, node_data):
        true_length = []
        features = []
        # NODE {
        #     "label": label,
        #     "node_type": node_type,
        #      "node_vulns": node_vulns, # node_vulns=['SC06:2025']
        #     "raw_ir": raw_ir, raw_ir="name(string) := tokenName(string)"
        #     "raw_code": raw_code,  raw_code="name = tokenName"
        #     "contract_name": contract_name, contract_name=TokenERC20
        #     "func_name": func_name, func_name="TokenERC20(uint256,string,string)"
        #     "structs": structs,
        #     "enums": enums,
        #     "errors": errors,
        #     "type_aliases": type_aliases 
        # }
        # One-hot encoding for node_type
        node_type = node_data.get("node_type", "")
        one_hot = node_type_to_one_hot(node_type, CFG_NODE_TYPE_LIST)
        features.extend(one_hot)
        true_length.append(len(one_hot))


        # 5. contract_name: hash encoding
        contract_name = node_data.get("contract_name", "")
        contract_hash = self._hash_md5_f32(contract_name) 
        features.append(contract_hash)
        true_length.append(1)

        # 6. func_name: hash encoding
        func_name = node_data.get("func_name", "")
        func_hash = self._hash_md5_f32(func_name)
        features.append(func_hash)
        true_length.append(1)

        structs = node_data.get("structs", [])
        structs = self._safe_literal_eval(structs, default=[])
        for m in structs[:self.max_structs]:
            features.append(self._hash_md5_f32(m))
        if len(structs) < self.max_structs:
            for _ in range(self.max_structs - len(structs)):
                features.append(0.0)  # padding
        true_length.append(self.max_structs)

        state_multability = node_data.get("stateMutability","none")
        sm_oh = func_stateMutability_to_one_hot(state_multability)
        features.extend(sm_oh)
        true_length.append(len(sm_oh))

        # visibility one-hot
        func_visibility = node_data.get("visibility", "internal")
        vis_oh = func_visibility_to_one_hot(func_visibility)
        features.extend(vis_oh)
        true_length.append(len(vis_oh))

        isConstructor = node_data.get("isConstructor", False)
        isConstructor = self._safe_literal_eval(isConstructor, default=False)
        features.append(float(isConstructor))
        true_length.append(1)

        isFallback = node_data.get("isFallback", False)
        isFallback = self._safe_literal_eval(isFallback, default=False)
        features.append(float(isFallback))
        true_length.append(1)

        isReceive = node_data.get("isReceive", False)
        isReceive = self._safe_literal_eval(isReceive, default=False)
        features.append(float(isReceive))
        true_length.append(1)

        # opcode histogram
        opcode_hist = node_data.get("opcode_hist", {})
        opcode_hist = self._safe_literal_eval(opcode_hist, default={})
        op_hist_feat = op_code_histogram_to_feat(opcode_hist)
        features.extend(op_hist_feat)
        true_length.append(len(op_hist_feat))


        if not sum(true_length) == self.cfg_node_feat_dim:
            logger.warning(f"CFG Node feature dim mismatch: expected {sum(true_length)}, got {self.cfg_node_feat_dim}")
        
        # Vuln node marking
        vuln_node = None
        node_vulns = node_data.get("node_vulns", [])
        node_vulns = self._safe_literal_eval(node_vulns, default=[])
        # print(f"Node vulns: {node_vulns}, Type: {type(node_vulns)}")
        if node_vulns:
            vuln_vector = vuln_to_label(node_vulns)
            # print(f"Vuln vector: {vuln_vector}")
            # vuln_node = [contract_hash, func_hash]
            # vuln_node.extend(one_hot)
            # vuln_node.extend(vuln_vector)
            vuln_node = vuln_vector
        
        
        return features, vuln_node
        
    def _cg_node_features(self, node_data):
        # NODE = {
        #     "node_type": node_type,
        #     "node_vulns": node_vulns,
        #     "contract_kind": str(contract_meta.get("contract_kind",None)),
        #   
        #     "structs": list((contract_meta.get("structs") or {}).keys()),
        #     "enums": list((contract_meta.get("enums") or {}).keys()),
        #     "errors": list((contract_meta.get("errors") or {}).keys()),
        #     "type_aliases": list((contract_meta.get("type_aliases") or {}).keys()),
        #    
        #     "inputs": func_meta.get("inputs", None),
        #     "outputs": func_meta.get("outputs", None),

        #     "modifiers": func_meta.get("modifiers", None),
        #     "is_implemented": func_meta.get("is_implemented", False),
        #     "stateMutability": str(func_meta.get("stateMutability", None)),
        #     "visibility": str(func_meta.get("visibility","internal")),
        #     "isConstructor": func_meta.get("isConstructor", False),
        #     "isFallback": func_meta.get("isFallback", False),
        #     "isReceive": func_meta.get("isReceive", False),

        #     "opcode_hist": func_meta.get("opcode_hist", None),
        # }

        true_length = []
        features = []
        # CG node features:
        node_type = node_data.get("node_type", "")
        one_hot = node_type_to_one_hot(node_type, CG_NODE_TYPE_LIST)
        features.extend(one_hot)
        true_length.append(len(one_hot))

        # contract_name hash
        contract_name = node_data.get("contract_name", "")
        contract_hash = self._hash_md5_f32(contract_name)
        features.append(contract_hash)
        true_length.append(1)

        # contract_kind one-hot
        contract_kind = node_data.get("contract_kind", "none")
        ck_oh = contract_kind_to_one_hot(contract_kind)
        features.extend(ck_oh)
        true_length.append(len(ck_oh))

        # func_name hash
        func_name = node_data.get("func_name", "")
        func_hash = self._hash_md5_f32(func_name)
        features.append(func_hash)
        true_length.append(1)

        # structs (fixed length via padding)
        structs = node_data.get("structs", [])
        structs = self._safe_literal_eval(structs, default=[])
        for m in structs[:self.max_structs]:
            features.append(self._hash_md5_f32(m))
        if len(structs) < self.max_structs:
            for _ in range(self.max_structs - len(structs)):
                features.append(0.0)
        true_length.append(self.max_structs)

        # inputs = node_data.get("inputs", [])
        # features.append(float(len(inputs)))

        # outputs = node_data.get("outputs", [])
        # features.append(float(len(outputs)))

        state_multability = node_data.get("stateMutability","none")
        sm_oh = func_stateMutability_to_one_hot(state_multability)
        features.extend(sm_oh)
        true_length.append(len(sm_oh))

        # visibility one-hot
        func_visibility = node_data.get("visibility", "internal")
        vis_oh = func_visibility_to_one_hot(func_visibility)
        features.extend(vis_oh)
        true_length.append(len(vis_oh))

        # modifiers (fixed length via padding)
        modifiers = node_data.get("modifiers", [])
        modifiers = self._safe_literal_eval(modifiers, default=[])
        modifier_hashes = [self._hash_md5_f32(m) for m in modifiers[:self.max_modifiers]]
        if len(modifier_hashes) < self.max_modifiers:
            modifier_hashes += [0.0] * (self.max_modifiers - len(modifier_hashes))
        features.extend(modifier_hashes)
        true_length.append(self.max_modifiers)


        isConstructor = node_data.get("isConstructor", False)
        isConstructor = self._safe_literal_eval(isConstructor, default=False)
        features.append(float(isConstructor))
        true_length.append(1)

        isFallback = node_data.get("isFallback", False)
        isFallback = self._safe_literal_eval(isFallback, default=False)
        features.append(float(isFallback))
        true_length.append(1)

        isReceive = node_data.get("isReceive", False)
        isReceive = self._safe_literal_eval(isReceive, default=False)
        features.append(float(isReceive))
        true_length.append(1)

        # opcode histogram
        opcode_hist = node_data.get("opcode_hist", {})
        opcode_hist = self._safe_literal_eval(opcode_hist, default={})
        op_hist_feat = op_code_histogram_to_feat(opcode_hist)
        features.extend(op_hist_feat)
        true_length.append(len(op_hist_feat))

        if not sum(true_length) == self.cg_node_feat_dim:
            logger.warning(f"CG Node feature dim mismatch: expected {sum(true_length)}, got {self.cg_node_feat_dim}")
        
        # Vulnerability labels
        vuln_node = None
        node_vulns = node_data.get("node_vulns", [])
        node_vulns = self._safe_literal_eval(node_vulns, default=[])
        # print(f"Node vulns: {node_vulns}, Type: {type(node_vulns)}")
        if node_vulns:
            vuln_vector = vuln_to_label(node_vulns)
            #print(f"Vuln vector: {vuln_vector}")
            # vuln_node = [contract_hash, func_hash]
            # vuln_node.extend(one_hot)
            # vuln_node.extend(vuln_vector)
            vuln_node = vuln_vector
        
        return features, vuln_node
        
    def _ast_node_features(self, node_data):
    #     NODE =  {
    #     "src": src,  # if present
    #     "contract_name": contract_name,  # from project_json
    #     "contract_kind": contract_kind,  # if available
    #     "function_name": function_name,  # if matched
    #     "opcode_hist": opcode_hist,  # if available
    #     "raw_code": raw_code,  # if available
    #     "variable_name": variable_name,  # if matched
    #     "event_name": event_name,  # if matched
    # }

        true_length = []
        features = []
        # AST node features
        node_type = node_data.get("node_type", "")
        one_hot = node_type_to_one_hot(node_type, AST_NODE_TYPE_LIST)
        features.extend(one_hot)
        true_length.append(len(one_hot))

        # contract_name hash
        contract_name = node_data.get("contract_name", "")
        contract_hash = self._hash_md5_f32(contract_name)
        features.append(contract_hash)
        true_length.append(1)

        # function_name hash
        function_name = node_data.get("function_name", "")
        func_hash = self._hash_md5_f32(function_name)
        features.append(func_hash)
        true_length.append(1)

        # variable_name hash
        variable_name = node_data.get("variable_name", "")
        var_hash = self._hash_md5_f32(variable_name)
        features.append(var_hash)
        true_length.append(1)

        # event_name hash
        event_name = node_data.get("event_name", "")
        event_hash = self._hash_md5_f32(event_name)
        features.append(event_hash)
        true_length.append(1)

        # opcode histogram
        opcode_hist = node_data.get("opcode_hist", {})
        opcode_hist = self._safe_literal_eval(opcode_hist, default={})
        ast_hist_feat = op_code_histogram_to_feat(opcode_hist)
        features.extend(ast_hist_feat)
        true_length.append(len(ast_hist_feat))

        # contract_kind hash
        contract_kind = node_data.get("contract_kind", "")
        contract_kind_hash = self._hash_md5_f32(contract_kind)
        features.append(contract_kind_hash)
        true_length.append(1)

        if not sum(true_length) == self.ast_node_feat_dim:
            logger.warning(f"AST Node feature dim mismatch: expected {sum(true_length)}, got {self.ast_node_feat_dim}")
        return features
        
    def _dfg_node_features(self, node_data):
        # DFG NODE DATA STRUCTURE (from combine_dfgs_to_pydot in e_4generate_enrichDFG.py):
        # Enhanced structure with enrichment from project_json and normalization:
        # {
        #     "node_type": normalized_category,          # e.g., "Return", "ElementaryType", "Identifier"
        #     "raw_node_type": original_type,            # e.g., "Return", "uint256"
        #     "name": identifier_name,                   # Variable/function/event name
        #     "contract_name": contract_name,            # Contract containing this node
        #     "function_name": function_name,            # Function containing this node
        #     "src": "offset:length:fileId",             # Source location
        #     
        #     # Type and storage
        #     "type": type_string,                       # Variable/expression type
        #     "storageLocation": storage,                # memory/storage/calldata
        #     "visibility": visibility,                  # private/public/internal/external
        #     
        #     # Mutability flags
        #     "constant": bool,                          # Is constant variable
        #     "stateVariable": bool,                     # Is state variable
        #     "isConstant": bool,                        # Is constant expression
        #     "isPure": bool,                            # Is pure expression
        #     "isLValue": bool,                          # Is L-value
        #     "mutability": str,                         # Variable mutability
        #     
        #     # Operators and access
        #     "operator": operator,                      # For BinaryOp/UnaryOp
        #     "memberName": member_name,                 # For member access
        #     
        #     # Literal values
        #     "value": literal_value,                    # For literals
        #     "hexValue": hex_value,                     # For literals
        #     
        #     # Function attributes (for function nodes)
        #     "func_visibility": visibility,             # Function visibility
        #     "func_state_mutability": state_mut,        # pure/view/nonpayable/payable
        #     "func_is_constructor": bool,
        #     "func_is_fallback": bool,
        #     "func_is_receive": bool,
        #     "func_virtual": bool,
        #     "func_implemented": bool,
        #     "func_modifiers": [modifier_list],         # Function modifiers
        #     
        #     # Scope and references
        #     "scope": scope_id,                         # Scope reference
        #     "referencedDeclaration": decl_id,          # Referenced declaration
        #     
        #     # Opcode analysis
        #     "opcode_hist": {opcode: count},            # Opcode histogram
        #     
        #     # Vulnerabilities
        #     "node_vulns": vuln_list                    # List of vulnerabilities like ["SC01:2025"]
        # }
        true_length = []
        features = []
        vuln_node = None
        
        # 1. Node type - one-hot encoding using AST_NODE_TYPE_LIST
        # DFG uses normalized AST node types, so we reuse AST_NODE_TYPE_LIST
        node_type = node_data.get("node_type", "")
        one_hot = node_type_to_one_hot(node_type, AST_NODE_TYPE_LIST)
        features.extend(one_hot)
        true_length.append(len(one_hot))
        
        # 2. Contract name hash
        contract_name = node_data.get("contract_name", "")
        contract_hash = self._hash_md5_f32(contract_name)
        features.append(contract_hash)
        true_length.append(1)
        
        # 3. Function name hash
        function_name = node_data.get("function_name", "")
        func_hash = self._hash_md5_f32(function_name)
        features.append(func_hash)
        true_length.append(1)
        
        # 4. Variable/identifier/node name hash
        # Priority: "name" field > "var" field > "node_name" field
        var_name = node_data.get("name", "") or node_data.get("var", "") or node_data.get("node_name", "")
        var_hash = self._hash_md5_f32(var_name)
        features.append(var_hash)
        true_length.append(1)
        
        # 5. Source location length (for positional encoding)
        src = node_data.get("src", "")
        src_length = 0.0
        if src and isinstance(src, str):
            parts = src.split(":")
            if len(parts) >= 2:
                try:
                    src_length = float(parts[1])  # Extract length from "offset:length:fileId"
                except:
                    pass
        features.append(src_length)
        true_length.append(1)
        
        # 6. Type string hash (variable/expression type)
        type_str = node_data.get("type", "")
        type_hash = self._hash_md5_f32(type_str)
        features.append(type_hash)
        true_length.append(1)
        
        # 7. Storage location one-hot (memory/storage/calldata)
        storage_location = node_data.get("storageLocation", "none")
        storage_oh = var_storage_to_one_hot(storage_location)
        features.extend(storage_oh)
        true_length.append(len(storage_oh))
        
        # 8. Visibility one-hot
        # For variables, use visibility; for functions, use func_visibility
        visibility = node_data.get("visibility", "") or node_data.get("func_visibility", "internal")
        vis_oh = func_visibility_to_one_hot(visibility)
        features.extend(vis_oh)
        true_length.append(len(vis_oh))
        
        # 9. Mutability flags (5 binary features)
        constant = node_data.get("constant", False)
        constant = self._safe_literal_eval(constant, default=False)
        features.append(float(constant))
        
        state_variable = node_data.get("stateVariable", False)
        state_variable = self._safe_literal_eval(state_variable, default=False)
        features.append(float(state_variable))
        
        is_constant = node_data.get("isConstant", False)
        is_constant = self._safe_literal_eval(is_constant, default=False)
        features.append(float(is_constant))
        
        is_pure = node_data.get("isPure", False)
        is_pure = self._safe_literal_eval(is_pure, default=False)
        features.append(float(is_pure))
        
        is_lvalue = node_data.get("isLValue", False)
        is_lvalue = self._safe_literal_eval(is_lvalue, default=False)
        features.append(float(is_lvalue))
        
        true_length.append(5)
        
        # 10. State mutability one-hot (for functions)
        state_mutability = node_data.get("func_state_mutability", "") or node_data.get("stateMutability", "none")
        sm_oh = func_stateMutability_to_one_hot(state_mutability)
        features.extend(sm_oh)
        true_length.append(len(sm_oh))
        
        # 11. Function flags (5 binary features)
        is_constructor = node_data.get("func_is_constructor", False)
        is_constructor = self._safe_literal_eval(is_constructor, default=False)
        features.append(float(is_constructor))
        
        is_fallback = node_data.get("func_is_fallback", False)
        is_fallback = self._safe_literal_eval(is_fallback, default=False)
        features.append(float(is_fallback))
        
        is_receive = node_data.get("func_is_receive", False)
        is_receive = self._safe_literal_eval(is_receive, default=False)
        features.append(float(is_receive))
        
        is_virtual = node_data.get("func_virtual", False)
        is_virtual = self._safe_literal_eval(is_virtual, default=False)
        features.append(float(is_virtual))
        
        is_implemented = node_data.get("func_implemented", True)
        is_implemented = self._safe_literal_eval(is_implemented, default=True)
        features.append(float(is_implemented))
        
        true_length.append(5)
        
        # 12. Operator hash (for binary/unary operations)
        operator = node_data.get("operator", "")
        operator_hash = self._hash_md5_f32(operator)
        features.append(operator_hash)
        true_length.append(1)
        
        # 13. Member name hash (for member access)
        member_name = node_data.get("memberName", "")
        member_hash = self._hash_md5_f32(member_name)
        features.append(member_hash)
        true_length.append(1)
        
        # 14. Literal value hash (for literal nodes)
        # Use hexValue if available, otherwise use value
        literal_value = node_data.get("hexValue", "") or node_data.get("value", "")
        literal_hash = self._hash_md5_f32(str(literal_value))
        features.append(literal_hash)
        true_length.append(1)
        
        # 15. Function modifiers (fixed length via padding)
        modifiers = node_data.get("func_modifiers", [])
        modifiers = self._safe_literal_eval(modifiers, default=[])
        modifier_hashes = []
        for m in modifiers[:self.max_dfg_modifiers]:
            # Extract modifier name if it's a dict structure
            if isinstance(m, dict):
                mod_name = m.get("name", "") or m.get("modifierName", {}).get("name", "")
            else:
                mod_name = str(m)
            modifier_hashes.append(self._hash_md5_f32(mod_name))
        
        if len(modifier_hashes) < self.max_dfg_modifiers:
            modifier_hashes.extend([0.0] * (self.max_dfg_modifiers - len(modifier_hashes)))
        features.extend(modifier_hashes)
        true_length.append(self.max_dfg_modifiers)
        
        # 16. Scope hash
        scope = node_data.get("scope", "")
        scope_hash = self._hash_md5_f32(str(scope))
        features.append(scope_hash)
        true_length.append(1)
        
        # 17. Opcode histogram
        opcode_hist = node_data.get("opcode_hist", {})
        opcode_hist = self._safe_literal_eval(opcode_hist, default={})
        op_hist_feat = op_code_histogram_to_feat(opcode_hist)
        features.extend(op_hist_feat)
        true_length.append(len(op_hist_feat))
        
        if not sum(true_length) == self.dfg_node_feat_dim:
            logger.warning(f"DFG Node feature dim mismatch: expected {self.dfg_node_feat_dim}, got {sum(true_length)}")
            logger.warning(f"Feature breakdown: {list(zip(['node_type', 'contract', 'function', 'var', 'src_len', 'type', 'storage', 'visibility', 'mutability_flags', 'state_mut', 'func_flags', 'operator', 'member', 'literal', 'modifiers', 'scope', 'opcode_hist'], true_length))}")
        
        # Check for vulnerability labels
        node_vulns = node_data.get("node_vulns", [])
        node_vulns = self._safe_literal_eval(node_vulns, default=[])
        if node_vulns:
            vuln_vector = vuln_to_label(node_vulns)
            vuln_node = vuln_vector
        
        return features, vuln_node
        
    def _creation_cfg_node_features(self, node_data):
        # Similar to CFG but for creation bytecode
        return self._cfg_node_features(node_data)
        
    def _runtime_cfg_node_features(self, node_data):
        # Similar to CFG but for runtime bytecode
        return self._cfg_node_features(node_data)
        
    def extract_node_features(self, nx_graph: nx.DiGraph, graph_type: str):
        """
        Extract node features from NetworkX graph.
        
        Args:
            nx_graph (nx.DiGraph): NetworkX directed graph
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping node IDs to feature tensors
        """
        node_features = {}
        vuln_nodes = {}
        
        for node_id, node_data in nx_graph.nodes(data=True): # Loop through nodes
            features = []
            features.append(self._hash_md5_f32(str(node_id)))  # Basic hash feature for node ID
            features_tmp = []
            
            if graph_type == "cfg":
                features_tmp, vuln_node = self._cfg_node_features(node_data)
                if vuln_node is not None and len(vuln_node) > 0:
                    #print(f"Vulnerable node found in CFG: Node ID {type(node_id)}{node_id} with vuln data {vuln_node}")
                    vuln_nodes[node_id] = vuln_node 
            elif graph_type == "cg":
                features_tmp, vuln_node = self._cg_node_features(node_data)
                if vuln_node is not None and len(vuln_node) > 0:
                    #print(f"Vulnerable node found in CG: Node ID {type(node_id)}{node_id} with vuln data {vuln_node}")
                    vuln_nodes[node_id] = vuln_node 
            elif graph_type == "ast":
                features_tmp = self._ast_node_features(node_data)
            elif graph_type == "dfg":
                features_tmp, vuln_node = self._dfg_node_features(node_data)
                if vuln_node is not None and len(vuln_node) > 0:
                    #print(f"Vulnerable node found in DFG: Node ID {type(node_id)}{node_id} with vuln data {vuln_node}")
                    vuln_nodes[node_id] = vuln_node
            elif graph_type == "creation_cfg":
                features_tmp = self._creation_cfg_node_features(node_data)
            elif graph_type == "runtime_cfg":
                features_tmp = self._runtime_cfg_node_features(node_data)
            else:
                # Placeholder for other graph types
                if graph_type == "cfg":
                    features_tmp = [0.0] * self.cfg_node_feat_dim
                elif graph_type == "cg":
                    features_tmp = [0.0] * self.cg_node_feat_dim
                elif graph_type == "ast":
                    features_tmp = [0.0] * self.ast_node_feat_dim
                elif graph_type == "dfg":
                    features_tmp = [0.0] * self.dfg_node_feat_dim
                else:
                    features_tmp = [0.0] * self.cfg_node_feat_dim
            features.extend(features_tmp)
            node_key = (graph_type, str(node_id))
            #print(f"Node Key: {node_key}, Features Length: {features}")
            node_features[node_key] = torch.tensor(features, dtype=self.dtype)
        
        return node_features , vuln_nodes
############################################################
############################################################

    # region EGDES
    def _cfg_edge_features(self, edge_data):
        true_length = []
        features = []
        
        edge_type = edge_data.get("edge_type", "")
        one_hot = edge_type_to_one_hot(edge_type, CFG_EDGE_TYPES)
        features.extend(one_hot)
        true_length.append(len(one_hot))

        features.append(float(edge_data.get("frequency", 0.0)))
        true_length.append(1)

        #     features.append(connects_vuln)
        if not sum(true_length) == self.cfg_edge_feat_dim:
            logger.warning(f"CFG Edge feature dim mismatch: expected {sum(true_length)}, got {self.cfg_edge_feat_dim}")
        return features
        
    def _cg_edge_features(self, edge_data):
        # EDGES = {
        #     "edge_type": edge_type,
        #     "frequency": frequency,
        #     "opcodes": [],
        #     "ir"= ["str"]
        # }
        true_length = []
        features = []
    
        edge_type = edge_data.get("edge_type", "")
        one_hot = edge_type_to_one_hot(edge_type, CG_EDGE_TYPES)
        features.extend(one_hot)
        true_length.append(len(one_hot))

        features.append(float(edge_data.get("frequency", 0.0)))
        true_length.append(1)
        
        call_ops = edge_data.get("opcodes", [])
        call_ops = self._safe_literal_eval(call_ops, default=[])
        call_feat = call_opcode_to_feat(call_ops)
        features.extend(call_feat)
        true_length.append(len(call_feat))

        if not sum(true_length) == self.cg_edge_feat_dim:
            logger.warning(f"CG Edge feature dim mismatch: expected {sum(true_length)}, got {self.cg_edge_feat_dim}")
        return features
    
    def _ast_edge_features(self, edge_data):
        true_length = []
        features = []
        
        edge_type = edge_data.get("edge_type", "")
        one_hot = edge_type_to_one_hot(edge_type, AST_EDGE_TYPES)
        features.extend(one_hot)
        true_length.append(len(one_hot))
        # No vuln for AST
        # features.append(0.0)  # but to match dim, perhaps not needed since dim is different
        
        if not sum(true_length) == self.ast_edge_feat_dim:
            logger.warning(f"AST Edge feature dim mismatch: expected {sum(true_length)}, got {self.ast_edge_feat_dim}")
        return features
    
    def _dfg_edge_features(self, edge_data):
        # DFG EDGE DATA STRUCTURE (from e_4generate_enrichDFG.py):
        # Enhanced structure with detailed flow tracking:
        # {
        #     "src_id": src_id,
        #     "dst_id": dst_id,
        #     "edge_type": edge_type,  # e.g., "init", "assign", "var-read", "var-write", "call-arg", "return", "member", "index-base", "index-idx"
        #     "flow": flow_key,        # Machine-readable flow type: "init_to_var", "rhs_to_lhs", "decl_to_use", etc.
        #     "flow_label": flow_label, # Human-readable flow description
        #     "src_var": src_var,      # Source variable name
        #     "dst_var": dst_var,      # Destination variable name
        #     "contract_name": contract_name,
        #     "function_name": function_name,
        #     "frequency": frequency,  # Edge frequency count
        #     "src_offset": src_offset,  # Source location offset
        #     "src_length": src_length,  # Source location length
        #     "src_file": src_file,      # Source file index
        #     "operator": operator,    # For binary operations (assign edges) - e.g., "=", "+="
        #     "is_external_call": is_external_call  # Boolean flag (only on call-arg edges)
        # }
        true_length = []
        features = []
        
        # 1. Edge type one-hot encoding (from DFG_EDGE_TYPES: init, assign, var-write, var-read, call-arg, return, member, index-base, index-idx)
        edge_type = edge_data.get("edge_type", "")
        edge_type_oh = edge_type_to_one_hot(edge_type, DFG_EDGE_TYPES)
        features.extend(edge_type_oh)
        true_length.append(len(edge_type_oh))
        
        # 2. Flow label one-hot encoding (from DFG_FLOW_LABELS: 13 types)
        # This captures the semantic relationship: init_to_var, rhs_to_lhs, decl_to_use, etc.
        flow = edge_data.get("flow", "")
        flow_oh = edge_type_to_one_hot(flow, DFG_FLOW_LABELS)
        features.extend(flow_oh)
        true_length.append(len(flow_oh))

        # 3. Frequency feature (how many times this edge appears)
        features.append(float(edge_data.get("frequency", 1.0)))
        true_length.append(1)
        
        # 4. Source variable hash
        src_var = edge_data.get("src_var", "")
        src_var_hash = self._hash_md5_f32(str(src_var) if src_var is not None else "")
        features.append(src_var_hash)
        true_length.append(1)
        
        # 5. Destination variable hash
        dst_var = edge_data.get("dst_var", "")
        dst_var_hash = self._hash_md5_f32(str(dst_var) if dst_var is not None else "")
        features.append(dst_var_hash)
        true_length.append(1)
        
        # 6. Operator hash (for assignment edges)
        # Captures operators like "=", "+=", "-=", "*=", "/="
        operator = edge_data.get("operator", "")
        operator_hash = self._hash_md5_f32(str(operator) if operator else "")
        features.append(operator_hash)
        true_length.append(1)
        
        # 7. Contract name hash
        contract_name = edge_data.get("contract_name", "") or edge_data.get("contract", "")
        contract_hash = self._hash_md5_f32(str(contract_name) if contract_name else "")
        features.append(contract_hash)
        true_length.append(1)
        
        # 8. Function name hash
        function_name = edge_data.get("function_name", "") or edge_data.get("function", "")
        function_hash = self._hash_md5_f32(str(function_name) if function_name else "")
        features.append(function_hash)
        true_length.append(1)
        
        # 11. Is external call flag (for call-arg edges)
        is_external_call = edge_data.get("is_external_call", False)
        is_external_call = self._safe_literal_eval(is_external_call, default=False)
        features.append(float(is_external_call))
        true_length.append(1)
        
        if not sum(true_length) == self.dfg_edge_feat_dim:
            logger.warning(f"DFG Edge feature dim mismatch: expected {self.dfg_edge_feat_dim}, got {sum(true_length)}")
            logger.warning(f"Feature breakdown: {list(zip(['edge_type', 'flow', 'frequency', 'src_var', 'dst_var', 'operator', 'contract', 'function', 'src_offset', 'src_length', 'is_external_call'], true_length))}")
        return features
    
    def _creation_cfg_edge_features(self, edge_data):
        return self._cfg_edge_features(edge_data)
    
    def _runtime_cfg_edge_features(self, edge_data):
        return self._cfg_edge_features(edge_data)
    
    def extract_edge_features(self, nx_graph: nx.DiGraph, graph_type: str, vuln_nodes) -> Dict[Tuple[str, str], torch.Tensor]:
        edge_features = {}

        for src, dst, edge_data in nx_graph.edges(data=True):
            src = str(src)
            dst = str(dst)
            features = []
            features.extend([self._hash_md5_f32(src),self._hash_md5_f32(dst)])
            if graph_type == "cfg":
                features = self._cfg_edge_features(edge_data)
                if vuln_nodes is not None and len(vuln_nodes) > 0:
                    x = vuln_nodes.get(src)
                    y = vuln_nodes.get(dst)
                    if x or y:
                        src_hash = self._hash_md5_f32(src)
                        dst_hash = self._hash_md5_f32(dst)
                        if x:
                            vuln_nodes[src].extend([src_hash, dst_hash if y else 0.0])
                        if y:
                            vuln_nodes[dst].extend([dst_hash, src_hash if x else 0.0])
                        # print(f"[CFG] Edge connecting vuln nodes: {src} -> {dst}")
                        
            elif graph_type == "cg":
                features = self._cg_edge_features(edge_data)
                if vuln_nodes is not None and len(vuln_nodes) > 0:
                    x = vuln_nodes.get(src)
                    y = vuln_nodes.get(dst)
                    if x or y:
                        src_hash = self._hash_md5_f32(src)
                        dst_hash = self._hash_md5_f32(dst)
                        if x:
                            vuln_nodes[src].extend([src_hash, dst_hash if y else 0.0])
                        if y:
                            vuln_nodes[dst].extend([dst_hash, src_hash if x else 0.0])
                        # print(f"[CG] Edge connecting vuln nodes: {src} -> {dst}")

            elif graph_type == "ast":
                features = self._ast_edge_features(edge_data)
            elif graph_type == "dfg":
                features = self._dfg_edge_features(edge_data)
                if vuln_nodes is not None and len(vuln_nodes) > 0:
                    x = vuln_nodes.get(src)
                    y = vuln_nodes.get(dst)
                    if x or y:
                        src_hash = self._hash_md5_f32(src)
                        dst_hash = self._hash_md5_f32(dst)
                        if x:
                            vuln_nodes[src].extend([src_hash, dst_hash if y else 0.0])
                        if y:
                            vuln_nodes[dst].extend([dst_hash, src_hash if x else 0.0])
                        # print(f"[DFG] Edge connecting vuln nodes: {src} -> {dst}")
            elif graph_type == "creation_cfg":
                features = self._creation_cfg_edge_features(edge_data)
            elif graph_type == "runtime_cfg":
                features = self._runtime_cfg_edge_features(edge_data)
            else:
                # Default edge features
                features = [0.0] * self.cfg_edge_feat_dim
            
            # Convert to tensor
            edge_features[(graph_type, src, dst)] = torch.tensor(features, dtype=self.dtype)
        
        return edge_features, vuln_nodes
    

    ###############################################################
    # region FUSE to CPG 
    ###############################################################
    def create_hetero_graph(self, project_name: str,
                            nx_cg: nx.DiGraph, 
                            nx_cfg: nx.DiGraph, 
                            nx_ast: nx.DiGraph,
                            nx_dfg: nx.DiGraph
                            ) -> Optional[dgl.DGLGraph]:
        """
        Creates a single DGL heterogeneous graph from CFG, CG, AST, DFG, ...
        It links the graphs using function names as anchors.

        Node Types:
        - 'function': Nodes from the Call Graph (CG)
        - 'block': Nodes from the Control Flow Graph (CFG)
        - 'ast_node': Nodes from the Abstract Syntax Tree (AST)
        - 'dfg_node': Nodes from the Data Flow Graph (DFG)

        Edge Types:
        - ('function', 'calls', 'function'): Original CG edges
        - ('block', 'flows', 'block'): Original CFG edges
        - ('ast_node', 'child_of', 'ast_node'): Original AST edges
        - ('dfg_node', 'data_flow', 'dfg_node'): Original DFG edges

        - ('function', 'has_block', 'block'): Linking CG nodes to CFG nodes
        - ('block', 'part_of_func', 'function'): Reverse of has_block

        - ('function', 'defines_ast', 'ast_node'): Linking CG nodes to AST func def nodes
        - ('ast_node', 'defined_by_func', 'function'): Reverse of defines_ast

        - ('block', 'related_dfg', 'dfg_node'): Linking CFG blocks to DFG nodes
        - ('dfg_node', 'related_block', 'block'): Reverse of related_dfg
        """

        if nx_cg is None:
            logger.error("Call Graph (CG) is None. Cannot create heterogeneous graph as this is core.")
            return None
        if nx_cfg is None:
            logger.error("Control Flow Graph (CFG) is None. There will be missing features.")
        if nx_ast is None:
            logger.error("Abstract Syntax Tree (AST) is None. There will be missing features.")
        if nx_dfg is None:
            logger.warning("Data Flow Graph (DFG) is None. DFG features will be missing.")
        
        
        try:
            # --- 1. Extract all features and create mappings ---
            
            # (graph_type, node_id) -> feature_tensor
            cg_node_features, cg_vuln_labels = self.extract_node_features(nx_cg, "cg") # must have

            cfg_node_features, cfg_vuln_labels = self.extract_node_features(nx_cfg, "cfg") if nx_cfg else ({}, {})
            ast_node_features, _ = self.extract_node_features(nx_ast, "ast") if nx_ast else ({}, {})
            dfg_node_features, dfg_vuln_labels = self.extract_node_features(nx_dfg, "dfg") if nx_dfg else ({}, {})
            
            # (src_id, dst_id) -> feature_tensor
            cg_edge_features, _ = self.extract_edge_features(nx_cg, "cg", {}) # must have

            cfg_edge_features, _ = self.extract_edge_features(nx_cfg, "cfg", {}) if nx_cfg else ({}, {})
            ast_edge_features, _ = self.extract_edge_features(nx_ast, "ast", {}) if nx_ast else ({}, {})
            dfg_edge_features, _ = self.extract_edge_features(nx_dfg, "dfg", {}) if nx_dfg else ({}, {})

            # --- 2. Create Node ID Mappings ---
            # DGL graphs require integer IDs from 0 to N-1 for each node type.
            # We use the original string node IDs from networkx for mapping.
            
            cg_nodes = sorted(list(nx_cg.nodes()))
            cg_node_to_id = {node: i for i, node in enumerate(cg_nodes)}
            
            cfg_node_to_id = {}
            if nx_cfg:
                cfg_nodes = sorted(list(nx_cfg.nodes()))
                cfg_node_to_id = {node: i for i, node in enumerate(cfg_nodes)}

            ast_node_to_id = {}
            if nx_ast:
                ast_nodes = sorted(list(nx_ast.nodes()))
                ast_node_to_id = {node: i for i, node in enumerate(ast_nodes)}

            dfg_node_to_id = {}
            if nx_dfg:
                dfg_nodes = sorted(list(nx_dfg.nodes()))
                dfg_node_to_id = {node: i for i, node in enumerate(dfg_nodes)}

            # --- 3. Build Semantic Lookups for Linking ---
            
            # Map func_name -> list of CFG block nodes
            # (e.g., "MyContract.myFunc(uint)" -> [1, 2, 3])
            cfg_func_to_nodes = defaultdict(list)
            if nx_cfg:
                for node_id, node_data in nx_cfg.nodes(data=True):
                    contract_name = node_data.get("contract_name")
                    func_name = node_data.get("func_name")
                    if func_name and contract_name:
                        cfg_func_to_nodes[(contract_name, func_name)].append(node_id)
            
            # Map func_name -> AST functiondefinition node
            # (e.g., "MyContract.myFunc(uint)" -> "Func_f_123")
            ast_func_to_node = defaultdict(list)
            if nx_ast:
                for node_id, node_data in nx_ast.nodes(data=True):
                    if node_data.get("node_type").lower() == "functiondefinition":
                        func_name = node_data.get("function_name")
                        contract_name = node_data.get("contract_name")
                        if func_name and contract_name:
                            ast_func_to_node[(contract_name, func_name)] = node_id

            dfg_func_to_nodes = defaultdict(list)
            if nx_dfg:
                for node_id, node_data in nx_dfg.nodes(data=True):
                    func_name = node_data.get("function_name")
                    contract_name = node_data.get("contract_name")
                    if func_name and contract_name:
                        dfg_func_to_nodes[(contract_name, func_name)].append(node_id)


            # --- 4. Initialize `data_dict` for DGL ---
            data_dict = {}

            # --- 5. Process "Source/Origin" Edges ---
            
            # CG def ('function', 'calls', 'function')
            cg_src, cg_dst, cg_efeat = [], [], []
            for src, dst in nx_cg.edges():
                if src in cg_node_to_id and dst in cg_node_to_id:
                    cg_src.append(cg_node_to_id[src])
                    cg_dst.append(cg_node_to_id[dst])
                    cg_efeat.append(cg_edge_features[("cg", str(src), str(dst))])
            data_dict[('function', 'calls', 'function')] = (torch.tensor(cg_src, dtype=self.int_dtype), torch.tensor(cg_dst, dtype=self.int_dtype))

            # CFG def ('block', 'flows', 'block')
            cfg_src, cfg_dst, cfg_efeat = [], [], []
            if nx_cfg:
                for src, dst in nx_cfg.edges():
                    if src in cfg_node_to_id and dst in cfg_node_to_id:
                        cfg_src.append(cfg_node_to_id[src])
                        cfg_dst.append(cfg_node_to_id[dst])
                        cfg_efeat.append(cfg_edge_features[("cfg", str(src), str(dst))])

            data_dict[('block', 'flows', 'block')] = (torch.tensor(cfg_src, dtype=self.int_dtype), torch.tensor(cfg_dst, dtype=self.int_dtype))

            # AST def ('ast_node', 'child_of', 'ast_node')
            ast_src, ast_dst, ast_efeat = [], [], []
            if nx_ast:
                for src, dst in nx_ast.edges():
                    if src in ast_node_to_id and dst in ast_node_to_id:
                        ast_src.append(ast_node_to_id[src])
                        ast_dst.append(ast_node_to_id[dst])
                        ast_efeat.append(ast_edge_features[("ast", str(src), str(dst))])
            data_dict[('ast_node', 'child_of', 'ast_node')] = (torch.tensor(ast_src, dtype=self.int_dtype), torch.tensor(ast_dst, dtype=self.int_dtype))

            # DFG def ('dfg_node', 'data_flow', 'dfg_node')
            dfg_src, dfg_dst, dfg_efeat = [], [], []
            if nx_dfg:
                for src, dst in nx_dfg.edges():
                    if src in dfg_node_to_id and dst in dfg_node_to_id:
                        dfg_src.append(dfg_node_to_id[src])
                        dfg_dst.append(dfg_node_to_id[dst])
                        dfg_efeat.append(dfg_edge_features[("dfg", str(src), str(dst))])
            data_dict[('dfg_node', 'data_flow', 'dfg_node')] = (torch.tensor(dfg_src, dtype=self.int_dtype), torch.tensor(dfg_dst, dtype=self.int_dtype))

            # --- 6. Process NEW Linking Edges (The "additional edges") ---
            link_func_block_src, link_func_block_dst = [], []
            link_block_func_src, link_block_func_dst = [], []
            link_func_ast_src, link_func_ast_dst = [], []
            link_ast_func_src, link_ast_func_dst = [], []
            link_block_dfg_src, link_block_dfg_dst = [], []
            link_dfg_block_src, link_dfg_block_dst = [], []

            for cg_node_str, cg_node_id in cg_node_to_id.items():
                node_data = nx_cg.nodes[cg_node_str]
                func_name = node_data.get("func_name")
                contract_name = node_data.get("contract_name")
                
                if not func_name or not contract_name:
                    logger.warning(f"CG missing func_name {func_name} or contract_name {contract_name}, skipping linking.")
                    continue

                # A. Link CG 'function' -> CFG 'block' (One-to-Many)
                if nx_cfg:
                    cfg_key = (contract_name, func_name)
                    if cfg_key in cfg_func_to_nodes:
                        for cfg_node_str in cfg_func_to_nodes[cfg_key]:
                            if cfg_node_str in cfg_node_to_id: # Ensure block node exists
                                cfg_node_id = cfg_node_to_id[cfg_node_str]
                                
                                link_func_block_src.append(cg_node_id)
                                link_func_block_dst.append(cfg_node_id)
                                
                                link_block_func_src.append(cfg_node_id)
                                link_block_func_dst.append(cg_node_id)

                # B. Link CG 'function' -> AST 'ast_node' (One-to-One)
                if nx_ast:
                    ast_key = (contract_name, func_name)
                    if ast_key in ast_func_to_node:
                        ast_node_str = ast_func_to_node[ast_key]
                        if ast_node_str in ast_node_to_id: # Ensure ast node exists
                            ast_node_id = ast_node_to_id[ast_node_str]

                            link_func_ast_src.append(cg_node_id)
                            link_func_ast_dst.append(ast_node_id)

                            link_ast_func_src.append(ast_node_id)
                            link_ast_func_dst.append(cg_node_id)

                # C. Link CFG 'block' -> DFG 'dfg_node' (One-to-Many)
                # C. Link 'block' -> 'dfg_node' (based on 'src' containment)
            if nx_cfg and nx_dfg:
                cfg_key = (contract_name, func_name)
                dfg_key = (contract_name, func_name)

                # Check if both graphs have nodes for this function
                if cfg_key in cfg_func_to_nodes and dfg_key in dfg_func_to_nodes:
                    
                    # Iterate through all CFG blocks for this function
                    for cfg_node_str in cfg_func_to_nodes[cfg_key]:
                        if cfg_node_str not in cfg_node_to_id:
                            continue
                        cfg_node_id = cfg_node_to_id[cfg_node_str]
                        
                        # Safely evaluate the src attribute for the CFG block
                        cfg_src_raw = nx_cfg.nodes[cfg_node_str].get("code_lines")
                        cfg_src = self._safe_literal_eval(cfg_src_raw, default=None)

                        # Skip block if its src is not a valid [start, end] pair
                        if not (isinstance(cfg_src, list) and len(cfg_src) == 2):
                            continue  
                        
                        try:
                            cfg_start, cfg_end = float(cfg_src[0]), float(cfg_src[1])
                        except (ValueError, TypeError):
                            continue # Skip if src values aren't numeric

                        # Iterate through all DFG nodes for this function
                        for dfg_node_str in dfg_func_to_nodes[dfg_key]:
                            if dfg_node_str not in dfg_node_to_id:
                                continue
                            dfg_node_id = dfg_node_to_id[dfg_node_str]
                            
                            # Safely evaluate the src attribute for the DFG node
                            dfg_src_raw = nx_dfg.nodes[dfg_node_str].get("code_lines")
                            dfg_src = self._safe_literal_eval(dfg_src_raw, default=None)
                            
                            # Skip DFG node if its src is not valid
                            if not (isinstance(dfg_src, list) and len(dfg_src) == 2):
                                continue 
                            
                            try:
                                dfg_start, dfg_end = float(dfg_src[0]), float(dfg_src[1])
                            except (ValueError, TypeError):
                                continue # Skip if src values aren't numeric

                            # --- THE KEY: LINK CONDITION ---
                            # Link if the block's range CONTAINS the DFG node's range.
                            if (cfg_start <= dfg_start) and (cfg_end >= dfg_end):
                                link_block_dfg_src.append(cfg_node_id)
                                link_block_dfg_dst.append(dfg_node_id)
                                
                                link_dfg_block_src.append(dfg_node_id)
                                link_dfg_block_dst.append(cfg_node_id)
            
            data_dict[('function', 'has_block', 'block')] = (torch.tensor(link_func_block_src, dtype=self.int_dtype), torch.tensor(link_func_block_dst, dtype=self.int_dtype))
            data_dict[('block', 'part_of_func', 'function')] = (torch.tensor(link_block_func_src, dtype=self.int_dtype), torch.tensor(link_block_func_dst, dtype=self.int_dtype))

            data_dict[('function', 'defines_ast', 'ast_node')] = (torch.tensor(link_func_ast_src, dtype=self.int_dtype), torch.tensor(link_func_ast_dst, dtype=self.int_dtype))
            data_dict[('ast_node', 'defined_by_func', 'function')] = (torch.tensor(link_ast_func_src, dtype=self.int_dtype), torch.tensor(link_ast_func_dst, dtype=self.int_dtype))

            data_dict[('block', 'related_dfg', 'dfg_node')] = (torch.tensor(link_block_dfg_src, dtype=self.int_dtype), torch.tensor(link_block_dfg_dst, dtype=self.int_dtype))
            data_dict[('dfg_node', 'related_block', 'block')] = (torch.tensor(link_dfg_block_src, dtype=self.int_dtype), torch.tensor(link_dfg_block_dst, dtype=self.int_dtype))

            # --- VALIDATION: Check edge consistency BEFORE creating DGL graph ---
            num_nodes_dict = {
                'function': len(cg_nodes),
                'block': len(cfg_nodes) if nx_cfg else 0,
                'ast_node': len(ast_nodes) if nx_ast else 0,
                'dfg_node': len(dfg_nodes) if nx_dfg else 0
            }
            
            # Validate that all edges reference valid node IDs
            for etype, (src_ids, dst_ids) in data_dict.items():
                src_type, _, dst_type = etype
                src_tensor = src_ids if isinstance(src_ids, torch.Tensor) else torch.tensor(src_ids)
                dst_tensor = dst_ids if isinstance(dst_ids, torch.Tensor) else torch.tensor(dst_ids)
                
                if len(src_tensor) > 0:
                    max_src = src_tensor.max().item()
                    max_dst = dst_tensor.max().item()
                    
                    if max_src >= num_nodes_dict[src_type]:
                        logger.error(f"Edge type {etype}: source node ID {max_src} >= num {src_type} nodes ({num_nodes_dict[src_type]})")
                        logger.error(f"Project: {project_name}")
                        raise ValueError(f"Invalid edge: source node {max_src} out of range for type {src_type}")
                    
                    if max_dst >= num_nodes_dict[dst_type]:
                        logger.error(f"Edge type {etype}: dest node ID {max_dst} >= num {dst_type} nodes ({num_nodes_dict[dst_type]})")
                        logger.error(f"Project: {project_name}")
                        raise ValueError(f"Invalid edge: dest node {max_dst} out of range for type {dst_type}")

            # --- 7. Create DGL Heterograph ---
            g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
            
            # --- 8. Assign Node Features --- 
            # We must stack the features in the *exact* order of the sorted node lists.
            # Convert all node IDs to strings since that's how they're stored in the feature dicts
            g.nodes['function'].data['feat'] = torch.stack([cg_node_features[("cg", str(n))] for n in cg_nodes])
            if nx_cfg:
                g.nodes['block'].data['feat'] = torch.stack([cfg_node_features[("cfg", str(n))] for n in cfg_nodes])
            if nx_ast:
                g.nodes['ast_node'].data['feat'] = torch.stack([ast_node_features[("ast", str(n))] for n in ast_nodes])
            if nx_dfg:
                g.nodes['dfg_node'].data['feat'] = torch.stack([dfg_node_features[("dfg", str(n))] for n in dfg_nodes])

            # --- 9. Assign Edge Features ---
            if cg_efeat:
                g.edges[('function', 'calls', 'function')].data['feat'] = torch.stack(cg_efeat)
            if cfg_efeat:
                g.edges[('block', 'flows', 'block')].data['feat'] = torch.stack(cfg_efeat)
            if ast_efeat:
                g.edges[('ast_node', 'child_of', 'ast_node')].data['feat'] = torch.stack(ast_efeat)
            if dfg_efeat:
                g.edges[('dfg_node', 'data_flow', 'dfg_node')].data['feat'] = torch.stack(dfg_efeat)
            
            # As discussed, the linking edges do not have features.
            # Their structure is the feature.

            # --- 10. Vulnerability Labels ---
            # For nodes without vulnerability labels, create benign label (all zeros - no vulnerabilities)
            # Helper function to create benign label
            def create_benign_label():
                """Create a benign label vector: all zeros (no vulnerabilities = benign)"""
                benign = torch.zeros(len(OWASP_VULN), dtype=self.dtype)
                return benign
            
            cg_vuln_tensors = []
            for n in cg_nodes:
                if n in cg_vuln_labels:
                    cg_vuln_tensors.append(torch.tensor(cg_vuln_labels[n], dtype=self.dtype))
                else:
                    cg_vuln_tensors.append(create_benign_label())
            
            cfg_vuln_tensors = []
            for n in cfg_nodes:
                if n in cfg_vuln_labels:
                    cfg_vuln_tensors.append(torch.tensor(cfg_vuln_labels[n], dtype=self.dtype))
                else:
                    cfg_vuln_tensors.append(create_benign_label())
            
            dfg_vuln_tensors = []
            if nx_dfg and dfg_nodes:
                for n in dfg_nodes:
                    if n in dfg_vuln_labels:
                        dfg_vuln_tensors.append(torch.tensor(dfg_vuln_labels[n], dtype=self.dtype))
                    else:
                        dfg_vuln_tensors.append(create_benign_label())
            # AST nodes don't have vuln labels, so we can skip
            #g.nodes['ast_node'].data['vuln'] = torch.zeros((len(ast_nodes), len(OWASP_VULN)), dtype=self.dtype)

            # Stack tensors, using empty tensor if list is empty
            cg_vuln_stacked = torch.stack(cg_vuln_tensors) if cg_vuln_tensors else torch.empty(0, len(OWASP_VULN), dtype=self.dtype)
            cfg_vuln_stacked = torch.stack(cfg_vuln_tensors) if cfg_vuln_tensors else torch.empty(0, len(OWASP_VULN), dtype=self.dtype)
            dfg_vuln_stacked = torch.stack(dfg_vuln_tensors) if dfg_vuln_tensors else torch.empty(0, len(OWASP_VULN), dtype=self.dtype)

            return (project_name, {'graph': g, 'cg_vuln': cg_vuln_stacked, 'cfg_vuln': cfg_vuln_stacked, 'dfg_vuln': dfg_vuln_stacked})
        except Exception as e:
            logger.error(f"Error in create_hetero_graph: {e}")
            logger.error(traceback.format_exc())
            return None
        
    ############################################################
    ############################################################
    def pipeline_graphs_to_features(self, obj_nx_graphs: List , mode :str="train"):
        try:
            results = []
            self.mode = mode
            for (project_name, proj_graphs_obj) in tqdm(obj_nx_graphs, desc=f"Processing graphs"):
                cfg_graph = proj_graphs_obj.get('CFG_combined.gpickle') #done
                cg_graph = proj_graphs_obj.get('CallGraph_combined.gpickle')#done
                ast_graph = proj_graphs_obj.get('AST_combined.gpickle')#done
                dfg_graph = proj_graphs_obj.get('DFG_combined.gpickle', None)
                results.append(self.create_hetero_graph(project_name, cg_graph, cfg_graph, ast_graph, dfg_graph))
            results = [g for g in results if g is not None]
            return results
        except Exception as e:
            logger.error(f"Error in pipeline_graphs_to_features: {e}")
            logger.error(traceback.format_exc())
            return []
        
if __name__ == "__main__":
    def make_cfg_graph():
        g = nx.DiGraph()
        g.add_node(1, node_type="basic_block", raw_ir="x := y + z", raw_code="x = y + z;",
                contract_name="C", func_name="f(uint256)", structs=["S1", "S2"]) 
        g.add_node(2, node_type="basic_block", raw_ir="return x", raw_code="return x;",
                contract_name="C", func_name="f(uint256)", structs=[])
        g.add_edge(1, 2, edge_type="fallthrough", frequency=1.0)
        return g

    def make_cg_graph():
        g = nx.DiGraph()
        g.add_node("C.f", node_type="function", contract_name="C", contract_kind="contract",
                func_name="f(uint256)", structs=["S1"], stateMutability="nonpayable", visibility="public",
                modifiers=["onlyOwner"], isImplemented=True, isConstructor=False, isFallback=False, isReceive=False,
                opcode_hist={"ADD": 2, "CALL": 1})
        g.add_node("C.g", node_type="function", contract_name="C", contract_kind="contract",
                func_name="g()", structs=[], stateMutability="view", visibility="external",
                modifiers=[], isImplemented=True, isConstructor=False, isFallback=False, isReceive=False,
                opcode_hist={"MLOAD": 1})
        g.add_edge("C.f", "C.g", edge_type="calls", frequency=1.0, opcodes=["CALL"])
        return g

    def make_ast_graph():
        g = nx.DiGraph()
        g.add_node("VarDecl_x", node_type="variable_declaration", contract_name="C", function_name="f(uint256)",
                variable_name="x", event_name="", opcode_hist={}, contract_kind="contract")
        g.add_node("Func_f", node_type="function_definition", contract_name="C", function_name="f(uint256)",
                variable_name="", event_name="", opcode_hist={}, contract_kind="contract")
        g.add_edge("Func_f", "VarDecl_x", edge_type="contains")
        return g

    extractor = GraphFeatureExtractor(tokenizer=None)

    tests = [
        ("cfg", make_cfg_graph()),
        ("cg", make_cg_graph()),
        ("ast", make_ast_graph()),
    ]

    for gtype, g in tests:
        print(f"\n=== Testing graph type: {gtype} ===")
        dgl_g, gfeat, _ = extractor.process_single_graph(g, gtype)
        print(f"Nodes: {dgl_g.number_of_nodes()}, Edges: {dgl_g.number_of_edges()}")
        nfeat = dgl_g.ndata.get('feat')
        efeat = dgl_g.edata.get('feat')
        print(f"Node feature tensor shape: {tuple(nfeat.shape) if nfeat is not None else None}")
        print(f"Edge feature tensor shape: {tuple(efeat.shape) if efeat is not None else None}")
        print(f"Graph-level feature shape: {tuple(gfeat.shape)} | dtype: {gfeat.dtype}")
