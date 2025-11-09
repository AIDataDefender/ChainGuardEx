

import os
from typing import Dict, List
OPCODE_VOCAB = [

        # ===============================
        # Arithmetic / Logical
        # ===============================
        'ADD',       # a + b
        'SUB',       # a - b
        'MUL',       # a * b
        'DIV',       # a / b
        'MOD',       # a % b
        'EXP',       # a ** b
        'NOT',       # !a or ~a
        'NEG',       # -a
        'AND',       # a && b or bitwise
        'OR',        # a || b or bitwise
        'XOR',       # bitwise XOR
        'LT',        # less than
        'GT',        # greater than
        'LE',        # <=
        'GE',        # >=
        'EQ',        # ==
        'NE',        # !=

        # ===============================
        # Variable / Data Movement
        # ===============================
        'ASSIGN',         # x = y
        'PHI',            # SSA merge node
        'LOAD',           # load from memory
        'STORE',          # store to memory
        'STORAGELOAD',    # load from storage
        'STORAGESTORE',   # store to storage
        'LOCAL',          # local variable declaration
        'CONST',          # constant literal assignment
        'TMP',            # temporary SSA variable

        # ===============================
        # Control Flow
        # ===============================
        'JUMP',           # unconditional jump
        'JUMPI',          # conditional jump
        'IF',             # if-branch
        'RETURN',         # return from function
        'REVERT',         # revert execution
        'THROW',          # legacy revert
        'STOP',           # halt execution
        'CONTINUE',       # loop continue
        'BREAK',          # loop break
        'ENDLOOP',        # internal marker for end of loop
        'PHI',            # SSA merge point (used at branch joins)

        # ===============================
        # Function Calls
        # ===============================
        'CALL',             # generic call (could be internal or external)
        'INTERNALCALL',     # call to another function in same contract
        'HIGHLLEVELCALL',   # high-level external call (e.g., token.transfer)
        'LIBRARYCALL',      # library function call
        'DELEGATECALL',     # delegatecall
        'STATICCALL',       # static call (read-only)
        'CONSTRUCTORCALL',  # contract creation
        'CREATE',           # low-level create
        'CREATE2',          # deterministic create

        # ===============================
        # Storage & Memory Ops
        # ===============================
        'MLOAD',           # memory load
        'MSTORE',          # memory store
        'SLOAD',           # storage load (EVM-level)
        'SSTORE',          # storage store (EVM-level)
        'BALANCE',         # address.balance
        'CALLVALUE',       # msg.value
        'CALLDATALOAD',    # calldata read
        'CALLDATASIZE',    # calldata size

        # ===============================
        # Environment / Blockchain Context
        # ===============================
        'TIMESTAMP',       # block.timestamp
        'NUMBER',          # block.number
        'COINBASE',        # block.coinbase
        'DIFFICULTY',      # block.difficulty
        'GAS',             # gasleft()
        'GASPRICE',        # tx.gasprice
        'ORIGIN',          # tx.origin
        'SENDER',          # msg.sender
        'VALUE',           # msg.value
        'ADDRESS',         # address(this)
        'BALANCEOF',       # balance of an address

        # ===============================
        # Data Structures / References
        # ===============================
        'INDEX',           # arr[i]
        'LENGTH',          # arr.length
        'MAPLOAD',         # mapping lookup
        'MAPSTORE',        # mapping update
        'STRUCTLOAD',      # struct field access
        'STRUCTSTORE',     # struct field write

        # ===============================
        # Events / Logging
        # ===============================
        'EVENTCALL',       # emit Event(...)
        'LOG',             # low-level LOG opcodes (LOG0–LOG4)

        # ===============================
        # Termination
        # ===============================
        'SELFDESTRUCT',    # contract selfdestruct
]

# === Module-level constant lists used by one-hot helpers ===
OWASP_VULN = [
    "SC01:2025",  # Reentrancy
    # "SC02:2025",
    "SC03:2025",  # Access Control
    "SC04:2025",  # Arithmetic
    "SC05:2025",  # Unchecked Call
    "SC06:2025",  # Denial of Service
    # "SC07:2025",
    "SC08:2025",  # Bad Randomness
    "SC09:2025",  # Front Running
    "SC10:2025",  # Time Manipulation
    # BENIGN removed - absence of all vulnerabilities = benign (multilabel)
]

VAR_VISIBILITY = [
    "private",
    "public",
    "internal",
]

FUNC_VISIBILITY = [
    "private",
    "internal",
    "external",
    "public",
]

VAR_STORAGE = [
    'memory', 'storage', 'calldata', 'none'
]

STATE_MUTABILITY = [
    'pure', 'view', 'nonpayable', 'payable', 'none'
]

EVENT_META = [
    'anonymous', 'non-anonymous', 'indexed', 'override', 'virtual', 'returns', 'none'
]

OPCODE_HIST = [
    "CALL",
    "STORAGE",
    "ADD_SAFE",
    "SUB_SAFE",
    "MUL_SAFE",
    "DIV_SAFE",
    "ADD_RAW",
    "SUB_RAW",
    "MUL_RAW",
    "DIV_RAW",
    "REQUIRE",
    "REVERT",
    "COND_BRANCH",
    "PHI",
]

# CFG node types (kept in sync with graph_processing expectations)
CFG_NODE_TYPE_LIST = [
    "entry_point",
    "expression",
    "return",
    "if",
    "new variable",
    "inline asm",
    "end inline asm",
    "if_loop",
    "end_if",
    "begin_loop",
    "end_loop",
    "throw",
    "break",
    "continue",
    "try",
    "catch",
    "other_entrypoint",
    "other",
]

# CG node types
CG_NODE_TYPE_LIST = [
    "contract_function",
    "fallback_function",
    "other",
]

# AST node types
AST_NODE_TYPE_LIST = [
    "functiondefinition", "contractdefinition", "ifstatement", "whilestatement", "forstatement", "block", "expressionstatement", "variabledeclarationstatement", "return", "assignment", "binaryoperation", "unaryoperation", "functioncall", "identifier", "literal", "memberaccess", "indexaccess", "typeconversion", "throw", "revert", "assert", "require", "inheritance", "modifierinvocation", "arraytype", "mapping", "newexpression", "tupleexpression", "variabledeclaration", "modifierdefinition", "eventdefinition", "structdefinition", "enumdefinition", "usingfordirective", "parameterlist", "other"
]

# Edge type lists
CFG_EDGE_TYPES = ["if_true", "if_false", "loop_exit", "loop_continue", "loop_back", "call", "return", "next"]

CG_EDGE_TYPES = ["internal_call", "external_call", "delegatecall", "callcode", "staticcall"]

AST_EDGE_TYPES = ["controlflow", "declaration", "expression", "functioncall", "access", "conversion", "return", "errorhandling", "inheritance", "modifierinvocation", "arraytype", "mappingtype", "newexpression", "tupleexpression", "variabledeclarationstatement", "contractbody", "functionparameters", "modifierparameters", "blockstatement", "next"]

# DFG edge_type values 
DFG_EDGE_TYPES = ["init", "assign", "var-write", "var-read", "call-arg", "return", "member", "index-base", "index-idx"]

# DFG flow labels 
DFG_FLOW_LABELS = ["init_to_var", "rhs_to_lhs", "rhs_to_decl", "decl_to_use", "arg_to_call", "decl_to_call_arg", "expr_to_return", "decl_to_return", "base_to_member", "decl_to_member", "base_to_index", "index_to_index", "decl_to_index"]

CONTRACT_KIND = [
    "library",
    "interface",
    "abstract",
    "contract",
    "none",
]

MODIFIERS = [
    "pure",
    "view",
    "payable",
    "constant",
    "virtual",
    "override",
    "indexed",
    "anonymous",
    "nonpayable",
    "external",
]

CALL_OPCODE = [
    # --- IR-level ---
    "CALL",                # generic low-level call
    "DELEGATECALL",        # delegatecall
    "CALLCODE",            # legacy call with code context
    "STATICCALL",          # staticcall (view)

    # --- Source-level pattern matches ---
    "DELEGATECALL_PATTERN",   # `.delegatecall(` in Solidity code
    "STATICCALL_PATTERN",     # `.staticcall(` in Solidity code
    "CALL_PATTERN",           # `.call(` in Solidity code
    "SEND_TRANSFER_PATTERN",  # `.send(` or `.transfer(`
    "LOW_LEVEL_DELEGATECALL", # bare `delegatecall(`
    "LOW_LEVEL_STATICCALL",   # bare `staticcall(`
    "LOW_LEVEL_CALL",         # bare `call(`

    # --- Cross-contract inference ---
    "CROSS_CONTRACT",         # destination in another contract
]


def get_all_list_lengths() -> Dict[str, int]:
    """Return lengths of the constant lists so callers (eg. GraphFeatureExtractor)
    can compute feature dimensions dynamically.
    """
    return {
        'OWASP_VULN': len(OWASP_VULN),
        'VAR_VISIBILITY': len(VAR_VISIBILITY),
        'FUNC_VISIBILITY': len(FUNC_VISIBILITY),
        'VAR_STORAGE': len(VAR_STORAGE),
        'STATE_MUTABILITY': len(STATE_MUTABILITY),
        'EVENT_META': len(EVENT_META),
        'OPCODE_HIST': len(OPCODE_HIST),
        'CFG_NODE_TYPE_LIST': len(CFG_NODE_TYPE_LIST),
        'CG_NODE_TYPE_LIST': len(CG_NODE_TYPE_LIST),
        'AST_NODE_TYPE_LIST': len(AST_NODE_TYPE_LIST),
        # 'DFG_NODE_TYPE_LIST': len(DFG_NODE_TYPE_LIST),
        'CFG_EDGE_TYPES': len(CFG_EDGE_TYPES),
        'CG_EDGE_TYPES': len(CG_EDGE_TYPES),
        'AST_EDGE_TYPES': len(AST_EDGE_TYPES),
        'DFG_EDGE_TYPES': len(DFG_EDGE_TYPES),
        'CONTRACT_KIND': len(CONTRACT_KIND),
        'MODIFIERS': len(MODIFIERS),
        'OPCODE_VOCAB': len(OPCODE_VOCAB),
    }

#############################################################
# region Vectors
#############################################################


def vuln_to_label(owasp_list):
    """
    Convert OWASP vulnerability list to multi-hot label vector.
    
    For multilabel classification with 8 vulnerability types:
    - If owasp_list is None/empty: returns all zeros (benign = no vulnerabilities)
    - Otherwise: sets corresponding vulnerability indices to 1.0
    
    Args:
        owasp_list: List of OWASP vulnerability IDs (e.g., ['SC01:2025', 'SC03:2025'])
    
    Returns:
        List of floats (length = 8) with 1.0 for present vulnerabilities, 0.0 otherwise
    """
    labels = [0.0] * len(OWASP_VULN)  # Initialize all to 0.0
    
    # Case 1: No vulnerabilities - return all zeros (benign)
    if not owasp_list:
        return labels
    
    # Case 2: Has vulnerabilities - set corresponding indices
    found_any = False
    for owasp in (owasp_list or []):
        key = str(owasp).upper().strip()
        
        if key in OWASP_VULN:
            idx = OWASP_VULN.index(key)
            labels[idx] = 1.0
            found_any = True
        else:
            # WARNING: Unknown vulnerability ID - not in OWASP_VULN list
            print(f"vuln_to_label: Unknown vulnerability '{key}' not in OWASP_VULN list")
    
    # SANITY CHECK: If we had a list but found nothing, log warning
    if not found_any and owasp_list:
        print(f"vuln_to_label: Had vulnerability list {owasp_list} but no matches found!")
    
    return labels

def op_code_histogram_to_feat(opcode_hist : Dict):
        
    # Use module-level OPCODE_HIST list defined above
    # produce a simple vector counts aligned with OPCODE_HIST
    op_hist_vector = [0.0] * len(OPCODE_HIST)
    if not opcode_hist:
        return op_hist_vector
    for i, k in enumerate(OPCODE_HIST):
        # opcode_hist may contain counts keyed by opcode-like strings
        op_hist_vector[i] = float(opcode_hist.get(k, 0))
    return op_hist_vector
    
def call_opcode_to_feat(call_opcodes: List):
    call_op_vec = [0.0] * len(CALL_OPCODE)
    if not call_opcodes:
        return call_op_vec  
    for co in call_opcodes:
        c = co.upper()
        if c in CALL_OPCODE:
            call_op_vec[CALL_OPCODE.index(c)] += 1.0
    return call_op_vec



#############################################################
# region One-hot 
#############################################################

def var_visibility_to_one_hot(vis:str):
    one_hot = [0.0] * len(VAR_VISIBILITY)
    if not vis:
        return one_hot
    v = vis.lower()
    if v in VAR_VISIBILITY:
        one_hot[VAR_VISIBILITY.index(v)] = 1.0
    return one_hot

def func_visibility_to_one_hot(vis:str):
    one_hot = [0.0] * len(FUNC_VISIBILITY)
    if not vis:
        return one_hot
    v = vis.lower()
    if v in FUNC_VISIBILITY:
        one_hot[FUNC_VISIBILITY.index(v)] = 1.0
    return one_hot

def var_storage_to_one_hot(storage:str):
    one_hot = [0.0] * len(VAR_STORAGE)
    if not storage:
        return one_hot
    s = storage.lower()
    if s in VAR_STORAGE:
        one_hot[VAR_STORAGE.index(s)] = 1.0
    return one_hot

def func_stateMutability_to_one_hot(stateMutability:str):
    one_hot = [0.0] * len(STATE_MUTABILITY)
    if not stateMutability:
        return one_hot
    s = stateMutability.lower()
    if s in STATE_MUTABILITY:
        one_hot[STATE_MUTABILITY.index(s)] = 1.0
    return one_hot

def event_meta_to_one_hot(meta:str):
    one_hot = [0.0] * len(EVENT_META)
    if not meta:
        return one_hot
    m = meta.lower()
    if m in EVENT_META:
        one_hot[EVENT_META.index(m)] = 1.0
    return one_hot

def contract_kind_to_one_hot(kind:str):
    one_hot = [0.0] * len(CONTRACT_KIND)
    if not kind:
        return one_hot
    k = kind.lower()
    if k in CONTRACT_KIND:
        one_hot[CONTRACT_KIND.index(k)] = 1.0
    return one_hot

def node_type_to_one_hot(node_type: str, type_list: list) -> list:
    """Encode node type to one-hot vector."""
    one_hot = [0.0] * len(type_list)
    if not node_type:
        return one_hot
    nt = node_type.lower()
    if nt in type_list:
        idx = type_list.index(nt)
        one_hot[idx] = 1.0
    return one_hot

def edge_type_to_one_hot(edge_type: str, type_list: list) -> list:
    """Encode edge type to one-hot vector."""
    one_hot = [0.0] * len(type_list)
    if not edge_type:
        return one_hot
    et = edge_type.lower()
    if et in type_list:
        idx = type_list.index(et)
        one_hot[idx] = 1.0
    return one_hot