# --- 1. Enums & Constants ---
from enum import StrEnum, auto
from pathlib import Path

# Directory paths
SOURCE_CODE_DIR = Path("DAppSCAN-source/contracts")
EXTRACTED_GRAPHS_DIR = Path("Extracted_Graphs")

# Separator used for unique names (e.g. Contract$Function$File)
SEPARATOR = "$"
LIST_SEP = "¥"


class GraphAttributes:
    """Attributes used in the graph nodes and edges."""

    CONTRACT = "contract"
    FUNCTION = "function"
    FILE = "file"

    UNIQUE_NAME = "unique_name"
    LABEL = "label"

    NODE_ID = "node_id"
    NODE_TYPE = "node_type"
    SUB_NODE_TYPE = "sub_node_type"
    EDGE_TYPE = "edge_type"
    SUB_EDGE_TYPE = "sub_edge_type"

    EXPRESSION = "expression"
    IR = "ir"
    RAW_CODE = "raw_code"

    VULN = "vuln"

    SRC = "src"

    VAR = "variable"  # DF
    CALL_SITE_ID = "call_site_id"
    CALL_DEST = "call_dest"
    CALL_ARG_COUNT = "call_arg_count"
    CALL_RETURNS_VALUE = "call_returns_value"

    #DGL
    DGL_TEXT_FEATURE = "dgl_text_feature"
    DGL_DATA = "data"



class EdgeTypes(StrEnum):
    """Types of edges in the CPG."""

    # Control Flow
    CF = auto()
    CF_TRUE = auto()
    CF_FALSE = auto()
    LOOP_EXIT = auto()
    LOOP_CONTINUE = auto()
    LOOP_BACK = auto()
    RETURN = auto()

    # Call Graph
    CALL = auto()

    # Data Flow
    DF = auto()
    DEF_USE = auto()
    DF_INIT = auto()
    DF_ASSIGN = auto()
    DF_CALL_ARG = auto()
    DF_RETURN = auto()
    DF_MEMBER = auto()
    DF_INDEX_BASE = auto()
    DF_INDEX_EXPR = auto()
    PARAMETER_IN = auto()
    PARAMETER_OUT = auto()

    # AST
    AST_CHILD = auto()

    # Linking
    RETURN_CALL = auto()
    AST_TO_CFG = auto()
    AST_TO_CFG_STRUCTURAL = auto()
    AST_CALL = auto()


class NodeTypes(StrEnum):
    """Types of nodes in the CPG."""

    CFG_NODE = auto()
    AST_NODE = auto()


FLOW_LABELS = {
    "VariableDeclarationStatement": "df_init",
    "Assignment": "df_assign",
    "FunctionCall": "df_call_arg",
    "Return": "df_return",
    "MemberAccess": "df_member",
    "IndexAccess": "df_index_base",
    "IndexExpr": "df_index_expr",
}
