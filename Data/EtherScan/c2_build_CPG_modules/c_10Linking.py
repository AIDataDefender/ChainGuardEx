import json
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import networkx as nx
from collections import defaultdict
import hashlib

from c2_build_CPG_modules.c_1AST import load_and_process_ast
from c2_build_CPG_modules.c_3IR import load_and_process_irs
from c2_build_CPG_modules.c_9CFG import load_and_process_cfg
import pickle

from .c_2constants import (
    NodeTypes,
    SEPARATOR,
    GraphAttributes,
)
from .c_2constants import EdgeTypes

# Debug flag for CFG/CG linking
DEBUG_LINKING = True


# def _escape_dot_value(value: Any) -> Any:
#     if isinstance(value, str):
#         sanitized = value.replace("\\", "\\\\").replace('"', '\\"')
#         sanitized = sanitized.replace("\r", "").replace("\n", "\\n")
#         return sanitized
#     return value


def clean_function_signature(func_sig):
    """
    Split function signature into name and cleaned args (types only).

    Args:
        func_sig: Function signature like "tryAdd(uint256 a,uint256 b)"

    Returns:
        tuple: (function_name, cleaned_args) like ("tryAdd", "(uint256,uint256)")
    """
    match = re.match(r"([^(]+)\((.*)\)", func_sig)
    if match:
        name = match.group(1)
        args = match.group(2)
        if args:
            cleaned_args = []
            for arg in args.split(","):
                arg = arg.strip()
                if " " in arg:
                    type_part = arg.split()[0]
                    cleaned_args.append(type_part)
                else:
                    cleaned_args.append(arg)
            cleaned_args_str = ",".join(cleaned_args)
        else:
            cleaned_args_str = ""
        return name, f"({cleaned_args_str})"
    else:
        return func_sig, ""


def _normalize_code_fragment(fragment: Optional[str]) -> str:
    """
    Normalize code fragment for matching between CFG and source code.

    Handles:
    1. Function type signatures: require(bool,string)(...) -> require(...)
    2. Constructor signatures: constructor(address,address,...) -> constructor(...)
    3. Function call type annotations: func(type1,type2)(...) -> func(...)
    4. Parameter names in declarations: func(uint256 x, bool y) -> func(uint256,bool)
    5. Whitespace removal

    Args:
        fragment: Code fragment to normalize

    Returns:
        Normalized code fragment
    """
    if not fragment:
        return ""

    normalized = fragment

    # Step 1: Handle function/require/call with type signatures
    # Matches: require(bool,string)(...) -> require(...)
    # Matches: constructor(address,address,uint256)(...) -> constructor(...)
    # Matches: SomeFunc(type1,type2)(...) -> SomeFunc(...)

    # Pattern: FunctionName(TypeList)(...) where TypeList contains type keywords
    type_keywords = (
        r"(?:address|uint\d*|int\d*|bool|string|bytes\d*|byte|tuple|mapping)"
    )

    # Match function with type signature followed by parentheses
    # e.g., require(bool,string)(...) or constructor(address,address,uint256)(...)
    # More specific: ensure we're matching type signatures (no spaces between commas and types)
    pattern_typed_call = (
        rf"(\w+)\({type_keywords}(?:\s*,\s*{type_keywords})*\)(\([^\)]*\))"
    )

    # Replace iteratively until no more matches (handles nested cases)
    prev = None
    while prev != normalized:
        prev = normalized
        normalized = re.sub(pattern_typed_call, r"\1\2", normalized)

    # Also handle require/revert/assert with type signatures but no second parens
    # e.g., require(bool,string) -> require
    # More specific pattern that only matches pure type lists
    pattern_typed_stmt = rf"(\b(?:require|revert|assert|call|delegatecall|staticcall)\b)\({type_keywords}(?:\s*,\s*{type_keywords})*\)\s*$"
    normalized = re.sub(pattern_typed_stmt, r"\1", normalized)

    # Handle single-type signatures like require(bool)(...)
    # Need to capture everything after the type signature, including nested parens
    pattern_single_type = rf"(\b(?:require|revert|assert)\b)\({type_keywords}\)"
    # First, just remove the type signature part
    normalized = re.sub(pattern_single_type, r"\1", normalized)

    # Step 2: Strip parameter names from function DECLARATIONS
    # Converts: constructor(uint256 amount, address to) -> constructor(uint256,address)
    # Only applies to patterns that look like type declarations, not expressions
    # A parameter declaration has the pattern: "type name" where type is a Solidity type

    def strip_param_names(match):
        """Helper to strip parameter names from function signature."""
        func_name = match.group(1)
        params_str = match.group(2)

        if not params_str.strip():
            return f"{func_name}()"

        # Split by comma, strip names from each parameter
        params = []
        for param in params_str.split(","):
            param = param.strip()
            if not param:
                continue

            # Check if this looks like a type declaration (starts with a Solidity type keyword)
            type_pattern = r"^(address|uint\d*|int\d*|bool|string|bytes\d*|byte|tuple|mapping)(\[\d*\])?\s+\w+"
            type_match = re.match(type_pattern, param)

            if type_match:
                # Has parameter name, extract just the type part
                # This is "type" or "type[]"
                type_with_array = type_match.group(0).split()[0]
                params.append(type_with_array)
            else:
                # Not a type declaration, keep as-is (might be an expression)
                params.append(param)

        return f"{func_name}({','.join(params)})"

    # Match function declarations/calls with parameters
    # This handles: funcName(type1 name1, type2 name2)
    pattern_param_names = r"(\w+)\(([^)]*)\)"

    # Only apply if we see parameter names that look like type declarations
    # More specific: must start with a type keyword (with optional array suffix) followed by a space and identifier
    if re.search(
        r"\((?:address|uint\d*|int\d*|bool|string|bytes\d*|byte)(?:\[\d*\])?\s+\w+",
        normalized,
    ):
        normalized = re.sub(pattern_param_names, strip_param_names, normalized)

    # Step 3: Remove all whitespace
    normalized = re.sub(r"\s+", "", normalized)

    return normalized


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (ValueError, TypeError, AttributeError):
        return None


# def _resolve_source_path(
#     project: Optional[str], relative_path: Optional[str]
# ) -> Optional[Path]:
#     if not relative_path:
#         return None
#     return (SOURCE_CODE_ROOT / relative_path).resolve()


class _NormalizedSource:
    __slots__ = (
        "normalized",
        "line_map",
        "raw_text",
        "simple_normalized",
        "simple_line_map",
    )

    def __init__(self, raw_text: str):
        """
        Create normalized source with enhanced normalization that strips:
        - Type signatures from function calls
        - Parameter names from function declarations
        - All whitespace

        Maintains line mapping for locating matches in original source.
        """
        self.raw_text = raw_text

        # First, apply the same normalization as _normalize_code_fragment
        # but we need to do it character-by-character to maintain line mapping

        # Strategy: Normalize the entire text first, then create char-to-line mapping
        # This is necessary because our normalization changes structure (removes type sigs, etc.)

        # Step 1: Simple whitespace-only normalization with line tracking
        # (We'll handle complex patterns differently)
        simple_normalized_chars: List[str] = []
        simple_line_map: List[int] = []
        current_line = 1

        for ch in raw_text:
            if ch == "\n":
                current_line += 1
                continue
            if ch.isspace():
                continue
            simple_normalized_chars.append(ch)
            simple_line_map.append(current_line)

        simple_normalized = "".join(simple_normalized_chars)

        # Step 2: Apply enhanced normalization to the simple normalized text
        # This creates the final normalized form used for matching
        enhanced_normalized = _normalize_code_fragment(simple_normalized)

        # Step 3: Create a mapping from enhanced positions to original lines
        # This is approximate but should work for most cases

        # For now, we'll use a simpler approach: store both versions
        # and use fuzzy matching if exact match fails
        self.normalized = enhanced_normalized
        self.line_map = simple_line_map
        self.simple_normalized = simple_normalized
        self.simple_line_map = simple_line_map

    @classmethod
    def from_path(cls, path: Path) -> Optional["_NormalizedSource"]:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"[WARN] Unable to read source file {path}: {exc}")
            return None
        return cls(text)

    def find_span(
        self,
        normalized_pattern: str,
        target_start: Optional[int],
        target_end: Optional[int],
    ) -> Optional[Tuple[int, int]]:
        """
        Find a normalized pattern in the source and return line span.

        Args:
            normalized_pattern: Already normalized search pattern (from CFG or snippet)
            target_start: Optional line number constraint (start)
            target_end: Optional line number constraint (end)

        Returns:
            Tuple of (start_line, end_line) if found, None otherwise
        """
        if not normalized_pattern:
            return None

        # Try to find in enhanced normalized version
        idx = self.normalized.find(normalized_pattern)

        if idx == -1:
            # If not found in enhanced, try simple normalized
            # This handles cases where the pattern itself wasn't enhanced
            idx = self.simple_normalized.find(normalized_pattern)
            if idx != -1 and idx < len(self.simple_line_map):
                start_line = self.simple_line_map[idx]
                end_idx = min(
                    idx + len(normalized_pattern) - 1, len(self.simple_line_map) - 1
                )
                end_line = self.simple_line_map[end_idx]

                if (
                    target_start is None
                    or target_end is None
                    or (start_line >= target_start and end_line <= target_end)
                ):
                    return start_line, end_line
            return None

        # For enhanced normalized, we need to map back approximately
        # Since enhanced removes characters (type signatures, etc.), we need to be careful

        # Strategy: Use the simple_normalized to find approximate location
        # by matching a substring that exists in both

        # For now, use a heuristic: find the pattern in simple_normalized
        # that's closest to our enhanced match position

        # Simple approach: ratio-based mapping
        if len(self.normalized) > 0:
            # Position ratio in enhanced text
            ratio = idx / len(self.normalized)
            # Approximate position in simple text
            approx_idx = int(ratio * len(self.simple_normalized))

            # Search around this position in simple_normalized
            search_radius = 100
            best_match_idx = -1

            # Try to find the pattern (or a simplified version) near the approx position
            for offset in range(-search_radius, search_radius):
                check_idx = approx_idx + offset
                if 0 <= check_idx < len(self.simple_normalized):
                    # Try substring match
                    if check_idx + len(normalized_pattern) <= len(
                        self.simple_normalized
                    ):
                        if (
                            self.simple_normalized[
                                check_idx : check_idx + len(normalized_pattern)
                            ]
                            == normalized_pattern
                        ):
                            best_match_idx = check_idx
                            break

            if best_match_idx != -1 and best_match_idx < len(self.simple_line_map):
                start_line = self.simple_line_map[best_match_idx]
                end_idx = min(
                    best_match_idx + len(normalized_pattern) - 1,
                    len(self.simple_line_map) - 1,
                )
                end_line = self.simple_line_map[end_idx]

                if (
                    target_start is None
                    or target_end is None
                    or (start_line >= target_start and end_line <= target_end)
                ):
                    return start_line, end_line

        return None


def extract_call_metadata(cpg: nx.DiGraph, cpg_index: Dict):
    """
    Scans CFG nodes for IR call instructions and updates cpg_index with structured metadata.
    Now records 'dest_variable' to track the specific contract instance variable being called.
    """
    # Regex 1: Static Dispatch (Internal, Library, Modifier)
    regex_static = re.compile(
        r"(INTERNAL|LIBRARY|MODIFIER)_CALL[^,]*,\s*([a-zA-Z0-9_$.]+)\.(.+)"
    )

    # Regex 2: Dynamic Dispatch (High Level)
    regex_dynamic = re.compile(
        r"HIGH_LEVEL_CALL[^,]*,\s*dest:(?P<dest>[^,]+),\s*function:(?P<func>[^,]+)(?:,\s*arguments:(?P<args>\[.*\]))?"
    )

    # Regex 3: Low Level Dispatch
    regex_low_level = re.compile(
        r"(LOW_LEVEL)_CALL[^,]*,\s*dest:(?P<dest>[^,]+),\s*function:(?P<func>[^,]+)(?:,\s*arguments:(?P<args>\[.*\]))?"
    )

    # Regex 4: Assignment (LHS extraction)
    regex_lhs = re.compile(r"^(?P<name>[^\(]+)(?:\((?P<type>[^\)]+)\))?$")

    # Regex 5: Destination Parsing (Extract "VAR" from "VAR(TYPE)")
    regex_dest = re.compile(r"^(?P<var>.+)\((?P<type>[^)]+)\)$")

    HANDLED_PREFIXES = ["INTERNAL", "LIBRARY", "MODIFIER", "HIGH_LEVEL", "LOW_LEVEL"]
    IGNORED_CALLS = ["SOLIDITY_CALL"]

    for node_id, data in cpg.nodes(data=True):
        if data.get(GraphAttributes.NODE_TYPE) != NodeTypes.CFG_NODE:
            continue

        ir_code = data.get(GraphAttributes.IR, "")
        if not ir_code:
            continue

        lines = ir_code.split("\n")

        caller_file = data.get(GraphAttributes.FILE, "")
        caller_contract = data.get(GraphAttributes.CONTRACT, "")
        caller_func_full = data.get(GraphAttributes.FUNCTION, "")

        f_name, f_args = clean_function_signature(caller_func_full)
        caller_key = f"{caller_file}{SEPARATOR}{caller_contract}{SEPARATOR}{f_name}{SEPARATOR}{f_args}"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # --- 1. Split Assignment (LHS) and Operation (RHS) ---
            if " = " in line:
                lhs_str, rhs_str = line.split(" = ", 1)
                lhs_str = lhs_str.strip()
                rhs_str = rhs_str.strip()
            else:
                lhs_str = None
                rhs_str = line

            ret_var_name = None
            ret_var_type = None
            if lhs_str:
                lhs_match = regex_lhs.match(lhs_str)
                if lhs_match:
                    name_group = (lhs_match.group("name") or "").strip()
                    ret_var_name = name_group or None
                    type_group = lhs_match.group("type")
                    if type_group:
                        ret_var_type = f"({type_group.strip()})"

            # --- 2. Parse Call (RHS) ---
            call_type = None
            target_contract = None
            dest_variable = None  # NEW FIELD

            target_func_name = None
            target_func_sig = None
            target_func_type_args = None
            call_passed_args = None

            # A. Static Dispatch
            match_static = regex_static.search(rhs_str)
            if match_static:
                call_type = match_static.group(1)
                target_contract = match_static.group(2)
                raw_full_part = match_static.group(3).strip()

                # Static calls don't have a "destination variable" (they are class-level)
                dest_variable = None

                # Parse Signature vs Args
                first_paren = raw_full_part.find("(")
                if first_paren != -1:
                    balance = 0
                    split_idx = -1
                    for i in range(first_paren, len(raw_full_part)):
                        char = raw_full_part[i]
                        if char == "(":
                            balance += 1
                        elif char == ")":
                            balance -= 1
                        if balance == 0:
                            split_idx = i
                            break

                    if split_idx != -1:
                        target_func_sig = raw_full_part[: split_idx + 1]
                        target_func_type_args = raw_full_part[
                            first_paren : split_idx + 1
                        ]
                        target_func_name = raw_full_part[:first_paren]
                        if split_idx + 1 < len(raw_full_part):
                            call_passed_args = raw_full_part[split_idx + 1 :].strip()
                        else:
                            call_passed_args = "()"
                    else:
                        target_func_sig = raw_full_part
                        target_func_name = raw_full_part
                        target_func_type_args = "()"
                        call_passed_args = "()"
                else:
                    target_func_sig = raw_full_part
                    target_func_name = raw_full_part
                    target_func_type_args = "()"
                    call_passed_args = "()"

            # B. Dynamic Dispatch (High Level)
            elif "HIGH_LEVEL_CALL" in rhs_str:
                match_dynamic = regex_dynamic.search(rhs_str)
                if match_dynamic:
                    call_type = "HIGH_LEVEL"
                    raw_dest = match_dynamic.group("dest").strip()
                    target_func_sig = match_dynamic.group("func").strip()
                    target_func_name = target_func_sig
                    target_func_type_args = ""

                    # Convert List "['a', 'b']" to Tuple "(a, b)"
                    raw_args = match_dynamic.group("args")
                    if raw_args:
                        clean_content = (
                            raw_args.strip("[]")
                            .replace(" ", "")
                            .replace("'", "")
                            .replace('"', "")
                        )
                        call_passed_args = f"({clean_content})"
                    else:
                        call_passed_args = "()"

                    dest_match = regex_dest.match(raw_dest)
                    if dest_match:
                        dest_variable = dest_match.group("var").strip()  # TMP_174
                        target_contract = dest_match.group("type").strip()  # IERC20
                    else:
                        # Fallback for untyped vars (e.g. "myVar")
                        dest_variable = raw_dest
                        target_contract = raw_dest

            # C. Low Level Dispatch
            elif "LOW_LEVEL_CALL" in rhs_str:
                match_low_level = regex_low_level.search(rhs_str)
                if match_low_level:
                    call_type = "LOW_LEVEL"
                    print("LOW LEVEL CALL DETECTED:", caller_key)
                    target_func_sig = match_low_level.group("func").strip()
                    target_func_name = target_func_sig
                    target_func_type_args = ""

                    raw_args = match_low_level.group("args")
                    if raw_args:
                        clean_content = (
                            raw_args.strip("[]").replace("'", "").replace('"', "")
                        )
                        call_passed_args = f"({clean_content})"
                    else:
                        call_passed_args = "()"

                    # Low level usually has raw dest like "addr"
                    dest_variable = match_low_level.group("dest").strip()
                    target_contract = "ADDRESS_CALL"

            # Diagnostic
            if not call_type:
                match_call_prefix = re.search(r"([A-Z_]+)_CALL", rhs_str)
                if match_call_prefix:
                    prefix = match_call_prefix.group(1)
                    if (
                        prefix not in HANDLED_PREFIXES
                        and f"{prefix}_CALL" not in IGNORED_CALLS
                    ):
                        print(f"[UNRECORDED CALL TYPE] {prefix}_CALL in: {line}")
                continue

            # --- 3. Save Metadata ---
            ir_hash = hashlib.md5(line.encode("utf-8")).hexdigest()
            unique_call_id = f"{caller_key}{SEPARATOR}{node_id}{SEPARATOR}{ir_hash}"

            call_site_info = {
                "key": caller_key,
                "cfg_id": node_id,
                "call_type": call_type,
                "file": caller_file,
                "contract": caller_contract,
                "function": f_name,
                "function_args": f_args,
                # Target Info
                "target_contract": target_contract,
                "dest_variable": dest_variable,  # <--- NEW FIELD
                "target_func_signature": target_func_sig,
                "target_func_name": target_func_name,
                "target_func_args": target_func_type_args,
                "call_arguments": call_passed_args,
                # Return Info
                "return_var_name": ret_var_name,
                "return_var_type": ret_var_type,
                "ir_raw": line,
            }

            cpg_index[unique_call_id] = call_site_info

    return cpg_index


def link_calls_by_attributes(cpg: nx.DiGraph, cpg_index: Dict):
    """
    Connects CFG Call Sites to their Target Function Entry Nodes.
    """
    print("[*] Linking Calls using Attribute Index...")
    edges_to_add = []

    # ---------------------------------------------------------
    # 1. Build Target Lookups (Definitions)
    # ---------------------------------------------------------
    target_registry = defaultdict(list)
    return_registry = defaultdict(list)

    # A. Register Definitions
    for key, data in cpg_index.items():
        if "ast_id" in data and "target_contract" not in data:
            contract = data.get("contract")
            func_name = data.get("function")
            func_args = data.get("function_args")
            entry_node = data.get("cfg_id")
            if contract and func_name and entry_node:
                target_registry[(contract, func_name)].append(
                    {"args": func_args, "cfg_id": entry_node, "full_key": key}
                )

    # B. Register Return Nodes
    for node, data in cpg.nodes(data=True):
        if (
            data.get(GraphAttributes.NODE_TYPE) == NodeTypes.CFG_NODE
            and data.get(GraphAttributes.SUB_NODE_TYPE) == "return"
        ):
            c = data.get(GraphAttributes.CONTRACT)
            f = data.get(GraphAttributes.FUNCTION)
            if c and f:
                f_clean = f.split("(")[0]
                return_registry[(c, f_clean)].append(node)

    # ---------------------------------------------------------
    # 2. Process Call Sites
    # ---------------------------------------------------------
    links_created = 0

    for key, data in cpg_index.items():
        if "target_contract" in data:

            src_node = data["cfg_id"]
            # Capture the call type from the SITE
            current_call_type = data["call_type"]

            tgt_contract = data["target_contract"]
            tgt_func_name = data["target_func_name"]
            tgt_type_args = data.get("target_func_args", "")
            call_arg_count = data.get("call_arg_count")
            returns_value = bool(data.get("return_var_name"))
            dest_variable = data.get("dest_variable")

            if tgt_contract == "ADDRESS_CALL":
                continue

            # --- Find Candidates ---
            potential_targets = target_registry.get((tgt_contract, tgt_func_name), [])
            final_targets = []

            if potential_targets:
                if tgt_type_args and tgt_type_args != "()":
                    for candidate in potential_targets:
                        if candidate["args"] == tgt_type_args:
                            final_targets.append(candidate["cfg_id"])

                if not final_targets:
                    for candidate in potential_targets:
                        final_targets.append(candidate["cfg_id"])

            # --- Create Edges ---
            if final_targets:
                links_created += 1

                for dest_node in final_targets:
                    # 1. Forward Edge: Call Site -> Function Entry
                    call_edge_attr = {
                        GraphAttributes.LABEL: current_call_type,
                        GraphAttributes.EDGE_TYPE: EdgeTypes.CALL,
                        GraphAttributes.SUB_EDGE_TYPE: current_call_type,
                        GraphAttributes.CALL_SITE_ID: key,
                        GraphAttributes.CALL_RETURNS_VALUE: returns_value,
                    }
                    if dest_variable:
                        call_edge_attr[GraphAttributes.CALL_DEST] = dest_variable
                    if call_arg_count is not None:
                        call_edge_attr[GraphAttributes.CALL_ARG_COUNT] = call_arg_count
                    edges_to_add.append((src_node, dest_node, call_edge_attr))

                    # 2. Backward Edge: Return Node -> Call Site Successor
                    successors = list(cpg.successors(src_node))
                    if successors:
                        target_returns = return_registry.get(
                            (tgt_contract, tgt_func_name), []
                        )
                        if target_returns:
                            for ret_node in target_returns:
                                for succ in successors:
                                    ret_edge_attr = {
                                        GraphAttributes.LABEL: "return",
                                        GraphAttributes.EDGE_TYPE: EdgeTypes.RETURN_CALL,
                                        GraphAttributes.SUB_EDGE_TYPE: current_call_type,
                                        GraphAttributes.CALL_SITE_ID: key,
                                        GraphAttributes.CALL_RETURNS_VALUE: returns_value,
                                    }
                                    if call_arg_count is not None:
                                        ret_edge_attr[
                                            GraphAttributes.CALL_ARG_COUNT
                                        ] = call_arg_count
                                    edges_to_add.append((ret_node, succ, ret_edge_attr))

    for u, v, attr in edges_to_add:
        cpg.add_edge(u, v, **attr)

    print(f"[*] Call Linking Complete.")
    print(f" - Linked {links_created} call sites.")
    print(f" - Added {len(edges_to_add)} inter-procedural edges.")

    return cpg


def _parse_entry_edges(entry: dict) -> List[Dict[str, Any]]:
    decoded: List[Dict[str, Any]] = []
    raw_edges = entry.get("edges", []) or []
    for edge_item in raw_edges:
        if isinstance(edge_item, str):
            parts = edge_item.split(SEPARATOR)
            while len(parts) < 5:
                parts.append("")
            decoded.append(
                {
                    "from": parts[0],
                    "to": parts[1],
                    "to_kind": parts[2],
                    "kind": parts[3],
                    "operand_index": parts[4] or None,
                }
            )
        elif isinstance(edge_item, dict):
            decoded.append(
                {
                    "from": edge_item.get("from") or edge_item.get("source") or "",
                    "to": edge_item.get("to") or "",
                    "to_kind": edge_item.get("to_kind") or edge_item.get("target_kind"),
                    "kind": edge_item.get("kind") or edge_item.get("edge_type"),
                    "operand_index": edge_item.get("operand_index"),
                }
            )
    return decoded


def _extract_defs_uses_from_entry(entry: dict) -> Tuple[Set[str], Set[str]]:
    defs: Set[str] = set()
    uses: Set[str] = set()

    for edge in _parse_entry_edges(entry):
        src = (edge.get("from") or "").strip()
        tgt = (edge.get("to") or "").strip()
        to_kind = (edge.get("to_kind") or "").lower()
        kind = (edge.get("kind") or "").lower()

        if to_kind == "var" and kind == "assign" and tgt:
            defs.add(tgt)

        if to_kind == "node" and src and not src.startswith("n"):
            uses.add(src)

    return defs, uses


def add_data_flow_edges(
    cpg: nx.DiGraph,
    cpg_index: Dict,
    cfg_index: Dict,
    ir_index: Dict,
):
    """Inject data flow edges using the precomputed IR JSON catalog."""

    print("[*] Generating Data Flow edges from IR catalog...")

    total_ir_entries = len(ir_index)
    skipped_no_cfg = 0
    skipped_no_defs_uses = 0
    missing_match_counts: Dict[str, int] = defaultdict(int)
    missing_match_examples: List[str] = []
    empty_ir_examples: List[str] = []

    def _build_keys(raw_key: str) -> Tuple[Optional[str], Optional[str]]:
        parts = raw_key.split(SEPARATOR)
        if len(parts) < 6:
            return None, None
        func_key = SEPARATOR.join(parts[:4])
        match_key = SEPARATOR.join(parts[:4] + [parts[5]])
        return func_key, match_key

    cfg_lookup: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for key, entry in cfg_index.items():
        func_key, match_key = _build_keys(key)
        if not match_key:
            continue
        cpg_node_id = entry.get("cpg_node_id")
        if cpg_node_id is None or not cpg.has_node(cpg_node_id):
            continue
        node_data = cpg.nodes[cpg_node_id]
        if node_data.get(GraphAttributes.NODE_TYPE) != NodeTypes.CFG_NODE:
            continue
        cfg_lookup[match_key].append(
            {
                "function_key": func_key,
                "cpg_node_id": cpg_node_id,
            }
        )

    grouped_entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ir_key, entry in ir_index.items():
        func_key, match_key = _build_keys(ir_key)
        if not match_key or func_key is None:
            continue
        cfg_candidates = cfg_lookup.get(match_key)
        if not cfg_candidates:
            skipped_no_cfg += 1
            missing_match_counts[func_key] += 1
            if len(missing_match_examples) < 5:
                expr_preview = (
                    entry.get("expression") or entry.get("ir") or ""
                ).splitlines()
                snippet = expr_preview[0] if expr_preview else ""
                snippet = snippet[:80]
                missing_match_examples.append(f"{func_key} :: {snippet}")
            continue

        defs, uses = _extract_defs_uses_from_entry(entry)
        if not defs and not uses:
            skipped_no_defs_uses += 1
            if len(empty_ir_examples) < 5:
                expr_preview = (
                    entry.get("expression") or entry.get("ir") or ""
                ).splitlines()
                snippet = expr_preview[0] if expr_preview else ""
                snippet = snippet[:80]
                empty_ir_examples.append(f"{func_key} :: {snippet}")
            continue

        entry_order = entry.get("order", 0)
        defs_list = sorted(defs)
        uses_list = sorted(uses)
        for cfg_info in cfg_candidates:
            grouped_entries[func_key].append(
                {
                    "order": entry_order,
                    "node_id": cfg_info["cpg_node_id"],
                    "defs": defs_list,
                    "uses": uses_list,
                }
            )

    edge_cache: Set[Tuple[int, int, str]] = set()
    edges_added = 0
    functions_covered = 0

    for func_key, nodes in grouped_entries.items():
        if not nodes:
            continue
        functions_covered += 1
        definitions: Dict[str, List[int]] = defaultdict(list)
        for node in sorted(nodes, key=lambda item: item["order"]):
            node_id = node["node_id"]
            if not cpg.has_node(node_id):
                continue

            uses = node["uses"]
            defs = node["defs"]

            for var in uses:
                if not var:
                    continue
                for def_node in definitions.get(var, []):
                    if def_node == node_id:
                        continue
                    edge_key = (def_node, node_id, var)
                    if edge_key in edge_cache:
                        continue
                    edge_cache.add(edge_key)
                    cpg.add_edge(
                        def_node,
                        node_id,
                        **{
                            GraphAttributes.LABEL: f"data_flow_{var}",
                            GraphAttributes.EDGE_TYPE: EdgeTypes.DF,
                            GraphAttributes.VAR: var,
                        },
                    )
                    edges_added += 1

            for var in defs:
                if not var:
                    continue
                tracked_defs = definitions[var]
                if node_id not in tracked_defs:
                    tracked_defs.append(node_id)

    print(f" - Processed {functions_covered} functions with IR coverage.")
    print(f" - Added {edges_added} Data Flow edges.")
    skipped_total = skipped_no_cfg + skipped_no_defs_uses
    if skipped_total:
        print(
            f" - Skipped {skipped_total} of {total_ir_entries} IR entries (no CFG match or empty def/use set)."
        )
    if skipped_no_cfg:
        unique_missing = len(missing_match_counts)
        samples = ", ".join(missing_match_examples)
        print(
            f"   - {skipped_no_cfg} entries missing CFG nodes across {unique_missing} functions. Samples: {samples}"
        )
    if skipped_no_defs_uses:
        samples = ", ".join(empty_ir_examples)
        print(
            f"   - {skipped_no_defs_uses} entries without def/use data. Samples: {samples}"
        )
    return cpg


def link_ast_to_cfg(cpg, cfg_graph, ast_graph, ast_func_index, cfg_function_index):
    """
    Merges AST and CFG into the CPG with new unique IDs and links them at both:
    1. Function definition level (AST FunctionDefinition -> CFG Entry)
    2. Expression/Statement level (AST expressions -> CFG blocks with matching expressions)
    """
    print("Transferring nodes to CPG and linking...")
    cpg_index = {}
    # 1. ID Management
    # Start counter after any existing nodes in CPG (safety check)
    if cpg.number_of_nodes() > 0:
        next_id = max(cpg.nodes()) + 1
    else:
        next_id = 1

    # Maps to track Old_ID -> New_CPG_ID
    ast_id_map = {}
    cfg_id_map = {}

    # ---------------------------------------------------------
    # 2. Transfer AST Graph (Nodes & Edges)
    # ---------------------------------------------------------
    for node, data in ast_graph.nodes(data=True):
        new_id = next_id
        next_id += 1
        ast_id_map[node] = new_id
        # Add to CPG (Preserve all attributes)
        cpg.add_node(new_id, **data)

    for u, v, data in ast_graph.edges(data=True):
        if u in ast_id_map and v in ast_id_map:
            cpg.add_edge(ast_id_map[u], ast_id_map[v], **data)

    # ---------------------------------------------------------
    # 3. Transfer CFG Graph (Nodes & Edges)
    # ---------------------------------------------------------
    for node, data in cfg_graph.nodes(data=True):
        new_id = next_id
        next_id += 1
        cfg_id_map[node] = new_id

        # Add to CPG
        cpg.add_node(new_id, **data)

    for u, v, data in cfg_graph.edges(data=True):
        if u in cfg_id_map and v in cfg_id_map:
            cpg.add_edge(cfg_id_map[u], cfg_id_map[v], **data)

    # Persist the remapped CPG node IDs on the CFG index so downstream passes
    # (e.g., data-flow linking) can resolve the actual node IDs inside the CPG.
    for cfg_entry in cfg_function_index.values():
        original_id = cfg_entry.get("node_id")
        if not original_id:
            continue
        mapped_id = cfg_id_map.get(original_id)
        if mapped_id is not None:
            cfg_entry["cpg_node_id"] = mapped_id

    # ---------------------------------------------------------
    # 4. Create Links (AST -> CFG)
    # ---------------------------------------------------------
    
    # 4A. Function-Level Links (AST FunctionDefinition -> CFG Entry)
    function_links_created = 0
    # Iterate over original CFG nodes to find entry points
    for cfg_node, cfg_data in cfg_graph.nodes(data=True):

        # Filter for Entry Points (In-Degree 0)
        if cfg_graph.in_degree(cfg_node) == 0:
            file = cfg_data.get(
                "file", ""
            )  # Using string literal or GraphAttributes.FILE
            contract = cfg_data.get("contract", "")
            function_full = cfg_data.get("function", "")

            # Generate Key
            function_name, function_args = clean_function_signature(function_full)
            key = f"{file}{SEPARATOR}{contract}{SEPARATOR}{function_name}{SEPARATOR}{function_args}"
            # Lookup in Index (Index contains original AST IDs)
            original_ast_id = ast_func_index.get(key)

            if original_ast_id:
                # Resolve to New CPG IDs
                # We need to ensure the original AST ID exists in our map
                # (It should, if ast_graph contains all nodes from index)
                if original_ast_id in ast_id_map and cfg_node in cfg_id_map:

                    cpg_ast_id = ast_id_map[original_ast_id]
                    cpg_cfg_id = cfg_id_map[cfg_node]

                    # Add the Inter-Graph Edge
                    cpg.add_edge(
                        cpg_ast_id,
                        cpg_cfg_id,
                        **{
                            GraphAttributes.EDGE_TYPE: EdgeTypes.AST_TO_CFG,
                            GraphAttributes.LABEL: "defines",
                            GraphAttributes.SUB_EDGE_TYPE: "function_definition",
                        },
                    )
                    function_links_created += 1
                    cpg_index[key] = {
                        "key": key,
                        "ast_id": cpg_ast_id,
                        "cfg_id": cpg_cfg_id,
                        "file": file,
                        "contract": contract,
                        "function": function_name,
                        "function_args": function_args,
                        "function_full": function_full,
                    }
            else:
                # Debugging for missing links
                # print(f"Missed Link: {key}")
                pass
    
    # 4B. Expression-Level Links (AST Expression/Statement -> CFG Block)
    expression_links_created = 0
    
    # Build lookup for CFG nodes by normalized expression
    cfg_expression_lookup = defaultdict(list)
    for old_cfg_id, cfg_data in cfg_graph.nodes(data=True):
        if old_cfg_id not in cfg_id_map:
            continue
        cfg_expr = (cfg_data.get(GraphAttributes.EXPRESSION) or "").strip()
        if cfg_expr:
            normalized_expr = _normalize_code_fragment(cfg_expr)
            if normalized_expr:
                cfg_expression_lookup[normalized_expr].append({
                    "cpg_id": cfg_id_map[old_cfg_id],
                    "contract": cfg_data.get(GraphAttributes.CONTRACT, ""),
                    "function": cfg_data.get(GraphAttributes.FUNCTION, ""),
                    "file": cfg_data.get(GraphAttributes.FILE, ""),
                })
    
    # Match AST expression/statement nodes to CFG blocks
    for old_ast_id, ast_data in ast_graph.nodes(data=True):
        if old_ast_id not in ast_id_map:
            continue
        
        ast_sub_type = (ast_data.get(GraphAttributes.SUB_NODE_TYPE) or "").lower()
        
        # Only link expression/statement level nodes (not containers like FunctionDefinition, ContractDefinition)
        expression_types = [
            "expressionstatement", "variabledeclarationstatement", "return", 
            "emitstatement", "revertstatement", "ifstatement", "forstatement",
            "whilestatement", "assignment", "functioncall", "binaryoperation",
            "unaryoperation", "memberaccess", "identifier"
        ]
        
        if ast_sub_type not in expression_types:
            continue
        
        ast_label = (ast_data.get(GraphAttributes.LABEL) or "").strip()
        if not ast_label:
            continue
        
        normalized_ast_expr = _normalize_code_fragment(ast_label)
        if not normalized_ast_expr:
            continue
        
        # Get context for matching
        ast_contract = ast_data.get(GraphAttributes.CONTRACT, "")
        ast_function = ast_data.get(GraphAttributes.FUNCTION, "")
        ast_file = ast_data.get(GraphAttributes.FILE, "")
        
        # Find matching CFG nodes
        cfg_candidates = cfg_expression_lookup.get(normalized_ast_expr, [])
        
        for cfg_candidate in cfg_candidates:
            # Match by context (same contract, function, file)
            if (cfg_candidate["contract"] == ast_contract and
                cfg_candidate["function"] == ast_function and
                cfg_candidate["file"] == ast_file):
                
                cpg_ast_id = ast_id_map[old_ast_id]
                cpg_cfg_id = cfg_candidate["cpg_id"]
                
                # Add bidirectional edges for expression-level mapping
                cpg.add_edge(
                    cpg_ast_id,
                    cpg_cfg_id,
                    **{
                        GraphAttributes.EDGE_TYPE: EdgeTypes.AST_TO_CFG,
                        GraphAttributes.LABEL: "expression_maps_to",
                        GraphAttributes.SUB_EDGE_TYPE: "expression",
                    },
                )
                expression_links_created += 1
                break  # Only link to first match
    
    print(f" - Total AST-CFG Function Links Created: {function_links_created}")
    print(f" - Total AST-CFG Expression Links Created: {expression_links_created}")
    return cpg_index


def propagate_vulnerabilities_via_ast_cfg_edges(cpg: nx.DiGraph, vuln_labels_map: Dict[str, Dict[str, Any]]):
    """
    Propagate vulnerabilities bidirectionally through AST-to-CFG edges.
    If an AST node is vulnerable, propagate to its linked CFG nodes.
    If a CFG node is vulnerable, propagate to its linked AST nodes.
    """
    print("\\n[*] Propagating vulnerabilities via AST-CFG edges...")
    
    # Build initial vulnerability sets
    vuln_ast_nodes = set()
    vuln_cfg_nodes = set()
    
    for vuln_entry in vuln_labels_map.values():
        vuln_ast_nodes.update(vuln_entry.get("ast_nodes", {}).keys())
        vuln_cfg_nodes.update(vuln_entry.get("nodes", {}).keys())
    
    print(f"  Initial vulnerable nodes: {len(vuln_ast_nodes)} AST, {len(vuln_cfg_nodes)} CFG")
    
    propagation_count = 0
    
    # Iterate through all edges to find AST-to-CFG connections
    for u, v, edge_data in cpg.edges(data=True):
        edge_type = edge_data.get(GraphAttributes.EDGE_TYPE)
        
        if edge_type != EdgeTypes.AST_TO_CFG:
            continue
        
        u_data = cpg.nodes[u]
        v_data = cpg.nodes[v]
        
        u_type = u_data.get(GraphAttributes.NODE_TYPE)
        v_type = v_data.get(GraphAttributes.NODE_TYPE)
        
        # AST -> CFG edge
        if (u_type == NodeTypes.AST_NODE and v_type == NodeTypes.CFG_NODE):
            u_str = str(u)
            v_str = str(v)
            
            # If AST node is vulnerable, propagate to CFG
            if u_str in vuln_ast_nodes:
                for vuln_entry in vuln_labels_map.values():
                    if u_str in vuln_entry.get("ast_nodes", {}):
                        owasp_id = vuln_entry.get("owasp_id")
                        
                        # Add to CFG nodes
                        if "nodes" not in vuln_entry:
                            vuln_entry["nodes"] = {}
                        
                        if v_str not in vuln_entry["nodes"]:
                            # Copy metadata from AST node
                            ast_node_data = vuln_entry["ast_nodes"][u_str]
                            vuln_entry["nodes"][v_str] = {
                                "expression": v_data.get(GraphAttributes.EXPRESSION, ""),
                                "ir": v_data.get(GraphAttributes.IR, ""),
                                "node_type": NodeTypes.CFG_NODE,
                                "sub_node_type": v_data.get(GraphAttributes.SUB_NODE_TYPE, ""),
                                "propagated_from": "AST",
                            }
                            vuln_cfg_nodes.add(v_str)
                            propagation_count += 1
            
            # If CFG node is vulnerable, propagate to AST
            if v_str in vuln_cfg_nodes:
                for vuln_entry in vuln_labels_map.values():
                    if v_str in vuln_entry.get("nodes", {}):
                        owasp_id = vuln_entry.get("owasp_id")
                        
                        # Add to AST nodes
                        if "ast_nodes" not in vuln_entry:
                            vuln_entry["ast_nodes"] = {}
                        
                        if u_str not in vuln_entry["ast_nodes"]:
                            vuln_entry["ast_nodes"][u_str] = {
                                "expression": u_data.get(GraphAttributes.LABEL, ""),
                                "node_type": NodeTypes.AST_NODE,
                                "sub_node_type": u_data.get(GraphAttributes.SUB_NODE_TYPE, ""),
                                "propagated_from": "CFG",
                            }
                            vuln_ast_nodes.add(u_str)
                            propagation_count += 1
    
    print(f"  Propagated {propagation_count} vulnerabilities via AST-CFG edges")
    print(f"  Final vulnerable nodes: {len(vuln_ast_nodes)} AST, {len(vuln_cfg_nodes)} CFG")
    
    return vuln_labels_map


def annotate_nodes_with_vulnerabilities(cpg: nx.DiGraph, line_vuln):
    if not line_vuln:
        print("[*] No vulnerability metadata provided; skipping node mapping.")
        return {}

    print("[*] Mapping line-level vulnerabilities to CFG nodes...")

    working_entries: Dict[str, Dict[str, Any]] = {}
    lookup: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    source_cache: Dict[Path, Optional[_NormalizedSource]] = {}

    for key, entry in line_vuln.items():
        if not isinstance(entry, dict):
            continue
        entry_copy = dict(entry)
        base_file = Path(entry_copy.get("file", "")).name
        func_name = (entry_copy.get("function") or "").strip()
        project = entry_copy.get("project")
        rel_path = entry_copy.get("file_full")  # Use full path for source lookup
        abs_source_path = Path(entry_copy.get("original_filename"))
        line_start = _coerce_int(entry_copy.get("line_from"))
        line_end = _coerce_int(entry_copy.get("line_to"))
        if line_start is not None and line_end is not None and line_start > line_end:
            line_start, line_end = line_end, line_start
        snippet_norm = _normalize_code_fragment(entry_copy.get("snippet"))

        working_entries[key] = {
            "meta": entry_copy,
            "file_base": base_file,
            "function_name": func_name,
            "line_from": line_start,
            "line_to": line_end,
            "snippet_norm": snippet_norm,
            "abs_path": abs_source_path,
            "cfg_nodes": {},
            "ast_nodes": {},
            "contract": entry_copy.get("contract"),
            "function_args": entry_copy.get("function_args"),
            "function_full": entry_copy.get("function_full"),
        }

        if base_file and func_name:
            lookup[(base_file, func_name)].append(key)
            # Debug: Track what we're looking for
            # print(f"[DEBUG] Vuln lookup registered: ({base_file}, {func_name})")

    def get_source(path: Optional[Path]) -> Optional[_NormalizedSource]:
        if not path:
            return None
        cached = source_cache.get(path)
        if cached is not None:
            return cached
        if not path.exists():
            print(f"[WARN] Source file not found for vulnerability mapping: {path}")
            source_cache[path] = None
            return None
        source_cache[path] = _NormalizedSource.from_path(path)
        return source_cache[path]

    total_cfg_nodes = 0
    total_ast_nodes = 0
    
    # Collect unique file/function pairs from CFG for debugging
    cfg_file_func_pairs = set()
    for node_id, data in cpg.nodes(data=True):
        node_type = data.get(GraphAttributes.NODE_TYPE)
        if node_type == NodeTypes.CFG_NODE:
            node_file = Path(str(data.get(GraphAttributes.FILE) or "")).name
            func_full = data.get(GraphAttributes.FUNCTION, "") or ""
            func_name, _ = clean_function_signature(func_full)
            if node_file and func_name:
                cfg_file_func_pairs.add((node_file, func_name))
    
    print(f" - Found {len(cfg_file_func_pairs)} unique (file, function) pairs in CFG")
    print(f" - Looking to match {len(lookup)} unique (file, function) pairs from vulnerabilities")
    
    # Show sample of what we're looking for vs what exists
    vuln_pairs = set(lookup.keys())
    matched_pairs = vuln_pairs & cfg_file_func_pairs
    unmatched_pairs = vuln_pairs - cfg_file_func_pairs
    
    if matched_pairs:
        print(f" - {len(matched_pairs)} (file, function) pairs will match")
        sample = list(matched_pairs)[:3]
        for pair in sample:
            print(f"   ✓ {pair}")
    
    if unmatched_pairs:
        print(f" - [WARN] {len(unmatched_pairs)} (file, function) pairs have NO matching CFG nodes:")
        sample = list(unmatched_pairs)[:5]
        for pair in sample:
            print(f"   ✗ {pair}")

    for node_id, data in cpg.nodes(data=True):
        node_type = data.get(GraphAttributes.NODE_TYPE)
        if node_type not in (NodeTypes.CFG_NODE, NodeTypes.AST_NODE):
            continue

        node_file = Path(str(data.get(GraphAttributes.FILE) or "")).name
        func_full = data.get(GraphAttributes.FUNCTION, "") or ""
        func_name, func_args = clean_function_signature(func_full)
        matching_keys = lookup.get((node_file, func_name))
        if not matching_keys:
            continue

        expression = (
            data.get(GraphAttributes.EXPRESSION)
            or data.get(GraphAttributes.LABEL)
            or ""
        ).strip()
        ir_text = (data.get(GraphAttributes.IR) or "").strip()
        normalized_expression = _normalize_code_fragment(expression)
        if not normalized_expression:
            continue

        for vuln_key in matching_keys:
            record = working_entries.get(vuln_key)
            if not record:
                continue

            source_obj = get_source(record["abs_path"])
            line_span = None
            in_scope = False

            if source_obj:
                line_span = source_obj.find_span(
                    normalized_expression,
                    record["line_from"],
                    record["line_to"],
                )
                if line_span:
                    in_scope = True

            if not in_scope and record["snippet_norm"]:
                if normalized_expression in record["snippet_norm"]:
                    in_scope = True

            if (
                not in_scope
                and record["line_from"] is None
                and record["line_to"] is None
            ):
                in_scope = True

            if not in_scope:
                continue

            node_entry: Dict[str, Any] = {
                "expression": expression,
                "ir": ir_text,
                "node_type": node_type,
                "sub_node_type": data.get(GraphAttributes.SUB_NODE_TYPE, ""),
            }
            if line_span:
                node_entry["line_from"] = line_span[0]
                node_entry["line_to"] = line_span[1]

            bucket = "cfg_nodes" if node_type == NodeTypes.CFG_NODE else "ast_nodes"
            record[bucket][str(node_id)] = node_entry
            record["contract"] = record["contract"] or data.get(
                GraphAttributes.CONTRACT, ""
            )
            record["function_args"] = record["function_args"] or func_args
            record["function_full"] = record["function_full"] or func_full
            if node_type == NodeTypes.CFG_NODE:
                total_cfg_nodes += 1
            else:
                total_ast_nodes += 1

    results: Dict[str, Dict[str, Any]] = {}
    populated_entries = 0
    for key, record in working_entries.items():
        meta = record["meta"].copy()
        meta["contract"] = record["contract"]
        meta["function_args"] = record.get("function_args")
        if record.get("function_full"):
            meta["function_full"] = record["function_full"]
        meta["nodes"] = record["cfg_nodes"]
        meta["ast_nodes"] = record["ast_nodes"]
        results[key] = meta
        if record["cfg_nodes"] or record["ast_nodes"]:
            populated_entries += 1

    # output_path = Path("test_output/vuln_node_mappings.json")
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # with output_path.open("w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2)

    print(
        f" - Populated {populated_entries}/{len(results)} vulnerabilities with CFG/AST nodes (CFG: {total_cfg_nodes}, AST: {total_ast_nodes})."
    )
    # print(f" - Mapping written to {output_path}")
    
    # Propagate vulnerabilities through AST-CFG edges
    results = propagate_vulnerabilities_via_ast_cfg_edges(cpg, results)

    return results


def _count_vuln_node_matches(vuln_labels_map: Dict[str, Dict[str, Any]]) -> int:
    total = 0
    for entry in vuln_labels_map.values():
        cfg_nodes = entry.get("nodes") or {}
        ast_nodes = entry.get("ast_nodes") or {}
        total += len(cfg_nodes) + len(ast_nodes)
    return total


def color_cpg(cpg: nx.DiGraph, vuln_labels_map: Dict[str, Dict[str, Any]]) -> nx.DiGraph:
    """Add color attributes to CPG nodes and edges for visualization."""
    colored_cpg = cpg.copy()
    vuln_node_ids = set()
    for entry in vuln_labels_map.values():
        vuln_node_ids.update(entry.get("nodes", {}).keys())
        vuln_node_ids.update(entry.get("ast_nodes", {}).keys())

    node_colors = {
        NodeTypes.AST_NODE: 'darkorange',
        NodeTypes.CFG_NODE: 'darkblue',
    }

    node_fill_colors = {
        NodeTypes.AST_NODE: 'orange',
        NodeTypes.CFG_NODE: 'lightblue',
    }

    edge_colors = {
        EdgeTypes.CF: 'blue',
        EdgeTypes.CF_TRUE: 'darkblue',
        EdgeTypes.CF_FALSE: 'navy',
        EdgeTypes.LOOP_EXIT: 'royalblue',
        EdgeTypes.LOOP_CONTINUE: 'mediumblue',
        EdgeTypes.LOOP_BACK: 'blue',
        EdgeTypes.RETURN: 'midnightblue',
        EdgeTypes.CALL: 'darkgreen',
        EdgeTypes.DF: 'saddlebrown',
        EdgeTypes.DEF_USE: 'chocolate',
        EdgeTypes.DF_INIT: 'sienna',
        EdgeTypes.DF_ASSIGN: 'peru',
        EdgeTypes.DF_CALL_ARG: 'burlywood',
        EdgeTypes.DF_RETURN: 'tan',
        EdgeTypes.DF_MEMBER: 'rosybrown',
        EdgeTypes.DF_INDEX_BASE: 'sandybrown',
        EdgeTypes.DF_INDEX_EXPR: 'goldenrod',
        EdgeTypes.PARAMETER_IN: 'wheat',
        EdgeTypes.PARAMETER_OUT: 'moccasin',
        EdgeTypes.AST_CHILD: 'darkorange',
        EdgeTypes.RETURN_CALL: 'purple',
        EdgeTypes.AST_TO_CFG: 'darkviolet',
        EdgeTypes.AST_TO_CFG_STRUCTURAL: 'indigo',
        EdgeTypes.AST_CALL: 'magenta',
    }

    for node, data in colored_cpg.nodes(data=True):
        if str(node) in vuln_node_ids:
            color = 'red'
            fillcolor = 'lightcoral'
        else:
            node_type = data.get(GraphAttributes.NODE_TYPE)
            color = node_colors.get(node_type, 'black')
            fillcolor = node_fill_colors.get(node_type, 'white')
        colored_cpg.nodes[node]['color'] = color
        colored_cpg.nodes[node]['fillcolor'] = fillcolor
        colored_cpg.nodes[node]['style'] = 'filled'

    for u, v, data in colored_cpg.edges(data=True):
        if str(u) in vuln_node_ids or str(v) in vuln_node_ids:
            color = 'red'
        else:
            sub_edge_type = data.get(GraphAttributes.SUB_EDGE_TYPE)
            edge_type_attr = data.get(GraphAttributes.EDGE_TYPE)
            if sub_edge_type and sub_edge_type in edge_colors:
                edge_type = sub_edge_type
            else:
                edge_type = edge_type_attr
            color = edge_colors.get(edge_type, 'black')
        colored_cpg.edges[u, v]['color'] = color
        colored_cpg.edges[u, v]['penwidth'] = '3'

    return colored_cpg


def build_cpg(
    project_extracted_data_root: Path, line_vuln: Dict[str, Any], output_path: Path,
    test_mode: bool = True
) -> Tuple[nx.DiGraph, Dict[str, int]]:
    """
    Build the Code Property Graph (CPG)
    """
    print(f"[*] Building CPG...")

    # 1. Initialize
    cpg = nx.DiGraph()
    # 1. Generate AST Graphs
    print(f"[*] Generating AST Graphs...")
    ast_graph, ast_func_index = load_and_process_ast(
        project_extracted_data_root / "ABI"
    )

    cfg_graph, cfg_function_index = load_and_process_cfg(
        project_extracted_data_root / "CFG"
    )

    ir_index = load_and_process_irs(project_extracted_data_root / "IR", test_mode, output_path)

    cpg_index = link_ast_to_cfg(
        cpg, cfg_graph, ast_graph, ast_func_index, cfg_function_index
    )

    print(f"   > Nodes after merge: {cpg.number_of_nodes()}")
    print(f"   > Edges after merge: {cpg.number_of_edges()}")

    print(f" - Total AST Nodes: {ast_graph.number_of_nodes()}")
    print(f" - Total AST Links: {ast_graph.number_of_edges()}")

    print(f" - Total CFG Nodes: {cfg_graph.number_of_nodes()}")
    print(f" - Total CFG Links: {cfg_graph.number_of_edges()}")

    extract_call_metadata(cpg, cpg_index)
    link_calls_by_attributes(cpg, cpg_index)
    add_data_flow_edges(cpg, cpg_index, cfg_function_index, ir_index)
    vuln_labels_map = annotate_nodes_with_vulnerabilities(cpg, line_vuln)
    
    print(f"[*] Trimming single-type clusters...")
    # trim_single_type_clusters(cpg)

    print(f"CPG (AST-CFG) Construction Complete.")
    print(f" - Total Nodes: {cpg.number_of_nodes()}")
    print(f" - Total Edges: {cpg.number_of_edges()}")

    json.dump(cpg_index, open(f"{output_path}/ast_cfg_cpg_index.json", "w"), indent=2)
    json.dump(
        cfg_function_index,
        open(f"{output_path}/cfg_function_index.json", "w"),
        indent=2,
    )
    json.dump(ir_index, open(f"{output_path}/ir_index.json", "w"), indent=2)
    json.dump(
        vuln_labels_map, open(f"{output_path}/vuln_node_mappings.json", "w"), indent=2
    )
    from networkx.drawing.nx_pydot import write_dot

    # Output 1: DOT File
    try: 
        colored_cpg = color_cpg(cpg, vuln_labels_map)
        write_dot(colored_cpg, f"{output_path}/cpg_graph.dot")
        write_dot(cfg_graph, f"{output_path}/cfg_graph.dot")
        write_dot(ast_graph, f"{output_path}/ast_graph.dot")
        # save as gpickle
        pickle.dump(cpg, open(f"{output_path}/cpg_graph.gpickle", "wb"), protocol=5)
        pickle.dump(
            cfg_graph, open(f"{output_path}/cfg_graph.gpickle", "wb"), protocol=5
        )
        pickle.dump(
            ast_graph, open(f"{output_path}/ast_graph.gpickle", "wb"), protocol=5
        )

    except Exception as e:
        print(f"Error saving graph to DOT file: {e}")

    print(f"[*] CPG Construction Complete.")

    vuln_node_matches = _count_vuln_node_matches(vuln_labels_map)
    build_stats = {
        "ast_node_count": ast_graph.number_of_nodes(),
        "ast_edge_count": ast_graph.number_of_edges(),
        "cfg_node_count": cfg_graph.number_of_nodes(),
        "cfg_edge_count": cfg_graph.number_of_edges(),
        "vuln_entry_count": len(vuln_labels_map),
        "vuln_matched_node_count": vuln_node_matches,
    }

    return cpg, build_stats
