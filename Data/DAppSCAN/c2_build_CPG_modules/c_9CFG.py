import json
import hashlib
import re
from collections import defaultdict
from pathlib import Path
import networkx as nx
from typing import Dict, Any, Iterator, Optional, Tuple
import os
import pydot
from .c_2constants import EdgeTypes, NodeTypes, GraphAttributes, SEPARATOR, LIST_SEP
from .c_3IR import analyze_ir_lines 

def get_all_pydot_nodes(graph: Any) -> Iterator[Any]:
    """
    Recursively yield all nodes from a pydot graph and its subgraphs.

    Args:
        graph: The pydot graph object.

    Yields:
        Pydot node objects.
    """
    for node in graph.get_nodes():
        yield node
    for subgraph in graph.get_subgraphs():
        yield from get_all_pydot_nodes(subgraph)


def parse_cfg_label(label_text: str) -> Dict[str, Any]:
    """
    Parse a node label into structured fields: node_type, node_expression, raw_ir.

    Args:
        label_text: The raw label text from the DOT file.

    Returns:
        A dictionary with parsed fields.
    """
    node_type = "generic"
    node_id = None
    node_expression = ""
    raw_ir = ""

    lines = label_text.splitlines()
    current_section = None

    for line in lines:
        line = line.strip()
        if line.startswith("Node Type:"):
            m = re.match(r"Node Type:\s*(\w+)\s*(\d*)", line)
            if m:
                node_type = m.group(1).lower()
                node_id = int(m.group(2)) if m.group(2) else None
            current_section = "node_type"
        elif line.startswith("EXPRESSION:"):
            current_section = "expression"
        elif line.startswith("IRs:"):
            current_section = "ir"
        else:
            if current_section == "expression":
                node_expression += line + "\n"
            elif current_section == "ir":
                raw_ir += line + "\n"

    return {
        f"{GraphAttributes.SUB_NODE_TYPE}": node_type,
        f"{GraphAttributes.NODE_ID}": node_id,
        f"{GraphAttributes.EXPRESSION}": node_expression.strip(),
        f"{GraphAttributes.IR}": raw_ir.strip(),
    }


def _collect_ir_statistics(
    ir_raw: str,
    ir_nodes: Optional[list],
    ir_edges: Optional[list],
    ir_vars: Optional[dict],
) -> Dict[str, Any]:
    ir_lines = [ln for ln in (ir_raw or "").splitlines() if ln.strip()]
    stats: Dict[str, Any] = {
        "ir_line_count": len(ir_lines),
        "ir_instruction_count": 0,
        "ir_unique_var_count": len(ir_vars or {}),
        "ir_edge_count": len(ir_edges or []),
        "ir_call_count": 0,
        "ir_assign_count": 0,
        "ir_phi_count": 0,
        "ir_return_count": 0,
        "ir_condition_count": 0,
        "ir_tmp_ref_count": 0,
        "ir_def_count": 0,
        "ir_use_count": 0,
    }

    if not ir_nodes:
        return stats

    for raw in ir_nodes:
        parts = (raw or "").split(SEPARATOR)
        if len(parts) < 9:
            continue
        kind = (parts[1] or "").lower()
        defines = [item for item in (parts[3] or "").split(LIST_SEP) if item]
        uses = [item for item in (parts[4] or "").split(LIST_SEP) if item]
        temps = [item for item in (parts[5] or "").split(LIST_SEP) if item]

        stats["ir_instruction_count"] += 1
        stats["ir_def_count"] += len(defines)
        stats["ir_use_count"] += len(uses)
        stats["ir_tmp_ref_count"] += len(temps)

        if kind == "call":
            stats["ir_call_count"] += 1
        elif kind in ("assign", "binary_operation"):
            stats["ir_assign_count"] += 1
        elif kind == "phi":
            stats["ir_phi_count"] += 1
        elif kind == "return":
            stats["ir_return_count"] += 1
        elif kind == "condition":
            stats["ir_condition_count"] += 1

    return stats


def _collect_cfg_ml_metadata(
    parsed_node: Dict[str, Any],
    ir_nodes: Optional[list],
    ir_edges: Optional[list],
    ir_vars: Optional[dict],
) -> Dict[str, Any]:
    expression = parsed_node.get(GraphAttributes.EXPRESSION, "") or ""
    expr_norm = re.sub(r"\s+", " ", expression).strip() or ""
    ir_raw = parsed_node.get(GraphAttributes.IR, "") or ""
    metadata: Dict[str, Any] = {}
    if expr_norm:
        metadata["expression_norm"] = expr_norm
        metadata["expression_char_len"] = len(expr_norm)
        metadata["expression_token_len"] = len(expr_norm.split())

    expr_lower = expr_norm.lower()
    metadata["expression_has_require"] = "require" in expr_lower
    metadata["expression_has_revert"] = "revert" in expr_lower or "assert" in expr_lower
    metadata["expression_is_call_like"] = "call" in expr_lower

    node_sub_type = (parsed_node.get(GraphAttributes.SUB_NODE_TYPE) or "").lower()
    metadata["node_is_branch_like"] = node_sub_type in {
        "if",
        "while",
        "for",
        "do_while",
        "switch",
        "condition",
    }
    metadata["node_is_terminator"] = node_sub_type in {
        "return",
        "stop",
        "throw",
        "revert",
    }
    metadata["node_is_callsite"] = "call" in node_sub_type or "call" in expr_lower

    ir_stats = _collect_ir_statistics(ir_raw, ir_nodes, ir_edges, ir_vars)
    metadata.update(ir_stats)

    return metadata


def _get_edge_data_cfg(
    src_node: Dict[str, Any], dst_node: Dict[str, Any], edge_label: str = ""
) -> Dict[str, Any]:

    src_type = (src_node.get(GraphAttributes.SUB_NODE_TYPE) or "").lower()
    src_ir = (src_node.get(GraphAttributes.IR) or "").upper()
    dst_expr = (dst_node.get(GraphAttributes.EXPRESSION) or "").upper()

    edge_type = EdgeTypes.CF

    # --- Explicit branch labels from DOT edge ---
    if edge_label:
        lbl = edge_label.strip().lower()
        if "true" in lbl:
            edge_type = EdgeTypes.CF_TRUE
        elif "false" in lbl:
            edge_type = EdgeTypes.CF_FALSE

    # --- Fallback if no edge label ---
    elif src_type == "if":
        if "REVERT" in dst_expr or "THROW" in dst_expr:
            edge_type = EdgeTypes.CF_FALSE
        else:
            edge_type = EdgeTypes.CF_TRUE

    elif src_type in ("for", "while", "do_while"):
        if "BREAK" in dst_expr:
            edge_type = EdgeTypes.LOOP_EXIT
        elif "CONTINUE" in dst_expr:
            edge_type = EdgeTypes.LOOP_CONTINUE
        else:
            edge_type = EdgeTypes.LOOP_BACK

    elif "CALL" in src_ir:
        edge_type = EdgeTypes.CALL

    elif src_type in ("return", "stop", "throw"):
        edge_type = EdgeTypes.RETURN

    return {GraphAttributes.EDGE_TYPE: EdgeTypes.CF, GraphAttributes.SUB_EDGE_TYPE: edge_type, GraphAttributes.LABEL: edge_type}


FUNC_SIG_RE = re.compile(r"([^(]+)\((.*)\)")


def _normalize_cfg_file_name(file_name: str) -> str:
    name = file_name.strip()
    if name.endswith(".json"):
        return name[:-5]
    return name


def _function_key(file_name: str, contract: str, func: str) -> Tuple[str, str, str]:
    match = FUNC_SIG_RE.match(func)
    if match:
        func_name = match.group(1)
        args = match.group(2).strip()
        args_fmt = f"({args})" if args else "()"
    else:
        func_name = func
        args_fmt = "()"
    clean_file = _normalize_cfg_file_name(file_name)
    key = f"{clean_file}{SEPARATOR}{contract}{SEPARATOR}{func_name}{SEPARATOR}{args_fmt}"
    return key, func_name, args_fmt


def combine_cfg(raw_cfg_dict: Dict[str, Any]) -> Tuple[nx.DiGraph, Dict[str, Dict[str, Any]]]:
    """
    Combine raw CFG dictionary into a NetworkX DiGraph.

    Args:
        raw_cfg_dict: The raw CFG dictionary.

    Returns:
        The combined Control Flow Graph.
    """
    combined = nx.DiGraph()
    cfg_index: Dict[str, Dict[str, Any]] = {}
    if not raw_cfg_dict:
        return combined, cfg_index

    seen = set()  # To dedupe CFG for same function (centralize inherited)
    node_mapping = {}
    edge_map = {}
    referenced_nodes = set()
    referencing_nodes = set()

    node_id_counter = 0

    # 1. Add nodes with global IDs and parse labels
    for key, pydot_graph in raw_cfg_dict.items():
        if not pydot_graph:
            continue

        key_parts = key.split("-")  # this is different from from internal separator

        file_name = key_parts[0]
        contract_name = key_parts[1]
        # Extract function name by removing .dot extension first, then handle parentheses
        function_with_ext = key_parts[2]
        if function_with_ext.endswith(".dot"):
            function_name = function_with_ext[:-4]  # Remove .dot extension
        else:
            function_name = function_with_ext

        # Dedupe: skip if already processed this function
        # if (contract_name, function_name) in seen:
        #     continue

        seen.add((contract_name, function_name))

        for node in get_all_pydot_nodes(pydot_graph):
            raw_name = str(node.get_name()).strip('"')
            if raw_name in ("node", "graph", "edge", "\\n") or not raw_name:
                continue

            node_id_counter += 1
            new_id = str(node_id_counter)

            label_text = node.get_label() or ""
            label_text = label_text.strip('"')

            cfg_node_data = parse_cfg_label(label_text)
            cfg_node_data[GraphAttributes.NODE_TYPE] = NodeTypes.CFG_NODE
            cfg_node_data[GraphAttributes.LABEL] = (
                f"[{contract_name}][{function_name}]{cfg_node_data[GraphAttributes.EXPRESSION]}"
            )
            cfg_node_data[GraphAttributes.FUNCTION] = function_name
            cfg_node_data[GraphAttributes.CONTRACT] = contract_name
            cfg_node_data[GraphAttributes.FILE] = file_name

            ir_vars, ir_nodes, ir_edges = analyze_ir_lines(
                str(cfg_node_data.get(GraphAttributes.IR) or "")
            )
            ml_metadata = _collect_cfg_ml_metadata(cfg_node_data, ir_nodes, ir_edges, ir_vars)
            cfg_node_data.update(ml_metadata)

            combined.add_node(new_id, **cfg_node_data)
            node_mapping[(key, raw_name)] = {"new_id": new_id, "parsed": cfg_node_data}

            func_key, func_name_clean, func_args = _function_key(
                file_name, contract_name, function_name
            )
            expression = cfg_node_data.get(GraphAttributes.EXPRESSION, "")
            expr_norm = expression.strip()
            expr_hash = hashlib.md5(expr_norm.encode("utf-8")).hexdigest()
            node_key = f"{func_key}{SEPARATOR}{new_id}{SEPARATOR}{expr_hash}"
            cfg_index[node_key] = {
                "node_id": new_id,
                "file": _normalize_cfg_file_name(file_name),
                "contract": contract_name,
                "function_name": func_name_clean,
                "function_args": func_args,
                "expression": expression,
                "ir": cfg_node_data.get(GraphAttributes.IR),
                "sub_node_type": cfg_node_data.get(GraphAttributes.SUB_NODE_TYPE),
                "ir_vars": ir_vars,
                "ir_nodes": ir_nodes,
                "ir_edges": ir_edges,
            }
            cfg_index[node_key].update(ml_metadata)

    # 2. Process edges with deduplication and frequency
    for key, pydot_graph in raw_cfg_dict.items():
        if not pydot_graph:
            continue

        for edge in pydot_graph.get_edges():
            src_raw = str(edge.get_source()).strip('"')
            dst_raw = str(edge.get_destination()).strip('"')
            edge_label = edge.get("label") or ""

            src_info = node_mapping.get((key, src_raw))
            dst_info = node_mapping.get((key, dst_raw))

            if not src_info or not dst_info:
                continue

            src_id = src_info["new_id"]
            dst_id = dst_info["new_id"]
            src_parsed = src_info["parsed"]
            dst_parsed = dst_info["parsed"]

            edge_data = _get_edge_data_cfg(src_parsed, dst_parsed, edge_label)
            edge_type = edge_data.get(GraphAttributes.SUB_EDGE_TYPE, edge_data[GraphAttributes.EDGE_TYPE])

            key_edge = (src_id, dst_id, edge_type)
            if key_edge not in edge_map:
                edge_map[key_edge] = edge_data
            # else: already have the data

            referencing_nodes.add(src_id)
            referenced_nodes.add(dst_id)

    # 3. Add deduped edges
    for (src_id, dst_id, edge_type), edge_data in edge_map.items():
        combined.add_edge(src_id, dst_id, **edge_data)

    # from networkx.drawing.nx_pydot import write_dot

    # dot_output = "test_output/cfg_graph.dot"
    # # Output 1: DOT File
    # try:
    #     write_dot(combined, dot_output)
    #     print(f"Success: Graph saved to {dot_output}")
    # except Exception as e:
    #     print(f"Error saving graph to DOT file: {e}")
    # index_path = Path("test_output/cfg_funtion_index.json")
    # index_path.parent.mkdir(parents=True, exist_ok=True)
    # with index_path.open("w", encoding="utf-8") as fh:
    #     json.dump(cfg_index, fh, indent=2, ensure_ascii=False)

    return combined, cfg_index

def load_cfg_dot_files(cfg_root):
    """
    Load all .dot files in the given folder and return a dict of pydot graphs.
    Key is the filename without extension.
    """
    raw_dot_dict = {}
    for file in os.listdir(cfg_root):
        if file.endswith(".dot"):
            # print(file)
            key = os.path.splitext(file)[0]
            try:
                graphs = pydot.graph_from_dot_file(os.path.join(cfg_root, file))
                if graphs:
                    raw_dot_dict[key] = graphs[0]
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
    return raw_dot_dict

def load_and_process_cfg(cfg_root: Path):
    raw_cfg_dict = load_cfg_dot_files(cfg_root)
    combined, cfg_index = combine_cfg(raw_cfg_dict)
    return combined, cfg_index