from collections import defaultdict, Counter
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple
from .c_2constants import SEPARATOR, LIST_SEP
import hashlib

# ==========================================
# 1. CONSTANTS & REGEX CONFIGURATION
# ==========================================
BINARY_OPS = ["&&", "||", "==", "!=", ">=", "<=", "+", "-", "*", "/", "%", ">", "<"]

# Regex to find variables (TMP, REF, or standard naming)
VAR_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*(_\d+)?|TMP_\d+|REF_\d+)\b"
)

# Regex to capture "var(type)" declarations (handles mapping(address => uint) etc.)
VAR_WITH_TYPE_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(\s*([^)]*)\s*\)(?=\s*(?::=|=>|=|;|,|$))"
)

ASSIGN_RE = re.compile(r":=")
PHI_RE = re.compile(r"ϕ|phi", re.IGNORECASE)
CALL_RE = re.compile(
    r"\b(CALL|INTERNAL_CALL|LIBRARY_CALL|MODIFIER_CALL|SOLIDITY_CALL|LOW_LEVEL_CALL)\b",
    re.IGNORECASE,
)
RETURN_RE = re.compile(r"^RETURN\b", re.IGNORECASE)
TMP_RE = re.compile(r"(TMP_\d+)")

# Regex for finding the arguments:[...] block
ARGS_BLOCK_RE = re.compile(r"arguments\s*:\s*\[(.*?)\]")

FUNC_SIG_PATTERN = re.compile(r"([^(]+)\((.*)\)")
FUNC_BLOCK_PATTERN = re.compile(r"^(Function|Modifier)\s+((?:[^.]+\.)?)(.+)$")
CONTRACT_LINE_RE = re.compile(r"^Contract\s+([A-Za-z0-9_]+)")
ENTRY_BLOCK_PATTERN = re.compile(r"^ENTRY_POINT(?:\s+(.*))?$", re.IGNORECASE)
EXPRESSION_LINE_RE = re.compile(r"^(?:Expression|EXPRESSION)\s*:?(.*)$")


# ==========================================
# 2. STRING PARSING UTILITIES
# ==========================================


def extract_vars_from_text(s: str) -> List[str]:
    """Return a list of variable-like tokens found in s (in order)."""
    if not s:
        return []
    # ignore placeholders like '_'
    return [m.group(1) for m in VAR_RE.finditer(s) if m.group(1) != "_"]


def extract_var_types(s: str) -> Dict[str, str]:
    """Return mapping var -> type from patterns like 'name(type)'."""
    mapping = {}
    for m in VAR_WITH_TYPE_RE.finditer(s):
        name = m.group(1)
        typ = m.group(2)
        mapping[name] = typ
    return mapping


def extract_arguments_from_list(args_str: str) -> List[str]:
    """
    Parses a string like "'REF_1', 'amount', 'Error msg'"
    Returns valid variables only, ignoring string literals.
    """
    uses: List[str] = []
    # Find all quoted strings '...' or "..."
    tokens = re.findall(r"['\"](.*?)['\"]", args_str)

    for t in tokens:
        # Heuristic: If it matches our variable regex perfectly, it's a variable.
        # If it has spaces or special chars (like 'Error: ...'), it's a string literal.
        if VAR_RE.fullmatch(t):
            uses.append(t)

    # Also capture unquoted variables (if any exist in the format)
    # Slither IR usually quotes arguments but be resilient.
    unquoted = [u.strip() for u in re.split(r",(?![^\[\]]*\])", args_str) if u.strip()]
    for u in unquoted:
        # remove any wrapping quotes
        u_clean = u.strip().strip("\"'")
        if VAR_RE.fullmatch(u_clean) and u_clean not in uses:
            uses.append(u_clean)

    return uses


def infer_role_and_ssa(varname: str) -> Tuple[str, str]:
    role, ssa = "var", ""
    if varname.startswith("TMP_"):
        role = "tmp"
    elif varname.startswith("REF_"):
        role = "ref"
    elif "_" in varname:
        parts = varname.rsplit("_", 1)
        if len(parts) > 1 and parts[1].isdigit():
            ssa = parts[1]
    return role, ssa


def parse_function_signature(funcsig: str) -> Tuple[str, str]:
    match = FUNC_SIG_PATTERN.match(funcsig)
    return (match.group(1).strip(), match.group(2).strip()) if match else (funcsig, "")


def _normalize_source_file_name(file_name: str) -> str:
    clean = file_name.removeprefix("ir_")
    if clean.endswith(".sol.json"):
        return clean[: -len(".json")]
    if clean.endswith(".json"):
        return clean[: -len(".json")]
    return clean


# ==========================================
# 3. GRAPH CONSTRUCTION LOGIC
# ==========================================


def register_node_and_edges(
    nodes: List[str],
    edges: List[str],
    node_idx: int,
    kind: str,
    op: str,
    defines: List[str],
    uses: List[str],
    temps: List[str],
    vtype: str,
    raw_ir: str,
    instruction_index: int,
) -> None:
    """Append a compact node row to nodes and corresponding edges."""
    node_id = f"n{node_idx}"

    # 1. Compact Node
    row = [
        node_id,
        kind or "",
        str(op) if op is not None else "",
        LIST_SEP.join(defines),
        LIST_SEP.join(uses),
        LIST_SEP.join(temps),
        vtype or "",
        str(instruction_index),
        raw_ir or "",
    ]
    nodes.append(SEPARATOR.join(row))

    # 2. Compact Edges (Uses -> Node)
    for idx, u in enumerate(uses):
        edge_row = [u, node_id, "node", "data", str(idx)]
        edges.append(SEPARATOR.join(edge_row))

    # 3. Compact Edges (Node -> Defines)
    for d in defines:
        edge_row = [node_id, d, "var", "assign", ""]
        edges.append(SEPARATOR.join(edge_row))


def analyze_ir_lines(ir_raw: str):
    """Main IR parsing routine. Returns ir_vars, nodes, edges."""
    nodes: List[str] = []
    edges: List[str] = []
    ir_vars: Dict[str, str] = {}

    var_types = extract_var_types(ir_raw)
    type_values = set(var_types.values())
    lines = [ln.strip() for ln in ir_raw.splitlines() if ln.strip()]
    node_idx = 0
    instruction_index = 0

    for ln in lines:
        # Update types for current line
        current_types = extract_var_types(ln)
        var_types.update(current_types)
        type_values = set(var_types.values())

        defines: List[str] = []
        uses: List[str] = []
        temps: List[str] = []
        kind, op, node_type = None, None, ""

        # --- 1. RETURN ---
        if RETURN_RE.match(ln):
            kind = "return"
            rest = ln[len("RETURN") :].strip()
            # handle possible parentheses: RETURN (a, b)
            if rest.startswith("(") and rest.endswith(")"):
                rest = rest[1:-1]
            raw_uses = [a.strip() for a in rest.split(",") if a.strip()]
            uses = [
                u
                for u in raw_uses
                if (
                    VAR_RE.match(u)
                    and u not in type_values
                    and u not in {"False", "True"}
                    and not u.isdigit()
                )
            ]

            register_node_and_edges(
                nodes,
                edges,
                node_idx,
                kind,
                "",
                [],
                uses,
                [],
                "",
                ln,
                instruction_index,
            )
            node_idx += 1
            instruction_index += 1
            continue

        # --- 2. CALLS (Library, Internal, etc) ---
        # Prioritize CALL check because they often look like assignments (TMP = CALL)
        if CALL_RE.search(ln):
            kind = "call"
            op = "call"

            # Extract arguments: [...]
            args_match = ARGS_BLOCK_RE.search(ln)
            if args_match:
                # Parse the content inside [ ... ]
                uses = extract_arguments_from_list(args_match.group(1))
            else:
                # Fallback to parenthesis extraction if arguments:[...] missing
                matches = re.findall(r"\(([^()]*)\)|\[([^\[\]]*)\]", ln)
                if matches:
                    # Flatten tuple → take the non-empty group
                    args_part = matches[-1][0] or matches[-1][1] or ""
                else:
                    args_part = ""

                uses = [x.strip() for x in args_part.split(",") if x.strip()]
                uses = [u for u in uses if u not in type_values and VAR_RE.fullmatch(u)]

            # Extract Defines (LHS of =)
            if "=" in ln:
                lhs = ln.split("=", 1)[0]
                defines = [
                    t for t in extract_vars_from_text(lhs) if t not in type_values
                ]

            temps = TMP_RE.findall(ln)
            if defines:
                node_type = var_types.get(defines[0], "")

            register_node_and_edges(
                nodes,
                edges,
                node_idx,
                kind,
                op,
                defines,
                uses,
                temps,
                node_type,
                ln,
                instruction_index,
            )
            node_idx += 1
            instruction_index += 1
            continue

        # --- 3. PHI NODES ---
        if PHI_RE.search(ln) and ":=" in ln:
            kind, op = "phi", ":="
            lhs, rhs = ln.split(":=", 1)

            defines = [t for t in extract_vars_from_text(lhs) if t not in type_values]

            # Handle phi list syntax: phi(['var1', 'var2'])
            if "[" in rhs and "]" in rhs:
                uses = extract_arguments_from_list(rhs)
            else:
                uses = [t for t in extract_vars_from_text(rhs) if t not in type_values]

            if defines:
                node_type = var_types.get(defines[0], "")
            register_node_and_edges(
                nodes,
                edges,
                node_idx,
                kind,
                op,
                defines,
                uses,
                [],
                node_type,
                ln,
                instruction_index,
            )
            node_idx += 1
            instruction_index += 1
            continue

        # --- 4. ASSIGNMENT (:=) ---
        if ASSIGN_RE.search(ln):
            lhs, rhs = ln.split(":=", 1)
            defines = [t for t in extract_vars_from_text(lhs) if t not in type_values]
            uses = [t for t in extract_vars_from_text(rhs) if t not in type_values]

            kind, op = "assign", ":="
            if defines:
                node_type = var_types.get(defines[0], "")

            register_node_and_edges(
                nodes,
                edges,
                node_idx,
                kind,
                op,
                defines,
                uses,
                [],
                node_type,
                ln,
                instruction_index,
            )
            node_idx += 1
            instruction_index += 1
            continue

        # --- 5. CONDITION ---
        if ln.startswith("CONDITION "):
            uses = [
                t
                for t in extract_vars_from_text(ln[len("CONDITION ") :].strip())
                if t not in type_values
            ]
            register_node_and_edges(
                nodes,
                edges,
                node_idx,
                "condition",
                "condition",
                [],
                uses,
                [],
                "",
                ln,
                instruction_index,
            )
            node_idx += 1
            instruction_index += 1
            continue

        # --- 6. END_IF ---
        if ln.strip() == "END_IF":
            register_node_and_edges(
                nodes,
                edges,
                node_idx,
                "end_if",
                "end_if",
                [],
                [],
                [],
                "",
                ln,
                instruction_index,
            )
            node_idx += 1
            instruction_index += 1
            continue

        # --- 7. ASSIGNMENT / BINARY OP (=) ---
        if "=" in ln and not any(
            o in ln for o in ["==", "!=", ">=", "<=", "&&", "||", "->", ":="]
        ):
            lhs, rhs = ln.split("=", 1)
            defines = [t for t in extract_vars_from_text(lhs) if t not in type_values]
            uses = [t for t in extract_vars_from_text(rhs) if t not in type_values]

            binary_op = next((o for o in BINARY_OPS if o in rhs), None)
            temps = TMP_RE.findall(rhs) + TMP_RE.findall(lhs)

            kind = "binary_operation" if binary_op else "assign"
            op = binary_op if binary_op else "="
            if defines:
                node_type = var_types.get(defines[0], "")

            register_node_and_edges(
                nodes,
                edges,
                node_idx,
                kind,
                op,
                defines,
                uses,
                temps,
                node_type,
                ln,
                instruction_index,
            )
            node_idx += 1
            instruction_index += 1
            continue

        # --- 8. MAPPING ASSIGN (->) ---
        if "->" in ln and "=" not in ln:
            lhs, rhs = ln.split("->", 1)
            defines = [t for t in extract_vars_from_text(lhs) if t not in type_values]
            uses = [t for t in extract_vars_from_text(rhs) if t not in type_values]
            temps = TMP_RE.findall(rhs) + TMP_RE.findall(lhs)
            if defines:
                node_type = var_types.get(defines[0], "")
            register_node_and_edges(
                nodes,
                edges,
                node_idx,
                "assign",
                "->",
                defines,
                uses,
                temps,
                node_type,
                ln,
                instruction_index,
            )
            node_idx += 1
            instruction_index += 1
            continue

        # --- 9. SPECIFIC KEYWORD LINES (REQUIRE, REVERT, ASSERT, EMIT, LOG) ---
        kw = ln.split()[0].upper() if ln.split() else ""
        if kw in {"REQUIRE", "REVERT", "ASSERT", "EMIT", "LOG", "LOW_LEVEL_CALL"}:
            # EMIT and LOG often have arguments in parentheses or brackets
            matches = re.findall(r"\((.*)\)|\[(.*)\]", ln)
            args_part = ",".join([m[0] or m[1] for m in matches if m[0] or m[1]])
            uses = [
                t for t in extract_vars_from_text(args_part) if t not in type_values
            ]
            temps = TMP_RE.findall(ln)
            register_node_and_edges(
                nodes,
                edges,
                node_idx,
                kw.lower(),
                kw.lower(),
                [],
                uses,
                temps,
                "",
                ln,
                instruction_index,
            )
            node_idx += 1
            instruction_index += 1
            continue

        # --- 10. FALLBACK ---
        # Try to detect generic usage
        uses = [t for t in extract_vars_from_text(ln) if t not in type_values]
        if uses:
            defines = []
            # If we found binary ops but failed earlier checks
            if any(op_sym in ln for op_sym in BINARY_OPS) and "=" in ln:
                lhs, rhs = ln.split("=", 1)
                defines = [
                    t for t in extract_vars_from_text(lhs) if t not in type_values
                ]
                uses = [t for t in extract_vars_from_text(rhs) if t not in type_values]
                kind, op = "binary_operation", (
                    next((o for o in BINARY_OPS if o in rhs), "op")
                )
            else:
                kind, op = "unknown", "unknown"

            register_node_and_edges(
                nodes,
                edges,
                node_idx,
                kind,
                op,
                defines,
                uses,
                TMP_RE.findall(ln),
                "",
                ln,
                instruction_index,
            )
            node_idx += 1
            instruction_index += 1

    # --- FINAL COMPACTION ---
    # Re-extract vars from generated nodes/edges to build the IR_VARS dict
    all_vars = set()

    # Extract vars from nodes (defines and uses fields are stored)
    for n_str in nodes:
        parts = n_str.split(SEPARATOR)
        # node columns: node_id, kind, op, defines, uses, temps, vtype, instruction_index, raw_ir
        if len(parts) < 9:
            continue
        defines_field = parts[3]
        uses_field = parts[4]
        if defines_field:
            for d in defines_field.split(LIST_SEP):
                if d:
                    all_vars.add(d)
        if uses_field:
            for u in uses_field.split(LIST_SEP):
                if u:
                    all_vars.add(u)

    # Also ensure edges capture any var names
    for e_str in edges:
        parts = e_str.split(SEPARATOR)
        if len(parts) < 3:
            continue
        src, dst, to_kind = parts[0], parts[1], parts[2]
        if to_kind == "node":
            all_vars.add(src)
        elif to_kind == "var":
            all_vars.add(dst)

    for v in sorted(all_vars):
        if (
            v not in {"CONDITION", "END_IF", "_", "False", "True", "RETURN"}
            and not v.isdigit()
        ):
            role, ssa = infer_role_and_ssa(v)
            ir_vars[v] = f"{var_types.get(v, '')}{SEPARATOR}{role}{SEPARATOR}{ssa}"

    return ir_vars, nodes, edges


# ==========================================
# 4. MAIN LOADER
# ==========================================

def load_and_process_irs(ir_root: Path, test_mode: bool = False, test_out_dir: Path = None):
    """Parse every Slither IR printer JSON under ir_root into a compact index."""
    ir_index = {}
    seen_entries = set()
    processed_functions = set()

    for file_path in ir_root.rglob("*.json"):
        try:
            relative_name = (
                file_path.relative_to(ir_root).as_posix()
                if file_path.is_relative_to(ir_root)
                else str(file_path)
            )
        except ValueError:
            relative_name = str(file_path)
        try:
            raw_content = file_path.read_text(encoding="utf-8", errors="ignore")
            if not raw_content.strip():
                continue
            data = json.loads(raw_content)
        except Exception:
            continue

        for printer in data.get("results", {}).get("printers", []):
            lines = printer.get("description", "").splitlines()
            current_contract_block = ""
            current_contract, current_funcsig, block_lines = "", "", []

            def process_block():
                nonlocal block_lines, current_contract, current_funcsig
                if not current_funcsig:
                    block_lines = []
                    return

                func, args = parse_function_signature(current_funcsig)
                func_block_key = (
                    relative_name,
                    current_contract,
                    func,
                    args,
                )
                if func_block_key in processed_functions:
                    block_lines = []
                    return
                processed_functions.add(func_block_key)
                current_expr, ir_lines, order = "", [], 0

                def process_expression():
                    nonlocal current_expr, ir_lines, order
                    if not current_expr:
                        return
                    expr_raw = current_expr.strip()
                    ir_raw = "\n".join(ir_lines).rstrip()
                    signature = (
                        relative_name,
                        current_contract,
                        func,
                        args,
                        expr_raw,
                        ir_raw,
                    )
                    if signature in seen_entries:
                        current_expr = ""
                        ir_lines = []
                        return
                    seen_entries.add(signature)
                    md5_hash = hashlib.md5((expr_raw).encode('utf-8')).hexdigest()
                    clean_file_name = _normalize_source_file_name(file_path.name)

                    ir_vars, nodes, edges = analyze_ir_lines(ir_raw)

                    key = f"{clean_file_name}{SEPARATOR}{current_contract}{SEPARATOR}{func}{SEPARATOR}({args}){SEPARATOR}{order}{SEPARATOR}{md5_hash}"
                    ir_index[key] = {
                        "key": key,
                        "order": order,
                        "expression": expr_raw,
                        "ir": ir_raw,
                        "file": clean_file_name,
                        "contract": current_contract,
                        "function": func,
                        "function_args": f"({args})",
                        "ir_vars": ir_vars,
                        "nodes": nodes,
                        "edges": edges,
                    }
                    order += 1
                    current_expr = ""
                    ir_lines = []

                for line in block_lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    expr_match = EXPRESSION_LINE_RE.match(stripped)
                    if expr_match:
                        process_expression()
                        current_expr = expr_match.group(1).strip()
                        continue
                    elif current_expr and not (
                        stripped.startswith("Function") or stripped.startswith("IRs:")
                    ):
                        ir_lines.append(stripped)
                process_expression()
                block_lines = []

            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                contract_match = CONTRACT_LINE_RE.match(stripped_line)
                if contract_match:
                    process_block()
                    current_contract_block = contract_match.group(1)
                    continue

                match = FUNC_BLOCK_PATTERN.match(stripped_line)
                if match:
                    process_block()
                    contract_prefix = match.group(2).rstrip(".")
                    effective_contract = current_contract_block or contract_prefix
                    current_contract = effective_contract or contract_prefix
                    current_funcsig = match.group(3)
                    block_lines = [line]
                    continue

                entry_match = ENTRY_BLOCK_PATTERN.match(stripped_line)
                if entry_match:
                    process_block()
                    entry_suffix = entry_match.group(1)
                    entry_label = (
                        entry_suffix.strip() if entry_suffix else "ENTRY_POINT"
                    )
                    current_funcsig = entry_label or "ENTRY_POINT"
                    block_lines = [line]
                    continue

                if current_funcsig:
                    block_lines.append(line)
            process_block()

    if test_mode:
        save_ir_index_as_dot(ir_index, test_out_dir)
    return ir_index


def save_ir_index_as_dot(ir_index: Dict[str, Dict], output_dir: Path) -> None:
    """Write a single unified, flattened DOT graph containing all items in `ir_index`.

    - Produces a single file named `unified_flattened.dot` in `output_dir`.
    - All node ids are continuous numeric values (1..N).
    - Each node label starts with `[contract][function]` followed by details.
    - Edges include detailed labels with edge type, kind and index.
    - Visual tweaks: thicker border (penwidth 3), stronger border colors and lighter fills.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / 'ir_flattened.dot'

    # Build mappings from original identifiers to numeric ids and node metadata
    id_map: Dict[str, int] = {}
    # id -> (label, border_color, fill_color, shape)
    nodes_meta: Dict[int, Tuple[str, str, str, str]] = {}
    next_id = 1

    # color palette (border, fill) - use named colors (avoid '#' hex notation)
    colors = {
        'blue_border': 'blue',
        'blue_fill': 'lightblue',
        'dark_border': 'black',
        'light_fill': 'lightgrey',
        'var_border': 'forestgreen',
        'var_fill': 'palegreen',
        'placeholder_border': 'grey',
        'placeholder_fill': 'whitesmoke',
        'edge_color': 'dimgray',
    }

    # helper to clean text fields: remove literal \n/\t, control chars, and compress whitespace
    def _clean_field(s: str) -> str:
        s = '' if s is None else str(s)
        # remove literal escape sequences and real control chars
        s = s.replace('\\n', ' ').replace('\\t', ' ').replace('\n', ' ').replace('\t', ' ')
        s = s.replace(LIST_SEP, ', ')
        s = s.replace('\\', '')
        s = re.sub(r'[\x00-\x1f]+', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    # Collect all unique variables across all items
    all_ir_vars: Dict[str, str] = {}
    for item in ir_index.values():
        all_ir_vars.update(item.get('ir_vars', {}))

    # 1) First pass: register nodes and var nodes for every item
    for key, item in sorted(ir_index.items()):
        contract = item.get('contract', '')
        function = item.get('function', '')
        location = f'[{contract}][{function}]'

        # Nodes
        for node_str in item.get('nodes', []):
            parts = node_str.split(SEPARATOR)
            if len(parts) < 9:
                continue
            orig_node = f"{key}::{parts[0]}"  # make unique across items
            if orig_node in id_map:
                continue
            num = next_id
            next_id += 1
            id_map[orig_node] = num

            kind = parts[1]
            op = parts[2]
            raw_ir = _clean_field(parts[8])

            if kind in ['assign', 'binary_operation', 'phi']:
                border = colors['blue_border']
                fill = colors['blue_fill']
            else:
                border = colors['dark_border']
                fill = colors['light_fill']
            shape = 'box'
            # cleaner label: location, kind:op, raw_ir
            label = f"{location}\n{kind}: {op}\n{raw_ir}"
            nodes_meta[num] = (label, border, fill, shape)

    # Variables: add all unique variables
    for var, varinfo in sorted(all_ir_vars.items()):
        orig_var = f"VAR::{var}"
        if orig_var in id_map:
            continue
        num = next_id
        next_id += 1
        id_map[orig_var] = num

        border = colors['var_border']
        fill = colors['var_fill']
        shape = 'ellipse'
        # cleaner label: location not needed for vars, just var: name
        label = f"var: {var}"
        nodes_meta[num] = (label, border, fill, shape)

    # 2) Second pass: ensure all endpoints referenced by edges have ids (create placeholders if needed)
    for key, item in sorted(ir_index.items()):
        for edge_str in item.get('edges', []):
            parts = edge_str.split(SEPARATOR)
            if len(parts) < 5:
                continue
            src, dst, to_kind, edge_type, idx = parts
            # determine composite keys
            src_key = f"{key}::{src}" if src.startswith('n') else f"VAR::{src}"
            dst_key = f"{key}::{dst}" if dst.startswith('n') else f"VAR::{dst}"
            for composite in (src_key, dst_key):
                if composite not in id_map:
                    num = next_id
                    next_id += 1
                    id_map[composite] = num
                    # placeholder metadata
                    label = f"{composite}"
                    nodes_meta[num] = (label, colors['placeholder_border'], colors['placeholder_fill'], 'ellipse')

    # 3) Write DOT file: nodes first, then edges with detailed labels
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('digraph G {\n')
        f.write('    rankdir=TD;\n')
        f.write('    bgcolor="white";\n')
        f.write('    graph [pad="0.5"];\n')

        # write node definitions (sorted by numeric id for deterministic output)
        for num in sorted(nodes_meta.keys()):
            label, border, fill, shape = nodes_meta[num]
            # sanitize label: remove control chars, escape backslashes and quotes
            label_s = label.replace('\r', '')
            label_s = label_s.replace('\\', '\\\\')
            label_s = re.sub(r'[\x00-\x1f]', '', label_s)
            label_escaped = label_s.replace('"', '\\"')
            # penwidth 3 for stronger borders, style filled with lighter fill
            f.write(f'    {num} [label="{label_escaped}", color="{border}", fillcolor="{fill}", style=filled, penwidth=3, shape={shape}];\n')

        # write edges
        for key, item in sorted(ir_index.items()):
            for edge_str in item.get('edges', []):
                parts = edge_str.split(SEPARATOR)
                if len(parts) < 5:
                    continue
                src, dst, to_kind, edge_type, idx = parts

                src_key = f"{key}::{src}" if src.startswith('n') else f"VAR::{src}"
                dst_key = f"{key}::{dst}" if dst.startswith('n') else f"VAR::{dst}"

                src_num = id_map.get(src_key)
                dst_num = id_map.get(dst_key)
                if src_num is None or dst_num is None:
                    # skip malformed edges
                    continue

                # create a detailed label for the edge
                et = _clean_field(edge_type)
                tk = _clean_field(to_kind)
                idx_clean = _clean_field(idx)
                edge_label = f"type: {et} | kind: {tk} | idx: {idx_clean}"
                edge_label_escaped = edge_label.replace('"', '\\"')
                f.write(f'    {src_num} -> {dst_num} [label="{edge_label_escaped}", color="{colors['edge_color']}", penwidth=2];\n')

        f.write('}\n')


def opcode_histogram(ir_text: str) -> Dict[str, int]:
    """Return a small opcode-style histogram from the IR lines.
    This function is intentionally heuristic and non-exhaustive.
    """
    if not ir_text:
        return {}
    buckets = Counter()
    ir_list = ir_text.split("\n") if isinstance(ir_text, str) else ir_text
    for ir in ir_list:
        if "LOW_LEVEL_CALL" in ir:
            buckets["CALL"] += 1
        elif "SSTORE" in ir or "SLOAD" in ir:
            buckets["STORAGE"] += 1
        elif re.search(r"INTERNAL_CALL.*(SafeMath|BNum).*add", ir):
            buckets["ADD_SAFE"] += 1
        elif re.search(r"INTERNAL_CALL.*(SafeMath|BNum).*sub", ir):
            buckets["SUB_SAFE"] += 1
        elif re.search(r"INTERNAL_CALL.*(SafeMath|BNum).*mul", ir):
            buckets["MUL_SAFE"] += 1
        elif re.search(r"INTERNAL_CALL.*(SafeMath|BNum).*div", ir):
            buckets["DIV_SAFE"] += 1
        elif re.search(r"\s=\s.*\+", ir):
            buckets["ADD_RAW"] += 1
        elif re.search(r"\s=\s.*-", ir):
            buckets["SUB_RAW"] += 1
        elif re.search(r"\s=\s.*\*", ir):
            buckets["MUL_RAW"] += 1
        elif re.search(r"\s=\s.*\/", ir):
            buckets["DIV_RAW"] += 1
        elif "REQUIRE" in ir.upper() or "require" in ir:
            buckets["REQUIRE"] += 1
        elif "REVERT" in ir.upper():
            buckets["REVERT"] += 1
        elif "CONDITION" in ir:
            buckets["COND_BRANCH"] += 1
        elif "ϕ" in ir or "phi" in ir.lower():
            buckets["PHI"] += 1
    return dict(buckets)
