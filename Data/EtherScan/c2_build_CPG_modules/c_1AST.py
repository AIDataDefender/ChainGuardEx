import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import networkx as nx

from .c_2constants import (
    EdgeTypes,
    NodeTypes,
    GraphAttributes,
    SEPARATOR,
)

# ==========================================
# 1. UTILITIES
# ==========================================


def clean_file_name(raw_name: str) -> str:
    """clean path from file name"""
    return Path(raw_name).name


def get_type_name(node):
    if not node:
        return ""
    node_type = node.get("nodeType", "")

    if node_type == "ElementaryTypeName":
        name = node.get("name", "")
        # --- Canonicalize Types ---
        if name == "uint":
            return "uint256"
        if name == "int":
            return "int256"
        if name == "bytes":
            return "bytes[]"

        return name

    if node_type == "UserDefinedTypeName":
        return node.get("name", node.get("path", ""))

    if node_type == "ArrayTypeName":
        base = get_type_name(node.get("baseType"))
        return f"{base}[]"

    if node_type == "Mapping":
        k = get_type_name(node.get("keyType"))
        v = get_type_name(node.get("valueType"))
        return f"mapping({k}=>{v})"

    return "unknown"


def reconstruct_code(node):
    """
    Recursively reconstructs a string representation of Solidity code
    from an AST node. Used to create readable labels for statement nodes.
    """
    if not node or not isinstance(node, dict):
        return ""

    node_type = node.get("nodeType", "")

    # --- Literals & Identifiers ---
    if node_type == "Literal":
        # value might be missing for some hex literals, fallback to hexValue
        return str(node.get("value", node.get("hexValue", "")))

    if node_type == "Identifier":
        return node.get("name", "")

    # --- Operations ---
    if node_type == "BinaryOperation":
        left = reconstruct_code(node.get("leftExpression"))
        right = reconstruct_code(node.get("rightExpression"))
        op = node.get("operator")
        return f"({left} {op} {right})"

    if node_type == "UnaryOperation":
        sub = reconstruct_code(node.get("subExpression"))
        op = node.get("operator")
        is_prefix = node.get("prefix", True)
        if is_prefix:
            return f"{op}{sub}"
        else:
            return f"{sub}{op}"

    if node_type == "Assignment":
        left = reconstruct_code(node.get("leftHandSide"))
        right = reconstruct_code(node.get("rightHandSide"))
        op = node.get("operator")
        return f"{left} {op} {right}"

    # --- Access & Calls ---
    if node_type == "MemberAccess":
        expr = reconstruct_code(node.get("expression"))
        name = node.get("memberName", "")
        return f"{expr}.{name}"

    if node_type == "IndexAccess":
        base = reconstruct_code(node.get("baseExpression"))
        idx = reconstruct_code(node.get("indexExpression"))
        return f"{base}[{idx}]"

    if node_type == "FunctionCall":
        expr = reconstruct_code(node.get("expression"))
        # Handle arguments
        args = []
        for arg in node.get("arguments", []):
            args.append(reconstruct_code(arg))
        return f"{expr}({', '.join(args)})"

    if node_type == "NewExpression":
        typename = reconstruct_code(node.get("typeName"))
        return f"new {typename}"

    if node_type == "TupleExpression":
        comps = [reconstruct_code(c) if c else "" for c in node.get("components", [])]
        return f"({', '.join(comps)})"

    # --- Types & Declarations ---
    if node_type == "VariableDeclaration":
        # e.g., "uint256 amount"
        t_name = reconstruct_code(node.get("typeName"))
        name = node.get("name", "")
        # Handle initialization if embedded in the declaration node
        if "value" in node and node["value"]:
            val = reconstruct_code(node.get("value"))
            return f"{t_name} {name} = {val}"
        return f"{t_name} {name}"

    if node_type == "ElementaryTypeName":
        return node.get("name", "")

    if node_type == "UserDefinedTypeName":
        return node.get("name", "")  # Or simplify path-qualified names if needed

    if node_type == "ArrayTypeName":
        base = reconstruct_code(node.get("baseType"))
        return f"{base}[]"

    if node_type == "Mapping":
        k = reconstruct_code(node.get("keyType"))
        v = reconstruct_code(node.get("valueType"))
        return f"mapping({k}=>{v})"

    return ""  # Fallback for unhandled types


def _flatten_child_types(children: List[Any]) -> List[str]:
    child_types: List[str] = []
    for child in children:
        if isinstance(child, dict):
            node_type = child.get("nodeType")
            if node_type:
                child_types.append(node_type)
        elif isinstance(child, list):
            for nested in child:
                if isinstance(nested, dict):
                    node_type = nested.get("nodeType")
                    if node_type:
                        child_types.append(node_type)
    return child_types


def _collect_param_types(container: Optional[Dict[str, Any]]) -> Tuple[int, List[str]]:
    if not isinstance(container, dict):
        return 0, []
    raw_params = container.get("parameters", []) or []
    types: List[str] = []
    for param in raw_params:
        if not isinstance(param, dict):
            continue
        type_node = param.get("typeName")
        type_label = get_type_name(type_node)
        if not type_label:
            type_desc = param.get("typeDescriptions") or {}
            type_label = (
                type_desc.get("typeString") or type_desc.get("typeIdentifier") or ""
            )
        if type_label:
            types.append(type_label)
    return len(raw_params), types


def _collect_ml_metadata(node: Dict[str, Any], children: List[Any]) -> Dict[str, Any]:
    """Assemble rich-yet-compact AST metadata for downstream ML tasks."""

    def _assign(target: Dict[str, Any], key: str, value: Any):
        if value is None:
            return
        if isinstance(value, str) and value == "":
            return
        if isinstance(value, (list, dict)) and not value:
            return
        target[key] = value

    metadata: Dict[str, Any] = {}
    _assign(metadata, "ast_id", node.get("id"))
    _assign(metadata, "kind", node.get("kind"))
    _assign(metadata, "state_mutability", node.get("stateMutability"))
    _assign(metadata, "visibility", node.get("visibility"))
    _assign(metadata, "storage_location", node.get("storageLocation"))
    _assign(metadata, "literal_value", node.get("value"))
    _assign(metadata, "hex_value", node.get("hexValue"))
    _assign(metadata, "literal", node.get("literal"))
    _assign(metadata, "operator", node.get("operator"))
    _assign(metadata, "function_selector", node.get("functionSelector"))
    _assign(metadata, "payable", node.get("payable"))
    _assign(metadata, "mutability", node.get("mutability"))
    _assign(metadata, "virtual", node.get("virtual"))
    _assign(metadata, "override", node.get("override"))
    _assign(metadata, "constant", node.get("constant"))
    _assign(metadata, "immutable", node.get("immutable"))
    _assign(metadata, "state_variable", node.get("stateVariable"))
    _assign(metadata, "inline_assembly", node.get("inlineAssembly"))
    _assign(metadata, "abstract", node.get("abstract"))
    
    documentation = node.get("documentation")
    if isinstance(documentation, dict):
        doc_text = documentation.get("text") or documentation.get("value")
    else:
        doc_text = documentation
    _assign(metadata, "documentation", doc_text)

    type_desc = node.get("typeDescriptions")
    if isinstance(type_desc, dict):
        td = {
            "type_identifier": type_desc.get("typeIdentifier"),
            "type_string": type_desc.get("typeString"),
        }
        if td["type_identifier"] or td["type_string"]:
            metadata["type_descriptions"] = td

    param_count, param_types = _collect_param_types(node.get("parameters"))
    _assign(metadata, "parameter_count", param_count if param_count else None)
    if param_types:
        metadata["parameter_types"] = param_types

    return_count, return_types = _collect_param_types(node.get("returnParameters"))
    _assign(metadata, "return_count", return_count if return_count else None)
    if return_types:
        metadata["return_types"] = return_types
    if return_count:
        metadata["has_returns"] = True

    modifiers = []
    for modifier in node.get("modifiers", []) or []:
        if not isinstance(modifier, dict):
            continue
        mod_name = modifier.get("modifierName")
        if isinstance(mod_name, dict):
            name_value = mod_name.get("name") or mod_name.get("pathNode")
        else:
            name_value = mod_name
        if name_value:
            modifiers.append(name_value)
    if modifiers:
        metadata["modifiers"] = modifiers

    base_contracts = []
    for base in node.get("baseContracts", []) or []:
        if not isinstance(base, dict):
            continue
        base_name = base.get("baseName")
        if isinstance(base_name, dict):
            name_value = base_name.get("name") or base_name.get("path")
        else:
            name_value = base_name
        if name_value:
            base_contracts.append(name_value)
    if base_contracts:
        metadata["base_contracts"] = base_contracts

    child_types = _flatten_child_types(children)
    _assign(metadata, "child_count", len(child_types) if child_types else None)
    if child_types:
        metadata["child_types"] = child_types

    statements = node.get("statements")
    if isinstance(statements, list):
        _assign(metadata, "statement_count", len(statements))

    body = node.get("body")
    if isinstance(body, dict):
        _assign(metadata, "body_type", body.get("nodeType"))
        metadata["has_body"] = True
    elif body is not None:
        metadata["has_body"] = bool(body)

    expression = node.get("expression")
    if isinstance(expression, dict):
        _assign(metadata, "expression_type", expression.get("nodeType"))

    arguments = node.get("arguments")
    if isinstance(arguments, list):
        _assign(metadata, "argument_count", len(arguments))

    return metadata


# ==========================================
# 2. Core Logic: Determine Graph Structure
# ==========================================
def get_node_label_and_children(node):
    """
    Determines the label for the graph node and which children to recurse into.
    Logic:
    - High-level containers (File, Contract, Function, Block): Recurse.
    - Control Flow (If, For, While): Recurse into bodies, summarize condition in label.
    - Statements (Expression, Return, Emit): Stop recursion, summarize code in label.
    """
    node_type = node.get("nodeType", "Unknown")
    label = node_type
    children = []

    # --- 1. High Level Containers (Recurse) ---
    if node_type == "SourceUnit":
        short_name = clean_file_name(node.get("absolutePath", ""))
        label = f"File: {short_name}"
        children = node.get("nodes", [])

    elif node_type == "ContractDefinition":
        label = f"Contract: {node.get('name')}"
        children = node.get("nodes", [])

    elif node_type == "FunctionDefinition" or node_type == "ModifierDefinition":
        name = node.get("name", "")
        if not name or name == "":
            name = node.get("kind", "unnamed")
        # Reconstruct parameters for the label
        params = node.get("parameters", {}).get("parameters", [])
        param_str = ",".join([reconstruct_code(p) for p in params])
        label = f"{name}({param_str})"

        # Only recurse into the body block
        if "body" in node and node["body"]:
            children.append(node["body"])

    elif node_type == "ModifierDefinition":
        name = node.get("name", "")
        label = f"Modifier: {name}"
        if "body" in node and node["body"]:
            children.append(node["body"])

    elif node_type == "Block":
        label = "Block"
        children = node.get("statements", [])

    # --- 2. Control Flow (Partial Recurse) ---
    elif node_type == "IfStatement":
        cond = reconstruct_code(node.get("condition"))
        label = f"If ({cond})"
        if "trueBody" in node and node["trueBody"]:
            children.append(node["trueBody"])
        if "falseBody" in node and node["falseBody"]:
            children.append(node["falseBody"])

    elif node_type == "ForStatement":
        init = reconstruct_code(node.get("initializationExpression"))
        cond = reconstruct_code(node.get("condition"))
        loop = reconstruct_code(node.get("loopExpression"))
        label = f"For({init}; {cond}; {loop})"
        if "body" in node and node["body"]:
            children.append(node["body"])

    elif node_type == "WhileStatement":
        cond = reconstruct_code(node.get("condition"))
        label = f"While ({cond})"
        if "body" in node and node["body"]:
            children.append(node["body"])

    # --- 3. Line-Level Statements (Stop Recursion & Summarize) ---
    elif node_type == "ExpressionStatement":
        # e.g., "count++;" or "require(x>0);"
        label = reconstruct_code(node.get("expression"))
        # No children added -> recursion stops here

    elif node_type == "VariableDeclarationStatement":
        # e.g., "uint x = 10;"
        decls = node.get("declarations", [])
        decl_strs = []
        for d in decls:
            if d:
                decl_strs.append(reconstruct_code(d))
            else:
                decl_strs.append("_")  # tuple unpacking empty slot

        left = ", ".join(decl_strs)
        if "initialValue" in node and node["initialValue"]:
            right = reconstruct_code(node.get("initialValue"))
            label = f"{left} = {right}"
        else:
            label = f"{left}"

    elif node_type == "Return":
        expr = reconstruct_code(node.get("expression"))
        label = f"Return {expr}"

    elif node_type == "EmitStatement":
        event = reconstruct_code(node.get("eventCall"))
        label = f"Emit {event}"

    elif node_type == "RevertStatement":
        # Older solidity uses FunctionCall for revert, newer has dedicated node
        label = "Revert"

    elif node_type == "VariableDeclaration":
        # Matches State Variables (children of Contract)
        label = reconstruct_code(node)

    elif node_type == "EventDefinition":
        name = node.get("name")
        label = f"Event {name}"

    elif node_type == "PragmaDirective":
        label = f"Pragma {', '.join(node.get('literals', []))}"

    elif node_type == "ImportDirective":
        label = f"Import {node.get('file', '')}"

    elif node_type == "UsingForDirective":
        label = f"Using For"

    # --- 4. Fallback ---
    else:
        # If we encounter something else (e.g. inside a list we iterated blindly),
        # try to find children to keep tree connected.
        if "nodes" in node:
            children.extend(node["nodes"])
        if "statements" in node:
            children.extend(node["statements"])

    return label, children


# ==========================================
# 3. BUILD AST GRAPH
# ==========================================


def build_ast_graph_line_level(ast_data):
    G = nx.DiGraph()
    node_mapping = {}  # The Master Index
    file_to_node = {}  # CleanFileName -> node_id

    # Global Maps
    global_contract_funcs = {}  # ContractName -> { func_sig: node_id }
    global_inheritance = {}  # ContractName -> [ParentNames]
    file_imports = {}  # CleanFileName -> Set(ImportedCleanFileNames)
    file_contracts = {}  # CleanFileName -> [DefinedContractNames]
    contract_to_file = {}  # ContractName -> CleanFileName

    node_id_counter = 0

    # --- Pass 1: Parsing & Basic Indexing ---
    def traverse(node, parent_id=None, context=None):
        nonlocal node_id_counter

        if not isinstance(node, dict) or "id" not in node:
            return

        node_id_counter += 1
        current_id = node_id_counter
        node_type = node.get("nodeType", "Unknown")
        label, children = get_node_label_and_children(node)

        if context is None:
            context = {"clean_file": "", "contract": "", "contract_kind": "none"}
        else:
            context = context.copy()
            context.setdefault("contract_kind", "none")
        # --- File & Contract Context ---
        if node_type == "SourceUnit":
            abs_path = node.get("absolutePath", "unknown")
            clean_name = clean_file_name(abs_path)
            context["clean_file"] = clean_name

            if clean_name not in file_contracts:
                file_contracts[clean_name] = []
                file_imports[clean_name] = set()

            file_to_node[clean_name] = current_id

        # Define ignored node types
        ignored_node_types = ["PragmaDirective"]

        if node_type in ignored_node_types:
            # Skip adding to graph, but process children with updated context
            if node_type == "ImportDirective":
                imp_path = node.get("absolutePath")
                if imp_path:
                    imp_clean = clean_file_name(imp_path)
                    # Name matching (endwith) - add edge if imp_clean ends with something, but for now, if in file_to_node
                    if imp_clean in file_to_node:
                        G.add_edge(
                            parent_id,
                            file_to_node[imp_clean],
                            **{f"{GraphAttributes.LABEL}": "import"},
                        )
                    # Keep for later logic
                    file_imports[context["clean_file"]].add(imp_clean)

            for child in children:
                if isinstance(child, dict):
                    traverse(child, parent_id, context)
                elif isinstance(child, list):
                    for item in child:
                        traverse(item, parent_id, context)
            return

        elif node_type == "ContractDefinition":
            c_name = node.get("name", "unknown")
            context["contract"] = c_name
            c_kind = node.get("contractKind", "") or "contract"
            if node.get("abstract"):
                c_kind = "abstract"
            context["contract_kind"] = c_kind

            file_contracts[context["clean_file"]].append(c_name)
            contract_to_file[c_name] = context["clean_file"]

            # Record Inheritance
            parents = []
            for base in node.get("baseContracts", []):
                base_name = base.get("baseName", {}).get("name")
                if base_name:
                    parents.append(base_name)
            global_inheritance[c_name] = parents

            if c_name not in global_contract_funcs:
                global_contract_funcs[c_name] = {}

        # --- Function Indexing ---
        if node_type == "FunctionDefinition" or node_type == "ModifierDefinition":
            # Name Resolution
            if node_type == "ModifierDefinition":
                func_name = node.get("name", "unnamed")
            else:
                kind = node.get("kind", "")
                # Crucial: CFG uses 'constructor', 'fallback' as names
                if kind in ["constructor", "fallback", "receive"]:
                    func_name = kind
                else:
                    func_name = node.get("name", "unnamed")

            # Arg Resolution
            params = node.get("parameters", {}).get("parameters", [])
            param_types = [get_type_name(p.get("typeName")) for p in params]
            signature_args = ",".join(param_types)
            func_sig = f"{func_name}{SEPARATOR}({signature_args})"

            # 1. Primary Key: File$Contract$Func
            key = f"{context['clean_file']}{SEPARATOR}{context['contract']}{SEPARATOR}{func_sig}"
            node_mapping[key] = current_id

            # Store for Propagation
            global_contract_funcs[context["contract"]][func_sig] = current_id

            context["function"] = f"{func_name}({signature_args})"

        _node_data = {}
        _node_data[GraphAttributes.NODE_TYPE] = NodeTypes.AST_NODE
        _node_data[GraphAttributes.SUB_NODE_TYPE] = node_type
        _node_data[GraphAttributes.LABEL] = label
        _node_data[GraphAttributes.FILE] = context.get("clean_file", "")
        _node_data[GraphAttributes.CONTRACT] = context.get("contract", "")
        _node_data["contract_kind"] = context.get("contract_kind", "none")
        _node_data[GraphAttributes.FUNCTION] = context.get("function", "")
        _node_data[GraphAttributes.SRC] = node.get("src", "")
        _node_data.update(_collect_ml_metadata(node, children))

        # --- Graph Nodes ---
        G.add_node(current_id, **_node_data)
        if parent_id is not None:
            G.add_edge(
                parent_id,
                current_id,
                **{
                    f"{GraphAttributes.LABEL}": EdgeTypes.AST_CHILD,
                    GraphAttributes.EDGE_TYPE: EdgeTypes.AST_CHILD,
                },
            )

        for child in children:
            if isinstance(child, dict):
                traverse(child, current_id, context)
            elif isinstance(child, list):
                for item in child:
                    traverse(item, current_id, context)

    # Execute Pass 1
    for filename, source_unit in ast_data.items():
        traverse(source_unit)

    # --- Pre-Computation: Transitive Inheritance & Imports ---

    # 1. Transitive Inheritance (Ancestors)
    all_ancestors = {}  # Contract -> Set(Ancestors)

    def get_ancestors(c_name):
        if c_name in all_ancestors:
            return all_ancestors[c_name]
        parents = global_inheritance.get(c_name, [])
        ancestors = set(parents)
        for p in parents:
            ancestors.update(get_ancestors(p))
        all_ancestors[c_name] = ancestors
        return ancestors

    for c in global_inheritance:
        get_ancestors(c)

    # 2. Transitive Imports
    all_visible_files = {}  # FileName -> Set(VisibleFileNames)

    def get_visible_files(f_name, visited=None):
        if visited is None:
            visited = set()
        if f_name in visited:
            return set()
        visited.add(f_name)

        direct = file_imports.get(f_name, set())
        visible = set(direct)
        for imp in direct:
            visible.update(get_visible_files(imp, visited))
        return visible

    for f in file_imports:
        all_visible_files[f] = get_visible_files(f)

    # --- Pass 2: Advanced Propagation ---

    # Iterate over every physical file context
    for current_file, defined_contracts in file_contracts.items():

        # A. Resolve Inheritance for Contracts in this File
        for c_name in defined_contracts:
            ancestors = all_ancestors.get(c_name, set())

            # 1. Propagate Grandparent functions to Child (A inherits C)
            for ancestor in ancestors:
                if ancestor in global_contract_funcs:
                    for func_sig, nid in global_contract_funcs[ancestor].items():

                        # Link 1: ChildFile$Child$AncestorFunc (Inherited)
                        k1 = f"{current_file}{SEPARATOR}{c_name}{SEPARATOR}{func_sig}"
                        if k1 not in node_mapping:
                            node_mapping[k1] = nid

                        # Link 2: ChildFile$Ancestor$AncestorFunc (Super Calls)
                        k2 = f"{current_file}{SEPARATOR}{ancestor}{SEPARATOR}{func_sig}"
                        if k2 not in node_mapping:
                            node_mapping[k2] = nid

            # 2. Intermediate Parent Aliasing (Fix for ERC20._msgData)
            # If A inherits B, and B inherits C. We need A's file to map B$C_Func -> C_Node
            for parent in ancestors:
                grandparents = all_ancestors.get(parent, set())
                for gp in grandparents:
                    if gp in global_contract_funcs:
                        for func_sig, nid in global_contract_funcs[gp].items():
                            # Link 3: ChildFile$Parent$GrandparentFunc
                            # e.g. ADaiTokenWrapper.sol$ERC20$_msgData -> Context._msgData
                            k3 = f"{current_file}{SEPARATOR}{parent}{SEPARATOR}{func_sig}"
                            if k3 not in node_mapping:
                                node_mapping[k3] = nid

        # B. Resolve Library/External Imports
        # If File A imports B, A sees B's contracts.
        visible_files = all_visible_files.get(current_file, set())
        for imp_file in visible_files:
            for imp_contract in file_contracts.get(imp_file, []):
                if imp_contract in global_contract_funcs:
                    for func_sig, nid in global_contract_funcs[imp_contract].items():
                        # Link 4: CurrentFile$ImportedContract$Func
                        # e.g. GoodGhostingBatched.sol$MerkleProof$verify
                        k4 = f"{current_file}{SEPARATOR}{imp_contract}{SEPARATOR}{func_sig}"
                        if k4 not in node_mapping:
                            node_mapping[k4] = nid

    # # Output
    # try:
    #     os.makedirs("test_output", exist_ok=True)
    #     write_dot(G, "test_output/ast_graph.dot")
    # except:
    #     pass

    # with open("test_output/ast_function_index.json", "w") as f:
    #     json.dump(node_mapping, f, indent=2)

    return G, node_mapping


def format_ast(raw_ast_jsons: dict) -> Dict[str, Any]:
    """
    Format for better AST processing [DONE]
    """
    results = {}
    for file_path, data in raw_ast_jsons.items():
        if isinstance(data, dict) and data.get("nodeType") == "SourceUnit":
            source_file_path = data.get("absolutePath", file_path)
            if source_file_path not in results:
                results[source_file_path] = data
        else:
            # If not SourceUnit, use the file_path as key
            if file_path not in results:
                results[file_path] = data
    return results


def load_ast(ast_root: Path):
    print("\nLoading AST data...")
    raw_ast_data = {}
    if ast_root.exists():
        for ast_file in ast_root.glob("*_ast.json"):
            try:
                with open(ast_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "nodeType" in data and data["nodeType"] == "SourceUnit":
                        # It's a single SourceUnit. Wrap it.
                        path = data.get("absolutePath", str(ast_file))
                        raw_ast_data[path] = data
                    else:
                        raw_ast_data.update(data)
            except Exception as e:
                print(f"Failed to load AST file {ast_file}: {e}")
    print(f"Loaded raw AST data from {len(raw_ast_data)} files")
    ast_data = format_ast(raw_ast_data)

    # json.dump(
    #     ast_data, open("test_output/processed_ast.json", "w"), indent=2
    # )
    return ast_data


def load_and_process_ast(ast_root: Path):
    raw_ast_data = load_ast(ast_root)
    ast_graph, ast_function_index = build_ast_graph_line_level(raw_ast_data)
    return ast_graph, ast_function_index
