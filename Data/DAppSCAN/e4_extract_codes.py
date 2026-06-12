import argparse
import hashlib
import json
import logging
import pickle
import time
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import traceback
from typing import Any, Dict, List, Tuple, Optional, Set, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import csv
import pandas as pd
import networkx as nx

from c2_build_CPG_modules.c_10Linking import build_cpg

try:
    import orjson as _orjson
except ImportError:
    _orjson = None

import re
from itertools import chain
import random

# Local imports
# NOTE: The following imports are commented out because the files are missing or paths are incorrect.
# from Data.DAppSCAN.e_2build_CPG import build_cpg
# from e_2enrich_CG import process_project_CG
# from e_1enrich_CFG import process_project_CFG
# from e_4generate_enrichDFG import generate_dfg_from_ast
# from e_3generate_enrichAST import generate_ast_graphs
# from e_6generate_enrich_Bytecode_CFG import process_contract_bytecodes, combine_bcfgs, validate_dependencies

# from e_7extract_code import extract_code_with_vulnerabilities
from the_utils.logger import setup_logger

from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILENAME = f"Logs/DataProc_Phase2_run_{RUN_TIMESTAMP}.log"

CURRENT_DIR = Path(__file__).resolve().parent
EXTRACTED_GRAPHS_DIR = CURRENT_DIR / "Extracted_Graphs"
WORKING_GRAPH_DIR = CURRENT_DIR / "collected_graphs"
OUTPUT_DIR = CURRENT_DIR / "ProcessedData"

# Ensure directories exist
WORKING_GRAPH_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
SUCCESS_OUTPUT_DIR = OUTPUT_DIR / "success"
FAILED_OUTPUT_DIR = OUTPUT_DIR / "failed"
BENIGN_OUTPUT_DIR = OUTPUT_DIR / "benign"
STAGING_OUTPUT_DIR = OUTPUT_DIR / "_staging"
for folder in (SUCCESS_OUTPUT_DIR, FAILED_OUTPUT_DIR, BENIGN_OUTPUT_DIR, STAGING_OUTPUT_DIR):
    folder.mkdir(exist_ok=True)

# Setup centralized logger
logger = setup_logger(LOG_FILENAME, log_level=logging.INFO)

# ----------------------
# Constants
# ----------------------
SOURCE_CODE_VULN_FOLDER = Path("DAppSCAN-source/SWCsource")
BYTECODE_VULN_FOLDER = Path("DAppSCAN-bytecode/SWCbytecode")
PROJECT_JSON_FOLDER = Path("project_json")
SWC_OWASP_CSV = Path("SWC_OWASP.csv")
SOURCE_DIR = CURRENT_DIR / "DAppSCAN-source" / "contracts"

def print_info(message):
    """Print information messages in white (default color)"""
    logger.info(message)

def print_error(message):
    """Print error messages in red"""
    logger.error(f"[RED]{message}")


def print_processing(message):
    """Print in-processing messages in yellow"""
    logger.info(f"[YELLOW]{message}")

def compact_code(code: str) -> str:
    # lines = code.split('\n')
    # stripped = [line.lstrip() for line in lines]
    # return '\n'.join(stripped).strip()
    return code

def _fetch_code_raw_from_vuln_entry(vuln_entry: Dict[str, Any]):
    project = vuln_entry["project"]
    file_partial = vuln_entry["file"]
    category = vuln_entry["category"]
    swc_id = vuln_entry["swc_id"]
    line_from = vuln_entry["line_from"].strip()
    line_to = vuln_entry["line_to"].strip()

    # Find the file
    file_path = SOURCE_DIR / project / file_partial
    if not file_path.exists():
        print_error(f"File not found: {file_path}")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print_error(f"Failed to read file {file_path}: {e}")
        return

    try:
        start = int(line_from)
        end = int(line_to)
        valid_lines = True
    except ValueError:
        valid_lines = False
    
    if valid_lines:
        try:
            start = start - 1  # 0-based
            end = end  # exclusive
            code = "".join(lines[start:end])
            # Check for category comment nearby (within 5 lines before and after)
            nearby_start = max(0, start - 5)
            nearby_end = min(len(lines), end + 5)
            nearby_lines = lines[nearby_start:nearby_end]
            comment_found = any(swc_id in line for line in nearby_lines)
            if comment_found:
                print_info(
                    f"Category comment '{swc_id}' found near lines {line_from}-{line_to} in {file_path}"
                )
            else:
                print_error(
                    f"Category comment '{swc_id}' not found near lines {line_from}-{line_to} in {file_path}"
                )
            vuln_entry["snippet"] = compact_code(code)
        except (ValueError, IndexError):
            print_error(f"Invalid line numbers: {line_from}-{line_to}")
    else:
        # Find the category comment
        comment_line = None
        for i, line in enumerate(lines):
            if swc_id in line:
                comment_line = i
                break

        if comment_line is None:
            print_error(f"Category comment '{swc_id}' not found in {file_path}")
            return

        # Find the function below the comment
        func_start = None
        for i in range(comment_line, len(lines)):
            line = lines[i].strip()
            if line.startswith("function ") or "function(" in line:
                func_start = i
                break
        if func_start is None:
            # Try above the comment
            for i in range(comment_line - 1, -1, -1):
                line = lines[i].strip()
                if line.startswith("function ") or "function(" in line:
                    func_start = i
                    break
        if func_start is None:
            print_error(f"Function not found near comment in {file_path}")
            return

        # Extract the function by matching braces
        brace_count = 0
        start_found = False
        func_lines = []
        for i in range(func_start, len(lines)):
            line = lines[i]
            func_lines.append(line)
            for char in line:
                if char == "{":
                    brace_count += 1
                    start_found = True
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0 and start_found:
                        break
            if brace_count == 0 and start_found:
                break
        code = "".join(func_lines)
        vuln_entry["snippet"] = compact_code(code)
        print_info(f"Extracted function from {file_path}")


def _load_vuln_json(_vuln_folder: Path) -> Dict[str, Any]:
    """Get vulnerability JSONs from the specified folder."""
    vuln_data = {}
    if not _vuln_folder.exists():
        # print_error(f"Vulnerability folder does not exist: {_vuln_folder}")
        return vuln_data
    json_files = list(_vuln_folder.rglob("*.json"))
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            vuln_data[jf.name] = data
        except Exception as e:
            pass #print_error(f"Failed to load vulnerability JSON {jf}: {e}")
    print_info(f"Loaded {len(vuln_data)} vulnerability JSON files from {_vuln_folder}")
    return vuln_data


def _swc_parsing(swc_str: str):
    """Parse SWC string."""
    if not swc_str:
        return "", ""
    parts = swc_str.split("-")
    swc_id = parts[0] + "-" + parts[1] if len(parts) >= 2 else swc_str
    name = parts[2] if len(parts) >= 3 else ""
    return swc_id, name


def _load_swc_owasp_mapping(csv_path: Path) -> Optional[Dict[str, str]]:
    """Load SWC->OWASP mapping and return dict of SWC_ID -> OWASP_ID, excluding rows with check=='x'.

    If file missing or unreadable, returns None to indicate no filtering.
    """
    try:
        if not csv_path.exists():
            print_info(
                f"SWC_OWASP mapping not found at {csv_path} (no filtering will be applied)"
            )
            return None
        df = pd.read_csv(csv_path, encoding="utf-8")
        # Remove rows where check is 'x' (case-insensitive, allow int 1 for keep)
        df = df[~df["check"].astype(str).str.lower().eq("x")]
        mapping: Dict[str, str] = {
            str(row["swc_id"]).strip().upper(): str(row["owasp_id"]).strip().upper()
            for _, row in df.iterrows()
        }
        print_info(f"Loaded {len(mapping)} SWC->OWASP mappings from {csv_path}")
        return mapping if mapping else None
    except Exception as e:
        print_error(f"Failed to load SWC_OWASP mapping: {e}")
        return None


def _get_func_line_vuln(
    project_name: str,
    SC_vuln_json: Dict[str, Any],
    BC_vuln_json: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        swc_owasp_map = _load_swc_owasp_mapping(SWC_OWASP_CSV)
        line_vuln = {}
        if len(SC_vuln_json) == 0:
            return line_vuln
        counter = 0
        for json_name, data in SC_vuln_json.items():
            file_path = data.get("filePath", "")
            swc_list = data.get("SWCs", [])
            if file_path and swc_list:
                for item in swc_list:
                    counter += 1
                    swc_id, name = _swc_parsing(str(item.get("category", "")))
                    owasp_id = (
                        swc_owasp_map.get(swc_id.upper())
                        if swc_owasp_map and swc_id
                        else None
                    )
                    if swc_owasp_map is not None and owasp_id is None:
                        continue
                    # Be resilient to ints, None, or strings for lineNumber
                    line_number_raw = item.get("lineNumber", "")
                    line_str = str(line_number_raw).strip().replace("L", "").replace("l", "")
                    lines = line_str.split("-")
                    if len(lines) == 2:
                        line_from = lines[0].strip()
                        line_to = lines[1].strip()
                    else:
                        line_from = lines[0].strip()
                        line_to = lines[0].strip()
                        if len(line_from) > 7:
                            print(
                                f"[INVALID LINE NUMBER] {line_from} in project {project_name}, file {file_path}"
                            )
                            line_from = "invalid"
                            line_to = "invalid"

                    if swc_id:
                        _data = {
                            "project": project_name,
                            "file": "/".join(file_path.split("/")[3:]),
                            "function": item.get("function", ""),
                            "category": item.get("category", ""),
                            "swc_id": swc_id,
                            "owasp_id": owasp_id,
                            "swc_name": name,
                            "line_from": line_from,
                            "line_to": line_to,
                            "detect_type": "source_code",
                        }
                        _fetch_code_raw_from_vuln_entry(_data)
                        key = f"{_data['file']}${_data['function']}${_data['swc_id']}${_data['owasp_id']}${_data['line_from']}${_data['line_to']}${str(counter)}"
                        line_vuln[key] = _data

        if len(BC_vuln_json) == 0:
            return line_vuln

        for json_name, data in BC_vuln_json.items():
            swc_list = data.get("SWCs", [])
            for item in swc_list:
                counter += 1
                swc_id, name = _swc_parsing(str(item.get("category", "")))
                owasp_id = (
                    swc_owasp_map.get(swc_id.upper())
                    if swc_owasp_map and swc_id
                    else None
                )
                if swc_owasp_map is not None and owasp_id is None:
                    continue
                line_number_raw = item.get("lines", "")
                line_str = str(line_number_raw).strip().replace("L", "").replace("l", "")
                lines = line_str.split("-")
                if len(lines) == 2:
                    line_from = lines[0].strip()
                    line_to = lines[1].strip()
                else:
                    line_from = lines[0].strip()
                    line_to = lines[0].strip()
                    if len(line_from) > 7:
                        print(
                            f"[INVALID LINE NUMBER] {line_from} in project {project_name}, file {item.get('sourcePath', '')}"
                        )
                        line_from = "invalid"
                        line_to = "invalid"

                if swc_id:
                    _data = {
                        "project": project_name,
                        "file": "/".join(
                            str(item.get("sourcePath", "")).split("/")[-3:]
                        ),
                        "function": item.get("function", ""),
                        "category": item.get("category", ""),
                        "swc_id": swc_id,
                        "owasp_id": owasp_id,
                        "swc_name": name,
                        "line_from": line_from,
                        "line_to": line_to,
                        "detect_type": "byte_code",
                    }
                    _fetch_code_raw_from_vuln_entry(_data)
                    key = f"{_data['file']}${_data['function']}${_data['swc_id']}${_data['owasp_id']}${_data['line_from']}${_data['line_to']}${str(counter)}"
                    line_vuln[key] = _data

        return line_vuln
    except Exception as e:
        traceback.print_exc()
        return {}


def _classify_build_stats(stats: Optional[Dict[str, int]]) -> str:
    if not stats:
        return "failed"
    has_cfg_graph = stats.get("cfg_node_count", 0) > 0 and stats.get("cfg_edge_count", 0) > 0
    has_ast_graph = stats.get("ast_node_count", 0) > 0
    has_vuln_match = stats.get("vuln_matched_node_count", 0) > 0
    if has_cfg_graph and has_ast_graph and has_vuln_match:
        return "success"
    if not has_cfg_graph or not has_ast_graph:
        return "failed"
    return "benign"


def _relocate_payload(src: Path, category: str, project_name: str) -> Path:
    if category == "success":
        target_root = SUCCESS_OUTPUT_DIR / project_name
    elif category == "failed":
        target_root = FAILED_OUTPUT_DIR / project_name
    else:
        target_root = BENIGN_OUTPUT_DIR / project_name

    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.parent.mkdir(parents=True, exist_ok=True)

    if src.exists():
        shutil.move(str(src), str(target_root))

    return target_root


def _write_project_log(root: Path, project_name: str, category: str, stats: Optional[Dict[str, int]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = [
        f"Project: {project_name}",
        f"Category: {category}",
        f"Timestamp: {datetime.utcnow().isoformat()}Z",
    ]
    if stats:
        for key in sorted(stats.keys()):
            value = stats[key]
            log_lines.append(f"{key}: {value}")
    else:
        log_lines.append("No build statistics available.")

    (root / "build_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")


# region process_project()
def process_project_folder(
    project_folder: Path,
    workers: int = 8,
    do_steps: Optional[Set[str]] = None,
    to_pydot: bool = False,
):
    try:
        print_processing(f"Processing project {project_folder}, to pydot={to_pydot}")
        project_name = project_folder.name
        print_info(f"Project name: {project_name}")
        compiled_root = STAGING_OUTPUT_DIR / project_name
        # Use local variables for per-project folders so we do not mutate module-level constants
        sc_vuln_folder = SOURCE_CODE_VULN_FOLDER / project_name
        bc_vuln_folder = BYTECODE_VULN_FOLDER / project_name
        print_info(f"Expected Source Code Vulnerability folder: {sc_vuln_folder}")
        print_info(f"Expected Bytecode Vulnerability folder: {bc_vuln_folder}")

        SC_vuln_json = _load_vuln_json(sc_vuln_folder)
        BC_vuln_json = _load_vuln_json(bc_vuln_folder)

        proj_line_func_level_vuln = _get_func_line_vuln(
            project_name, SC_vuln_json, BC_vuln_json
        )
        print_processing("Function/Line level vulnerabilities:")

        # Pretty print one vulnerability per line
        if not proj_line_func_level_vuln:
            print_processing("  none")
            benign_root = BENIGN_OUTPUT_DIR / project_name
            if benign_root.exists():
                shutil.rmtree(benign_root)
            benign_root.mkdir(parents=True, exist_ok=True)
            (benign_root / "line_level_vulnerabilities.json").write_text(
                json.dumps({}, indent=2)
            )
            _write_project_log(benign_root, project_name, "benign", None)
            print_info(f"No vulnerabilities found; saved placeholder under {benign_root}")
            return None
        else:
            for k, item in proj_line_func_level_vuln.items():
                file = item.get("file") or "N/A"
                func = item.get("function") or "N/A"
                swc_id = item.get("swc_id") or ""
                swc_name = item.get("swc_name") or ""
                owasp = item.get("owasp_id") or ""
                line_from = item.get("line_from")
                line_to = item.get("line_to")

                line_info = f"{line_from}-{line_to}"
                msg = f"- {file} fn: {func} : {line_info} | {swc_id} [{owasp}] ({swc_name})"
                print_processing(f"\t{msg}")

        print_info(f"Writing compiled data to {compiled_root}")
        if compiled_root.exists():
            shutil.rmtree(compiled_root)
        os.makedirs(compiled_root, exist_ok=True)
        (compiled_root / "line_level_vulnerabilities.json").write_text(
            json.dumps(proj_line_func_level_vuln, indent=2)
        )

        ####################
        # region  CPG
        ###################
        build_stats: Optional[Dict[str, int]] = None
        if do_steps is None or "cpg" in do_steps:
            _, build_stats = build_cpg(project_folder, proj_line_func_level_vuln, compiled_root)

        category = _classify_build_stats(build_stats) if build_stats is not None else "benign"
        _write_project_log(compiled_root, project_name, category, build_stats)
        final_root = _relocate_payload(compiled_root, category, project_name)
        print_info(f"Stored compiled payload under {category}: {final_root}")
    except Exception as e:
        traceback.print_exc()


# region main()
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect and preprocess extracted graph projects"
    )
    parser.add_argument(
        "--checkpoint",
        default="analysis_checkpoint.json",
        help="path to checkpoint JSON",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing collected targets"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="process only one random project (quick test)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of worker threads for hashing/processing",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        default=False,
        help="skip copy to collect_graph dir",
    )
    parser.add_argument(
        "--do",
        type=str,
        default=None,
        help="Comma-separated list of steps to run (e.g. cg,cfg,dfg,ast,dd,dg,fs,flattened,extract,bytecode_cfg,vuln)",
    )
    parser.add_argument(
        "--to-pydot",
        action="store_true",
        help="output pydot files instead of .dot files where applicable",
    )
    args = parser.parse_args(argv)



    projects = [p for p in Path(EXTRACTED_GRAPHS_DIR).iterdir() if p.is_dir()]
    print_info(f"Found {len(projects)} projects to process")
    if args.test:
        if projects:
            chosen = random.choice(projects)
            print_info(
                f"--test provided: selecting single random project {chosen.name}"
            )
            projects = [chosen]
        else:
            print_info("--test provided but no projects found to select from")

    do_steps = None
    if args.do:
        do_steps = set(s.strip().lower() for s in args.do.split(",") if s.strip())

    for i, p in enumerate(projects):
        try:
            print_info(f"Processing project {i+1}/{len(projects)}: {p.name}")
            process_project_folder(
                p, workers=args.workers, do_steps=do_steps, to_pydot=args.to_pydot
            )
        except Exception as e:
            print_error(f"Failed to process {p}: {e}")

    return 0


if __name__ == "__main__":
    main()
