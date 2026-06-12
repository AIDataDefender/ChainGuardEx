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
import re
import random

# from e_7extract_code import extract_code_with_vulnerabilities
from the_utils.logger import setup_logger

from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

SOURCE_CODE_ROOT = [
    #r'source_code/access_control/clean_57_buggy_curated_0',
    #r'source_code/arithmetic/clean_60_buggy_curated_0',
    #r'source_code/denial_of_service/clean_46_buggy_curated_0',
    r'source_code/reentrancy/clean_71_buggy_curated_0',
    #r'source_code/unchecked_low_level_calls/clean_95_buggy_curated_0',
    
]

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILENAME = f"Logs/DataProc_Phase2_run_{RUN_TIMESTAMP}.log"

CURRENT_DIR = Path(__file__).parent.resolve()
EXTRACTED_GRAPHS_DIR = CURRENT_DIR / "Extracted_Graphs"
OUTPUT_DIR = CURRENT_DIR / "test_data" # or "ProcessedData" as needed
RENAMED_SOL_DIR = CURRENT_DIR / "Renamed_SOL_hashed"
# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR = CURRENT_DIR / "Logs"
LOG_DIR.mkdir(exist_ok=True)
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
SOURCE_CODE_VULN_FOLDER = Path("all_compiled_vulnerabilities.json")
LABELS_DIR = Path("labels")

def compile_vulnerabilities(labels_dir: Path, output_file: Path) -> None:
    """Compile all vulnerability JSONs from labels/ into a single curated format."""
    category_mapping = {
        "AC_vulnerabilities.json": "access_control",
        "ARITH_vulnerabilities.json": "arithmetic",
        "DOS_vulnerabilities.json": "denial_of_service",
        "LC_vulnerabilities.json": "unchecked_low_level_calls",
        "RE_vulnerabilities.json": "reentrancy",
        "additional_vulnerabilities.json": None,  # Has category in each vuln
    }
    compiled = {}
    renamed_sol_dir = RENAMED_SOL_DIR
    renamed_sol_dir.mkdir(exist_ok=True)
    
    # First, scan all .sol files in SOURCE_CODE_ROOT, compute hashes, and create mapping
    name_to_hashed = {}
    for root in SOURCE_CODE_ROOT:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for sol_file in root_path.glob("*.sol"):
            hash_value=""
            with open(sol_file, "rb") as f:
                hash_value = hashlib.md5(f.read()).hexdigest()

            if 'arithmetic' in sol_file.parts:
                cat = 'arithmetic'
            elif 'access_control' in sol_file.parts:
                cat = 'access_control'
            elif 'denial_of_service' in sol_file.parts:
                cat = 'denial_of_service'
            elif 'unchecked_low_level_calls' in sol_file.parts:
                cat = 'unchecked_low_level_calls'
            elif 'reentrancy' in sol_file.parts:
                cat = 'reentrancy'
            else:
                cat = 'unknown'
                print_error(f"Unknown category for file: {sol_file}")
                
            original_name = sol_file.name
            name_without_ext = original_name.split(".sol")[0]
            hashed_name = f"{name_without_ext}${hash_value}.sol"
            name_to_hashed[f"{cat}_{original_name}"] = hashed_name
            # Copy to renamed_sol_dir
            shutil.copy(sol_file, renamed_sol_dir / hashed_name)
    
    print_info(f"Scanned and renamed {len(name_to_hashed)} files to {renamed_sol_dir}")
    #print(name_to_hashed)
    for file_name, implied_category in category_mapping.items():
        file_path = labels_dir / file_name
        if not file_path.exists():
            print_error(f"Labels file not found: {file_path}")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            contract_name = entry["name"]
            for vuln in entry["vulnerabilities"]:
                if implied_category:
                    category = implied_category
                else:
                    category = vuln.get("category")
                    if category not in [
                        "access_control",
                        "arithmetic",
                        "denial_of_service",
                        "unchecked_low_level_calls",
                        "reentrancy",
                    ]:
                        print_error(f"Unknown category {category} in {file_name}, {contract_name}")
                        continue
                    if not category:
                        print_error(f"No category for vuln in {file_name}, {contract_name}")
                        continue
                ori_hash_key = name_to_hashed.get(f"{category}_{contract_name}", name_to_hashed.get(f"{'unknown'}_{contract_name}", contract_name))
                #print_info(f"Processing {contract_name} as {ori_hash_key} from {file_name}")
                hash_key = ori_hash_key if not "$" in ori_hash_key else "_".join(ori_hash_key.split("$"))
                if hash_key not in compiled:
                    compiled[hash_key] = {}

                if category not in compiled[hash_key]:
                    compiled[hash_key][category] = []
                compiled[hash_key][category].extend(vuln["lines"])
                compiled[hash_key]["original_name"] = contract_name
                compiled[hash_key]["hash_key"] = ori_hash_key
    
    # Remove duplicates in lines
    for contract, cats in compiled.items():
        for cat, lines in cats.items():
            if cat in ["original_name", "hash_key"]:
                continue
            compiled[contract][cat] = sorted(list(set(lines)))
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(compiled, f, indent=2)
    print_info(f"Compiled vulnerabilities saved to {output_file}")
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
    if code.strip() == "}":
        return ""
    return code

def _fetch_code_raw_from_vuln_entry(vuln_entry: Dict[str, Any], file_path: Path) -> None:
    if not file_path.exists():
        print_error(f"File not found: {file_path}")
        vuln_entry["function"] = "N/A"
        vuln_entry["snippet"] = "N/A"
        return
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    valid_lines = True
    # Find the function containing line_from
    try:
        line_from = int(vuln_entry["line_from"])
        line_to = int(vuln_entry["line_to"])
        valid_lines = True
    except ValueError:
        valid_lines = False
    if valid_lines:
        start = line_from - 1  # 0-based
        end = line_to  # exclusive

        func_start = None
        func_name = "N/A"
        for i in range(start, -1, -1):
            line = lines[i].strip()
            if line.startswith("function ") or "function(" in line:
                func_start = i
                # extract name
                match = re.search(r'function\s+(\w+)', line)
                if match:
                    func_name = match.group(1)
                break
        if func_start is None:
            vuln_entry["function"] = "N/A"
            vuln_entry["snippet"] = "N/A"
            return

        code = "".join(lines[start:end])
        vuln_entry["function"] = func_name
        vuln_entry["snippet"] = compact_code(code)
    else:
        vuln_entry["function"] = "N/A"
        vuln_entry["snippet"] = "N/A"
        print_error(f"Not found code!")

def group_consecutive(lines: List[int]) -> List[Tuple[int, int]]:
    """Group consecutive integers into ranges."""
    if not lines:
        return []
    lines = sorted(set(lines))  # ensure sorted and unique
    ranges = []
    start = lines[0]
    prev = lines[0]
    for num in lines[1:]:
        if num == prev + 1:
            prev = num
        else:
            ranges.append((start, prev))
            start = num
            prev = num
    ranges.append((start, prev))
    return ranges

def _get_func_line_vuln(
    project_name: str,
    vuln_json: Dict[str, Any],
) -> Dict[str, Any]:
    category_mapping = {
        "access_control": "SC01",
        "arithmetic": "SC08",
        "reentrancy": "SC05",
        "unchecked_low_level_calls": "SC06",
        "time_manipulation": "N/A",
        "denial_of_service": "SC10",
        "front_running": "N/A",
    }
    line_vuln = {}
    counter = 0
    mapping_key = f"{project_name}.sol"
    # Filter vuln_json to only include the matching key
    if mapping_key and mapping_key in vuln_json.keys():
        filtered_vuln_json = {mapping_key: vuln_json[mapping_key]}
    else:
        filtered_vuln_json = {}
    for contract_file_w_hash, categories in filtered_vuln_json.items():
        for category, lines in categories.items():
            if category in ["original_name", "hash_key"]:
                continue
            ori_contract_name = categories.get("original_name", "N/A")
            hash_key = categories.get("hash_key", contract_file_w_hash)

            owasp_id = category_mapping.get(category, "N/A")
            if owasp_id == "N/A" or not owasp_id:
                continue
            # Find the correct source root for this category
            base_path = RENAMED_SOL_DIR
            path = base_path / hash_key
            filename = str(path)
        
            
            for start, end in group_consecutive(lines):
                _data = {
                    "project": project_name,
                    "file": ori_contract_name,  # Use base filename to match CFG
                    "file_full": hash_key,  # Keep full filename for reference
                    "old_hash_key": contract_file_w_hash,
                    "abs_path": path.resolve().as_posix(),
                    "function": "",
                    "category": category,
                    "swc_id": "N/A",
                    "owasp_id": owasp_id,
                    "swc_name": category,
                    "line_from": str(start),
                    "line_to": str(end),
                    "detect_type": "source_code",
                    "filename": filename,
                    "snippet": "",
                }
                _fetch_code_raw_from_vuln_entry(_data, Path(filename))
                # print(json.dumps(_data, indent=2))
                key = f"{hash_key}${category}${str(start)}${str(end)}${str(counter)}"
                line_vuln[key] = _data
                counter += 1
    return line_vuln


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
    do_steps: Optional[Set[str]] = None,
    to_pydot: bool = False,
    test_mode: bool = False,
):
    try:
        print_processing(f"Processing project {project_folder}, to pydot={to_pydot}")
        project_name = project_folder.name
        print_info(f"Project name: {project_name}")
        compiled_root = STAGING_OUTPUT_DIR / project_name
        # Use local variables for per-project folders so we do not mutate module-level constants
        sc_vuln_folder = SOURCE_CODE_VULN_FOLDER# / project_name
        #bc_vuln_folder = BYTECODE_VULN_FOLDER / project_name
        print_info(f"Expected Source Code Vulnerability folder: {sc_vuln_folder}")
        #print_info(f"Expected Bytecode Vulnerability folder: {bc_vuln_folder}")

        SC_vuln_json = json.load(open(sc_vuln_folder, "r", encoding="utf-8")) # _load_vuln_json(sc_vuln_folder)
        #BC_vuln_json = _load_vuln_json(bc_vuln_folder)

        proj_line_func_level_vuln = _get_func_line_vuln(
            project_name, SC_vuln_json
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
            _, build_stats = build_cpg(project_folder, proj_line_func_level_vuln, compiled_root, test_mode=test_mode)

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

    # Compile vulnerabilities if not already done
    if not SOURCE_CODE_VULN_FOLDER.exists():
        compile_vulnerabilities(LABELS_DIR, SOURCE_CODE_VULN_FOLDER)
        SOURCE_CODE_ROOT = [str(RENAMED_SOL_DIR)]

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
            process_project_folder(p, do_steps=do_steps, to_pydot=args.to_pydot, test_mode=args.test)
        except Exception as e:
            print_error(f"Failed to process {p}: {e}")

    return 0


if __name__ == "__main__":
    main()
