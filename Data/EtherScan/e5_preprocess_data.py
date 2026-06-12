import argparse
import hashlib
import json
import logging
import pickle
import time
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import traceback
from typing import Any, Dict, List, Tuple, Optional, Set, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
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

random.seed(42) # seed for holdout selection

# from e_7extract_code import extract_code_with_vulnerabilities
from the_utils.logger import setup_logger

from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILENAME = f"Logs/DataProc_Phase2_run_{RUN_TIMESTAMP}.log"

CURRENT_DIR = Path(__file__).parent.resolve()
EXTRACTED_GRAPHS_DIR = CURRENT_DIR / "Extracted_Graphs"
OUTPUT_DIR = CURRENT_DIR / "ProcessedData_test"
SOURCE_CODE_DIR = CURRENT_DIR / "parsed_filtered_contracts" / "raw_code"

OUTPUT_DIR.mkdir(exist_ok=True)
SUCCESS_OUTPUT_DIR = OUTPUT_DIR / "success"
FAILED_OUTPUT_DIR = OUTPUT_DIR / "failed"
BENIGN_OUTPUT_DIR = OUTPUT_DIR / "benign"
STAGING_OUTPUT_DIR = OUTPUT_DIR / "_staging"
HOLDOUT_OUTPUT_DIR = OUTPUT_DIR / "holdout"
for folder in (SUCCESS_OUTPUT_DIR, FAILED_OUTPUT_DIR, BENIGN_OUTPUT_DIR, STAGING_OUTPUT_DIR, HOLDOUT_OUTPUT_DIR):
    folder.mkdir(exist_ok=True)

# Setup centralized logger
logger = setup_logger(LOG_FILENAME, log_level=logging.INFO)

# ----------------------
# Constants
# ----------------------
SOURCE_CODE_VULN_FOLDER = Path("compiled_findings.json")

def print_info(message):
    """Print information messages in white (default color)"""
    logger.info(message)

def print_error(message):
    """Print error messages in red"""
    logger.error(message)


def print_processing(message):
    """Print in-processing messages in yellow"""
    logger.info(f"[YELLOW]{message}")

def compact_code(code: str) -> str:
    # lines = code.split('\n')
    # stripped = [line.lstrip() for line in lines]
    # return '\n'.join(stripped).strip()
    return code

def _fetch_code_raw_from_vuln_entry(vuln_entry: Dict[str, Any]):
    file_path = vuln_entry["file"]
    sol_path = Path("parsed_filtered_contracts/raw_code") / os.path.basename(file_path)
    if not sol_path.exists():
        print_error(f"SOL file not found: {sol_path}")
        return
    with open(sol_path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.split('\n')

    line_from = vuln_entry["line_from"]
    line_to = vuln_entry["line_to"]
    file_path = vuln_entry["file"]

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
            vuln_entry["snippet"] = compact_code(code)
        except (ValueError, IndexError):
            print_error(f"Invalid line numbers: {line_from}-{line_to}")
    else:
        print_error(f"Invalid line numbers: {line_from}-{line_to}")


def _extract_function_name(file_path: str, line_number: int) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if line_number < 1 or line_number > len(lines):
            return ""
        # Start from the line (0-based), go backwards
        for i in range(line_number - 1, -1, -1):
            line = lines[i].strip()
            # Match function definition: function name(
            match = re.match(r'^\s*function\s+(\w+)\s*\(', line)
            if match:
                return match.group(1)
        return ""
    except Exception as e:
        print_error(f"Error extracting function name from {file_path}:{line_number}: {e}")
        return ""


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


def _load_slither_mapping(csv_path: Path) -> Dict[str, Dict[str, str]]:
    try:
        if not csv_path.exists():
            print_info(f"Slither mapping not found at {csv_path}")
            return {}
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines='skip')
        mapping = {}
        for _, row in df.iterrows():
            slither = str(row["slither-detect"]).strip()
            owasp = row["OWASP"]
            if pd.isna(owasp):
                owasp = "N/A"
            else:
                owasp = str(owasp).strip()
            swc = "N/A"
            mapping[slither] = {"OWASP": owasp, "SWC": swc}
        print_info(f"Loaded {len(mapping)} slither mappings from {csv_path}")
        return mapping
    except Exception as e:
        print_error(f"Failed to load slither mapping: {e}")
        return {}


def _get_func_line_vuln(
    project_name: str,
    vuln_json: Dict[str, Any],
) -> Dict[str, Any]:
    slither_mapping = _load_slither_mapping(Path("slither_mapping.csv"))
    oyente_mapping = {
        "Callstack Depth Attack Vulnerability": "SC06",
        "Integer Overflow": "SC08",
        "Integer Underflow": "SC08",
        "Transaction-Ordering Dependence":"N/A",
        "Timestamp Dependency":"N/A",
        "Re-Entrancy Vulnerability":"SC05",
        "Parity Multisig Bug 2":"SC03",
    }
    honeybadger_mapping = {
        "Money flow" :"SC03",
        "Balance disorder" :"SC03",
        "Hidden transfer":"N/A",
        "Inheritance disorder":"SC03",
        "Uninitialised struct":"SC03",
        "Type overflow":"SC08",
        "Skip empty string":"SC03",
        "Hidden state update":"N/A",
        "Straw man contract":"N/A",
    }

    def get_contract_bounds(file_path: str, contract_name: str) -> Tuple[int, int]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Find all contract positions
            contract_pattern = re.compile(r'\bcontract\s+(\w+).*?\{', re.MULTILINE | re.DOTALL)
            for match in contract_pattern.finditer(content):
                name = match.group(1)
                if name == contract_name:
                    start_pos = match.start()
                    start_line = content[:start_pos].count('\n') + 1
                    # Now, from start_pos, count braces
                    brace_count = 0
                    i = start_pos
                    while i < len(content):
                        if content[i] == '{':
                            brace_count += 1
                        elif content[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i
                                end_line = content[:end_pos].count('\n') + 1
                                return start_line, end_line
                        i += 1
            # If not found, return default
            return 1, 1
        except Exception as e:
            print_error(f"Error getting contract bounds for {file_path}: {e}")
            return 1, 1

    line_vuln = {}
    counter = 0
    for contract_file, tools in vuln_json.items():
        if not contract_file.startswith(project_name):
            continue
        for tool, findings in tools.items():
            if tool == "slither-0.11.3":
                for finding in findings:
                    if "line" not in finding:
                        continue
                    category = finding["name"]
                    mapping = slither_mapping.get(category, {})
                    swc_id = mapping.get("SWC")
                    owasp_id = mapping.get("OWASP", "N/A")
                    if owasp_id == "N/A":
                        print_error(f"[SLITHER] Got no OWASP mapping: {category}")
                        continue
                    line_from = str(finding["line"])
                    line_to = str(finding.get("line_end", finding["line"]))
                    function = finding.get("function", "N/A")
                    if function == "N/A":
                        function = _extract_function_name(str(Path(SOURCE_CODE_DIR) / os.path.basename(finding["filename"])), int(line_from))
                    filename = os.path.basename(finding["filename"])
                    _data = {
                        "project": project_name,
                        "file": filename,
                        "function": function,
                        "category": category,
                        "swc_id": swc_id,
                        "owasp_id": owasp_id,
                        "swc_name": category,
                        "line_from": line_from,
                        "line_to": line_to,
                        "detect_type": "source_code",
                        "original_filename": finding["filename"],
                    }
                    _fetch_code_raw_from_vuln_entry(_data)
                    key = f"{contract_file}${function}${swc_id}${owasp_id}${line_from}${line_to}${str(counter)}"
                    line_vuln[key] = _data
                    counter += 1
            elif tool == "oyente":
                for finding in findings:
                    if "line" not in finding:
                        continue
                    category = finding["name"]
                    owasp_id = oyente_mapping.get(category,"N/A")  # if not in mapping, use as is, but correct
                    if owasp_id == "N/A":
                        print_error(f"[OYENTE] Got no OWASP mapping: {category}")
                        continue
                    swc_id = None
                    line_from = str(finding["line"])
                    line_to = str(finding.get("line_end", finding["line"]))
                    filename = os.path.basename(finding["filename"])
                    function = _extract_function_name(str(Path(SOURCE_CODE_DIR) / filename), int(line_from))
                    _data = {
                        "project": project_name,
                        "file": filename,
                        "function": function,
                        "category": category,
                        "swc_id": swc_id,
                        "owasp_id": owasp_id,
                        "swc_name": category,
                        "line_from": line_from,
                        "line_to": line_to,
                        "detect_type": "source_code",
                        "original_filename": finding["filename"],
                    }
                    _fetch_code_raw_from_vuln_entry(_data)
                    key = f"{contract_file}${function}${swc_id}${owasp_id}${line_from}${line_to}${str(counter)}"
                    line_vuln[key] = _data
                    counter += 1
            elif tool == "honeybadger":
                for finding in findings:
                    category = finding["name"]
                    owasp_id = honeybadger_mapping.get(category, "N/A")  # if not in mapping, use as is
                    if owasp_id == "N/A":
                        print_error(f"[HB] Got no OWASP mapping: {category}")
                        continue
                    swc_id = None
                    sol_path = Path(SOURCE_CODE_DIR) / os.path.basename(finding["filename"])
                    contract_name = finding.get("contract", "")
                    line_from, line_to = get_contract_bounds(str(sol_path), contract_name)
                    line_from = str(line_from)
                    line_to = str(line_to)
                    print(f"Determined contract bounds for {finding['filename']}: {line_from}-{line_to}")
                    if not line_from:
                        print("[HB] Could not determine line numbers for contract:", finding["filename"])
                        continue
                    function = _extract_function_name(str(sol_path), int(line_from))
                    filename = os.path.basename(finding["filename"])
                    _data = {
                        "project": project_name,
                        "file": filename,
                        "function": function,
                        "category": category,
                        "swc_id": swc_id,
                        "owasp_id": owasp_id,
                        "swc_name": category,
                        "line_from": line_from,
                        "line_to": line_to,
                        "detect_type": "source_code",
                        "original_filename": finding["filename"],
                    }
                    _fetch_code_raw_from_vuln_entry(_data)
                    key = f"{contract_file}${function}${swc_id}${owasp_id}${line_from}${line_to}${str(counter)}"
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
        # 50:50 chance to success or holdout
        if random.choice([True, False]):
            target_root = SUCCESS_OUTPUT_DIR / project_name
        else:
            target_root = HOLDOUT_OUTPUT_DIR / project_name
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
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}Z",
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
    vuln_json: Optional[Dict[str, Any]] = None,
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

        if vuln_json is None:
            with open(sc_vuln_folder, "r", encoding="utf-8") as f:
                SC_vuln_json = json.load(f)
        else:
            SC_vuln_json = vuln_json
        #SC_vuln_json = json.load(open(sc_vuln_folder, "r", encoding="utf-8"))  # _load_vuln_json(sc_vuln_folder)
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
        "--rerun-missing-gpickle",
        action="store_true",
        help="Re-run processing for projects in success/holdout missing cpg_graph.gpickle",
    )
    args = parser.parse_args(argv)

    if args.rerun_missing_gpickle:
        # Find projects in success and holdout missing cpg_graph.gpickle
        missing_projects = set()
        for category in ['success', 'holdout']:
            cat_dir = OUTPUT_DIR / category
            if cat_dir.exists():
                for proj_dir in cat_dir.iterdir():
                    if proj_dir.is_dir():
                        gpickle_path = proj_dir / 'cpg_graph.gpickle'
                        if not gpickle_path.exists():
                            missing_projects.add(proj_dir.name)
        tt_projects = [p for p in Path(EXTRACTED_GRAPHS_DIR).iterdir() if p.is_dir() and p.name in missing_projects]
        projects = tt_projects  # Process all missing, even if already done
        print_info(f"Re-running for {len(projects)} projects missing cpg_graph.gpickle")
    else:
        tt_projects = [p for p in Path(EXTRACTED_GRAPHS_DIR).iterdir() if p.is_dir()]
        valid_sols = [s.stem for s in Path(SOURCE_CODE_DIR).rglob("*.sol")]
        done = [p.name for p in tt_projects if (SUCCESS_OUTPUT_DIR / p.name).exists() or (FAILED_OUTPUT_DIR / p.name).exists() or (BENIGN_OUTPUT_DIR / p.name).exists()]
        projects = [p for p in tt_projects if p.name in valid_sols and p.name not in done]

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

    # Load vulnerability data once
    with open("compiled_findings.json", "r", encoding="utf-8") as f:
        vuln_data = json.load(f)

    # Parallel processing of projects
    max_workers = min(len(projects), os.cpu_count() or 4)  # Limit to CPU count or 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for p in projects:
            proj_name = p.name
            proj_vuln = {k: v for k, v in vuln_data.items() if k.startswith(proj_name)}
            futures.append(executor.submit(process_project_folder, p, args.workers, do_steps, False, proj_vuln))
        for i, future in enumerate(as_completed(futures)):
            try:
                future.result()
                print_info(f"Completed project {i+1}/{len(projects)}")
            except Exception as e:
                print_error(f"Failed to process a project: {e}")

    return 0


if __name__ == "__main__":
    main()
