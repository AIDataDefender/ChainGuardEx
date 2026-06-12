import json
import random
import os
from pathlib import Path
import subprocess
import shutil
import logging
import sys
import time
import re
import glob
import traceback
import stat
import datetime
import hashlib
from typing import List, Optional, Dict, Any, Union

from the_utils.logger import setup_logger

def get_md5(file_path: Path) -> str:
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

#########################################
# ██╗░░░░░██╗███╗░░██╗██╗░░░██╗██╗░░██╗░░░░██╗░██╗░░░░░░░██╗░██████╗██╗░░░░░
# ██╗░░░░░██║████╗░██║██╗░░░██║╚██╗██╔╝░░░██╔╝░██╗░░██╗░░██║██╔════╝██║░░░░░
# ██╗░░░░░██║██╔██╗██║██╗░░░██║░╚███╔╝░░░██╔╝░░╚██╗████╗██╔╝╚█████╗░██║░░░░░
# ██╗░░░░░██║██║╚████║██╗░░░██║░██╔██╗░░██╔╝░░░░████╔═████║░░╚═══██╗██║░░░░░
# ███████╗██║██║░╚███║╚██████╔╝██╔╝╚██╗██╔╝░░░░░╚██╔╝░╚██╔╝░██████╔╝███████╗
# ╚══════╝╚═╝╚═╝░░╚══╝░╚═════╝░╚═╝░░╚═╝╚═╝░░░░░░░╚═╝░░░╚═╝░░╚═════╝░╚══════╝
# region Linux/WSL
#########################################
# Create a unique timestamped log file for this run
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"Logs/t{RUN_TIMESTAMP}"
LOG_FILENAME = f"{LOG_DIR}/b1_GraphExtractor.log"
# Ensure Logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)
os.environ["LOG_DIR"] = LOG_DIR
# SET UP LOGGER FOR SAME DIRECTORY
from b1_modules.b_1checkpoint_manager import CheckpointManager
from b1_modules.b_3solc_and_npm import (
    build_remappings,
    ensure_node_project,
    install_dependencies,
    set_solc_version,
    ensure_solc_for_file,
    get_project_pragma_version,
)
from b1_modules.b_2file_ops import move_dot_files, safe_move, cleanup_project

CWD = os.path.dirname(os.path.abspath(__file__))
# Setup centralized logger
logger = setup_logger(LOG_FILENAME, log_level=logging.INFO)

# Log the start of a new run
logger.info(f"Starting new analysis run at {RUN_TIMESTAMP}")

CHECKPOINT_FILE = Path("./analysis_checkpoint.json")
ALL_POSSIBLE = ["cfg", "ir", "abi"]
CHECKPOINT_MANAGER = CheckpointManager(
    all_step=ALL_POSSIBLE, file_path=str(CHECKPOINT_FILE)
)


SOURCE_CODE_ROOT = [
    os.path.join(CWD, 'source_code', 'access_control', 'clean_57_buggy_curated_0'),
    os.path.join(CWD, 'source_code', 'arithmetic', 'clean_60_buggy_curated_0'),
    os.path.join(CWD, 'source_code', 'denial_of_service', 'clean_46_buggy_curated_0'),
    os.path.join(CWD, 'source_code', 'reentrancy', 'clean_71_buggy_curated_0'),
    os.path.join(CWD, 'source_code', 'unchecked_low_level_calls', 'clean_95_buggy_curated_0'),
]


def process_projects(
    source_code_roots: List[str],
    ignore_errors: bool = True,
    load_checkpoint: bool = True,
    retry_failed: bool = False,
    single_project: bool = False,
    setup_only: bool = False,
    do_steps: Optional[List[str]] = None,
):
    """
    Process a complete Solidity project with checkpoint support.

    Args:
        source_code_roots: List of directories containing source code.
        ignore_errors: Whether to ignore errors.
        load_checkpoint: Whether to load checkpoint.
        retry_failed: Whether to retry failed projects.
        single_project: Whether to process only one project.
        setup_only: Whether to only setup the project.
        do_steps: List of steps to perform.
    """
    global CHECKPOINT_MANAGER

    if load_checkpoint:
        CHECKPOINT_MANAGER.load_checkpoint(single_project=single_project)
        # CHECKPOINT_MANAGER.print_checkpoint_summary()
        logger.info("Loading checkpoint - will only process pending steps...")
    else:
        logger.info("Starting fresh analysis - ignoring any existing checkpoint...")
    logger.info(f"Analysis log will be saved to: {LOG_FILENAME}")

    if do_steps:
        logger.info(f"--do filter specified; only running steps: {', '.join(do_steps)}")

    # Find all .sol files from the source roots and create isolated projects
    project_to_process = {}
    temp_projects_dir = Path("./temp_projects")
    temp_projects_dir.mkdir(parents=True, exist_ok=True)
    for root in source_code_roots:
        src_path = Path(root)
        if src_path.exists() and src_path.is_dir():
            sol_files = list(src_path.rglob("*.sol"))
            for sol_file in sol_files:
                md5_hash = get_md5(sol_file)
                project_name = f"{sol_file.stem}_{md5_hash}"
                # Create isolated temp directory for this project
                temp_dir = temp_projects_dir / project_name
                temp_dir.mkdir(parents=True, exist_ok=True)
                # Copy the sol file to temp dir
                copied_file = temp_dir / sol_file.name
                shutil.copy2(sol_file, copied_file)
                proj_item = {
                    "project_name": project_name,
                    "project_path": temp_dir,
                    "original_path": sol_file.parent,
                    "sol_file": copied_file,
                }
                project_to_process[project_name] = proj_item
        else:
            logger.error(f"Source directory does not exist: {root}")
            if not ignore_errors:
                sys.exit(1)

    total_projects = len(project_to_process)
    logger.info(f"Starting analysis of {total_projects} projects...")

    for i, (project_name, project_data) in enumerate(
        project_to_process.items(), start=1
    ):
        # Use the parent directory name from our mapping, or fallback to directory name if not found
        # project_dir_top should be the top-level directory (where package.json / node_modules live)
        project_dir_top = project_data.get("project_path")
        project_name = project_data.get("project_name", project_dir_top.name)
        logger.info(
            f"[BLUE]Process {project_name} with {len(project_data.get('sol_dir_to_process',[]))} solidity directories to analyze."
        )
        # Check if we should process this project based on checkpoint
        if load_checkpoint:
            pending_steps = CHECKPOINT_MANAGER.get_pending_steps(
                project_name, retry_failed
            )
            # If user specified --do, filter pending steps to only those requested
            # If --do is provided, bypass the checkpoint (re-run the steps requested)
            if do_steps:
                # Ensure we only run known steps
                allowed = [s for s in do_steps if s in ALL_POSSIBLE]
                pending_steps = allowed
                logger.info(
                    f"--do supplied: bypassing checkpoint and running: {', '.join(pending_steps)}"
                )
            logger.info(f"Pending steps for {project_name}: {', '.join(pending_steps)}")
            if not pending_steps:
                logger.info(
                    "[GREEN]"
                    + f"[{i}/{total_projects}] Project {project_name} already completed - skipping setup and analysis..."
                )
                continue
            if retry_failed:
                logger.info(
                    f"Pending/Failed steps for {project_name}: {', '.join(pending_steps)}"
                )
            else:
                logger.info(
                    f"Pending steps for {project_name}: {', '.join(pending_steps)}"
                )

        try:
            logger.info(
                f"[BLUE][{i}/{total_projects}] Processing project: {project_name}..."
            )
            ###########################################################
            ###########################################################

            try:
                logger.info(
                    "[YELLOW]"
                    + f"Configuring project environment for {project_name}..."
                )
                ensure_node_project(project_dir_top)

                solc_ver = set_solc_version(project_dir_top)
                install_dependencies(project_dir_top, solc_ver)
                project_data["solc_version"] = solc_ver
                logger.info("[GREEN]" + f"Project setup completed for {project_name}")
                if setup_only:
                    logger.info(
                        "[GREEN]"
                        + f"--setup-only specified: Completed setup for {project_name}, skipping further analysis."
                    )
                    continue
            except Exception as e:
                error_msg = f"Project setup failed for {project_name}: {e}"
                logger.error(f"⚠ {error_msg}")
                traceback.print_exc()

                if not ignore_errors:
                    sys.exit(1)
                else:
                    logger.info("Continuing with analysis despite setup failure...")

            # Run Slither analysis on the single file
            sol_file = project_data.get("sol_file")
            if sol_file:
                logger.info(f"[BLUE]Processing file: {sol_file}")
                run_slither_analysis(
                    project_data,
                    sol_file,
                    ignore_errors,
                    load_checkpoint,
                    retry_failed,
                    do_steps,
                )

            time.sleep(1)  # Short pause after analysis
            logger.info(
                "[GREEN]" + f"[{i}/{total_projects}] Processed project: {project_name}"
            )
        except Exception as e:
            error_msg = f"Project process failed for {project_name}: {e}"
            logger.error(f"⚠ {error_msg}")
            traceback.print_exc()

            if not ignore_errors:
                logger.error(
                    "Stopping analysis due to error. Use ignore_errors=True to continue."
                )
                break
            else:
                logger.info("Continuing with next project...")
    logger.info("[GREEN]" + "Analysis completed for all projects.")


def run_slither_command(
    command: List[str],
    project_name: str,
    analysis_type: str,
    cwd: Union[Path, str] = CWD,
) -> Optional[str]:
    """
    Run a Slither command and log its output to a file.

    Args:
        command: The command to run.
        project_name: The project name.
        analysis_type: The type of analysis.
        cwd: Current working directory.

    Returns:
        None on success, error string on failure.
    """
    logger.info(f"[BLUE]Running {analysis_type} analysis on {project_name}...")
    try:
        res = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
        if res.returncode == 0:
            return None
        return res.stderr
    except Exception as e:
        logger.error(
            f"❌ Slither {analysis_type} analysis on {project_name} failed: {e}"
        )
        traceback.print_exc()
        return str(e)


def run_slither_analysis(
    project_data: Dict[str, Any],
    sol_file: Path,
    ignore_errors: bool = True,
    load_checkpoint: bool = True,
    retry_failed: bool = False,
    do_steps: Optional[List[str]] = None,
):
    """
    Run Slither to generate Call Graph, CFG, Inheritance Graph, Data Dependency, and project-level ABI.

    Args:
        project_data: Project data dictionary.
        sol_dir: Directory containing Solidity files.
        ignore_errors: If True, continue analysis even when individual steps fail.
        load_checkpoint: If True, check checkpoint before running each step.
        retry_failed: Whether to retry failed steps.
        do_steps: List of steps to perform.

    This function performs multiple analyses:
    1. Call Graph generation - visual representation of function calls
    2. CFG (Control Flow Graph) - control flow within functions
    5. Project-level ABI Generation - consolidated contract interfaces in JSON format for the entire project
    """
    global CHECKPOINT_MANAGER
    # Create absolute paths for output files
    project_name = project_data.get("project_name")
    project_dir = project_data.get("project_path")

    if not project_name or not project_dir:
        logger.error(f"Missing project name or path in data: {project_data}")
        return

    base_extracted_dir = Path("./Extracted_Graphs") / project_name
    base_extracted_dir.mkdir(parents=True, exist_ok=True)

    # Create specific folders for each graph type
    cg_dir = base_extracted_dir / "CG"
    cfg_dir = base_extracted_dir / "CFG"
    # inheritance_dir = base_extracted_dir / "DependencyGraph"
    # data_dep_dir = base_extracted_dir / "DataDependency"
    abi_dir = base_extracted_dir / "ABI"
    # flat_dir = base_extracted_dir / "Flattened"
    # summary_dir = base_extracted_dir / "FunctionSummaries"
    ir_dir = base_extracted_dir / "IR"

    # Create all directories
    for dir_path in [cg_dir, cfg_dir, abi_dir, ir_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    proj_dir = Path(project_data.get("original_path")).resolve().as_posix()
    logger.info(f"Project directory: {proj_dir}\n\n")
    node_modules_dir = (Path(project_data.get("project_path")) / "node_modules").resolve().as_posix()

    logger.info(f"Expected node_modules directory: {node_modules_dir}\n\n")
    # print all folder in node_modules_dir - each on new line
    logger.info(f"Contents of node_modules directory:")
    if os.path.exists(node_modules_dir):
        logger.info(f"\n { "\n-".join(os.listdir(node_modules_dir))}")
    else:
        logger.info(f"Directory does not exist")
    print("\n")

    logger.info(f"Solidity file to process: {sol_file}\n\n")
    sol_files = [sol_file]

    # Remapping for standard and custom import names
    remappings = [
        f".={proj_dir}/",
        f"./={proj_dir}/",
        f"../={proj_dir}/",
        f"../../={proj_dir}/",
        f"./interfaces/={proj_dir}/interfaces/",
        f"interface/={proj_dir}/interface/",
        f"./libraries/={proj_dir}/libraries/",
    ]
    remappings.extend(
        build_remappings(node_modules_dir, str(project_data.get("solc_version")))
    )

    # move to CWD
    os.chdir(CWD)
    logger.info(f"Current working directory: {os.getcwd()}")

    slither_failed = False  # if failed - means there an dependency error - skip all related to slither
    # region BEGIN ANALYZE
    ####################
    # region  ABI
    # If do_steps is provided, skip this step if 'abi' not requested
    forced_abi = do_steps and ("abi" in do_steps)
    if do_steps and "abi" not in do_steps:
        logger.info("Skipping ABI generation due to --do filter")
    else:
        try:
            logger.info(f"Running ABI generation...")
            abi_success_count = 0
            for sol_file in sol_files:
                if (
                    load_checkpoint
                    and not forced_abi
                    and CHECKPOINT_MANAGER.should_skip_file_step(
                        project_name, str(sol_file), "abi", retry_failed
                    )
                ):
                    logger.info(f"ABI already completed for {sol_file} - skipping...")
                    abi_success_count += 1
                    continue
                logger.info(f"Generating ABI file for {sol_file}")
                # Ensure solc matches the file's pragma before attempting compilation
                try:
                    ok = ensure_solc_for_file(sol_file)
                    if not ok:
                        logger.error(
                            f"Could not set solc to match pragma for {sol_file}; continuing with current solc."
                        )
                except Exception as e:
                    logger.error(
                        f"Error trying to match solc to pragma for {sol_file}: {e}"
                    )
                    traceback.print_exc()
                abi_command = [
                    "solc",
                    *remappings,  # each mapping is its own argument
                    str(sol_file),
                    "--combined-json",
                    "abi,bin,bin-runtime,srcmap,srcmap-runtime",
                    "--allow-paths",
                    f".,{proj_dir},{abi_dir},{node_modules_dir}",
                ]
                ast_command = [
                    "solc",
                    *remappings,  # each mapping is its own argument
                    str(sol_file),
                    "--ast-compact-json",
                    "--allow-paths",
                    f".,{proj_dir},{abi_dir},{node_modules_dir}",
                ]

                try:
                    # # logger.info("[YELLOW]"+f"Running command: {' '.join(abi_command)}")
                    # output_file_abi = abi_dir / f"{sol_file.stem}_abi.json"
                    # with open(output_file_abi, "w") as f:
                    #     subprocess.run(abi_command, check=True, stdout=f)

                    output_file_ast = abi_dir / f"{sol_file.stem}_ast.json"

                    # 1. Run command
                    result = subprocess.run(
                        ast_command, check=True, capture_output=True, text=True
                    )
                    raw_stdout = result.stdout

                    # 2. Parse the Output
                    # The output looks like:
                    # { main_ast }
                    # ======= path/to/dep.sol =======
                    # { dep_ast }

                    # This regex splits by the separator and captures the filename
                    parts = re.split(r"=======\s*(.*?)\s*=======", raw_stdout)

                    collected_asts = {}

                    # PART A: The Main AST (Index 0)
                    # This is the text *before* the first separator.
                    main_content = parts[0].strip()
                    if main_content:
                        try:
                            # We use the current sol_file path as the key for the main AST
                            collected_asts[str(sol_file)] = json.loads(main_content)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Could not parse main AST for CONTENT: {main_content[:50]}"
                            )

                    # PART B: The Dependencies (Indices 1, 2, 3...)
                    # re.split returns [text, separator_capture, text, separator_capture, text...]
                    # So we iterate starting from index 1, in steps of 2
                    for i in range(1, len(parts) - 1, 2):
                        source_path = parts[i].strip()  # The captured filename
                        json_content = parts[
                            i + 1
                        ].strip()  # The JSON body following it

                        if json_content:
                            try:
                                collected_asts[source_path] = json.loads(json_content)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Could not parse AST for dependency: {source_path}"
                                )

                    # 3. Save Combined JSON
                    if collected_asts:
                        with open(output_file_ast, "w") as f:
                            json.dump(collected_asts, f, indent=2)

                    logger.info(
                        "[GREEN]"
                        + f"AST and ABI file generated: {output_file_ast}"
                    )
                    CHECKPOINT_MANAGER.set_file_status(
                        project_name, str(sol_file), "abi", 1
                    )
                    abi_success_count += 1
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error generating ABI for {sol_file}: {e.stderr}")
                    CHECKPOINT_MANAGER.set_file_status(
                        project_name,
                        str(sol_file),
                        "abi",
                        -1,
                        str(e.stderr),
                    )
                    if not ignore_errors:
                        sys.exit(1)
                    else:
                        continue

            if abi_success_count > 0:
                logger.info(
                    "[GREEN]"
                    + f"ABI generation completed for {abi_success_count} files"
                )
            else:
                logger.error(f"❌ ABI generation failed for all files")
        except Exception as e:
            error_msg = f"ABI generation failed"
            logger.error(f"❌ {error_msg}")
            traceback.print_exc()
            if not ignore_errors:
                sys.exit(1)
            else:
                logger.info(f"Continuing despite ABI generation failure...")

    # ####################
    # # region  CallGraph
    # # If do_steps is provided, skip this step if 'cg' not requested
    # forced_cg = do_steps and ("cg" in do_steps)
    # if do_steps and "cg" not in do_steps:
    #     logger.info("Skipping Call Graph (cg) due to --do filter")
    # else:
    #     ####################
    #     # CG is per directory, so always run (no checkpoint skip)
    #     try:
    #         logger.info(f"Running Call Graph analysis in {str(sol_dir)}...")
    #         call_graph_command = [
    #             "slither",
    #             str(sol_dir),
    #             "--print",
    #             "call-graph",
    #             "--solc-disable-warnings",
    #             "--solc-remaps",
    #             " ".join(remappings),
    #         ]
    #         logger.info("[YELLOW]" + f"Running command: {' '.join(call_graph_command)}")
    #         res = run_slither_command(call_graph_command, project_name, "call-graph")
    #         move_result = move_dot_files(sol_dir, cg_dir, "call_graph")

    #         if not move_result:
    #             CHECKPOINT_MANAGER.set_file_status(
    #                 project_name,
    #                 str(sol_dir),
    #                 "cg",
    #                 -1,
    #                 str(res),
    #             )
    #             logger.error(f"❌ Call Graph analysis failed: {res} {move_result}")
    #             slither_failed = True
    #         else:
    #             CHECKPOINT_MANAGER.set_file_status(project_name, str(sol_dir), "cg", 1)
    #             logger.info("[GREEN]" + f"Call Graph analysis completed successfully")
    #     except Exception as e:
    #         error_msg = f"Call Graph analysis failed"
    #         CHECKPOINT_MANAGER.set_file_status(
    #             project_name,
    #             str(sol_dir),
    #             "cg",
    #             -1,
    #             traceback.format_exc(),
    #         )
    #         logger.error(f"❌ {error_msg}")
    #         traceback.print_exc()
    #         slither_failed = True
    #         if not ignore_errors:
    #             sys.exit(1)
    #         else:
    #             logger.info(f"Continuing despite Call Graph failure...")

    # ####################
    # region  CFG
    # If do_steps is provided, skip this step if 'cfg' not requested
    forced_cfg = do_steps and ("cfg" in do_steps)
    if do_steps and "cfg" not in do_steps:
        logger.info("Skipping CFG due to --do filter")
    else:
        ####################
        # CFG is per directory, so always run (no checkpoint skip)
        try:
            logger.info(f"Running CFG analysis in {str(sol_file)}...")
            cfg_command = [
                "slither",
                str(sol_file),
                "--print",
                "cfg",
                "--solc-disable-warnings",
                "--solc-remaps",
                " ".join(remappings),
            ]
            logger.info("[YELLOW]" + f"Running command: {' '.join(cfg_command)}")
            res = run_slither_command(cfg_command, project_name, "cfg")
            move_result = move_dot_files(sol_file.parent, cfg_dir, "cfg")
            if not move_result:
                CHECKPOINT_MANAGER.set_file_status(
                    project_name,
                    str(sol_file),
                    "cfg",
                    -1,
                    str(res),
                )
                logger.error(f"❌ CFG analysis failed")
            else:
                CHECKPOINT_MANAGER.set_file_status(project_name, str(sol_file), "cfg", 1)
                logger.info("[GREEN]" + f"CFG analysis completed successfully")
        except Exception as e:
            error_msg = f"CFG analysis failed"
            CHECKPOINT_MANAGER.set_file_status(
                project_name,
                str(sol_file),
                "cfg",
                -1,
                traceback.format_exc(),
            )
            logger.error(f"❌ {error_msg}")
            traceback.print_exc()
            if not ignore_errors:
                sys.exit(1)
            else:
                logger.info(f"Continuing despite CFG failure...")

    # region  IR
    # If do_steps is provided, skip this step if 'ir' not requested
    forced_ir = do_steps and ("ir" in do_steps)
    if do_steps and "ir" not in do_steps:
        logger.info("Skipping IR due to --do filter")
    else:
        try:
            logger.info(f"Running IR analysis...")
            ir_success_count = 0
            ress = []
            for sol_file in sol_files:
                if (
                    load_checkpoint
                    and not forced_ir
                    and CHECKPOINT_MANAGER.should_skip_file_step(
                        project_name, str(sol_file), "ir", retry_failed
                    )
                ):
                    logger.info(f"IR already completed for {sol_file} - skipping...")
                    ir_success_count += 1
                    continue
                logger.info(f"Generating IR file for {sol_file}")
                unique_name = str(sol_file.name)
                ir_file_path = ir_dir / f"ir_{unique_name}.json"
                if ir_file_path.exists():
                    ir_file_path.unlink()
                # Ensure the solc in use matches the pragma for this file.
                # If the pragma in the file differs from the currently selected solc, try to set the solc to the file's pragma.
                try:
                    ok = ensure_solc_for_file(sol_file)
                    if not ok:
                        logger.error(
                            f"Failed to match solc to pragma of {sol_file}. Continuing with current solc."
                        )
                except Exception as e:
                    logger.error(
                        f"Error while ensuring solc version for {sol_file}: {e}"
                    )
                    traceback.print_exc()

                res = run_slither_command(
                    [
                        "slither",
                        str(sol_file),
                        "--print",
                        "slithir-ssa",
                        "--json",
                        str(ir_file_path),
                        "--solc-disable-warnings",
                        "--solc-remaps",
                        " ".join(remappings),
                    ],
                    project_name,
                    "ir",
                )
                if res:
                    ress.append(res)
                    CHECKPOINT_MANAGER.set_file_status(
                        project_name,
                        str(sol_file),
                        "ir",
                        -1,
                        res,
                    )
                else:
                    CHECKPOINT_MANAGER.set_file_status(
                        project_name, str(sol_file), "ir", 1
                    )
                    ir_success_count += 1

            rex = "\n\n".join(ress)
            if ir_success_count > 0:
                logger.info(
                    "[GREEN]" + f"IR analysis completed for {ir_success_count} files"
                )
            else:
                logger.error(f"❌ IR analysis failed for all files")
        except Exception as e:
            error_msg = f"IR analysis failed:"
            logger.error(f"❌ {error_msg}")
            traceback.print_exc()
            if not ignore_errors:
                sys.exit(1)
            else:
                logger.info(f"Continuing despite IR failure...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract graphs and ABI from Solidity projects with checkpoint support"
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        default=True,
        help="Continue analysis even when individual steps fail (default: True)",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        default=False,
        help="Make a new analysis, ignore all the previous/saved ones(default: False)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        default=False,
        help="Retry previously failed steps (status -1)",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="analysis_checkpoint.json",
        help="Path to checkpoint file (default: analysis_checkpoint.json)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Path to a single project directory to process (default: None)",
    )
    parser.add_argument(
        "--multi-projects",
        type=str,
        default=None,
        help="Comma-separated list of project directories to process as individual projects (default: None)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help="Remove node_modules and package.json from the target project(s) before processing",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        default=False,
        help="Only setup the project(s) (install deps & solc) and exit (no analysis)",
    )
    parser.add_argument(
        "--do",
        type=str,
        default=None,
        help="Comma-separated list of analysis steps to run. Allowed values: cg,cfg,abi,ir",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Run a quick test on one random top-level project from the source directory (default: False)",
    )

    args = parser.parse_args()

    is_load_checkpoint = not args.fresh_start

    if not is_load_checkpoint and os.path.exists(CHECKPOINT_MANAGER.checkpoint_file):
        logger.info(
            f"--fresh-start specified: backup and removing existing checkpoint file {CHECKPOINT_MANAGER.checkpoint_file}"
        )
        try:
            backup_file = CHECKPOINT_MANAGER.checkpoint_file + ".bak"
            shutil.copy2(CHECKPOINT_MANAGER.checkpoint_file, backup_file)
            logger.info(f"Backup created: {backup_file}")
            time.sleep(1)  # Ensure backup timestamp differs
            os.remove(CHECKPOINT_MANAGER.checkpoint_file)
        except Exception as e:
            logger.error(f"Failed to remove existing checkpoint file: {e}")
            traceback.print_exc()

    logger.info("=" * 40)
    logger.info("GRAPH EXTRACTOR WITH CHECKPOINT SYSTEM")
    logger.info("=" * 40)
    logger.info(f"Ignore errors: {args.ignore_errors}")
    logger.info(f"Load checkpoint: {is_load_checkpoint}")
    logger.info(f"Retry failed: {args.retry_failed}")
    logger.info(f"Checkpoint file: {CHECKPOINT_MANAGER.checkpoint_file}")
    logger.info("=" * 40)

    # Parse --do list of steps
    do_steps = None
    if args.do:
        do_steps = [s.strip().lower() for s in args.do.split(",") if s.strip()]
        invalid = [s for s in do_steps if s not in ALL_POSSIBLE]
        if invalid:
            logger.error(
                f"Invalid --do steps: {invalid}. Allowed: {', '.join(sorted(ALL_POSSIBLE))}"
            )
            sys.exit(1)

    # If --test is specified, select a random project and test only it
    if args.test:
        # --test is mutually exclusive with --project / --multi-projects
        if args.project or args.multi_projects:
            logger.error("--test cannot be used with --project or --multi-projects")
            sys.exit(1)
        logger.info(f"SOURCE_CODE_ROOT: {SOURCE_CODE_ROOT}")
        all_sol_files = []
        for root in SOURCE_CODE_ROOT:
            src_path = Path(root)
            logger.info(f"Checking root: {root}, exists: {src_path.exists()}, is_dir: {src_path.is_dir()}")
            if src_path.exists() and src_path.is_dir():
                sol_files = list(src_path.rglob("*.sol"))
                logger.info(f"Found {len(sol_files)} .sol files in {root}")
                all_sol_files.extend(sol_files)
        logger.info(f"Total .sol files found: {len(all_sol_files)}")
        if not all_sol_files:
            logger.error(f"No .sol files found in SOURCE_CODE_ROOT to run --test on.")
            sys.exit(1)
        chosen_file = random.choice(all_sol_files)
        md5_hash = get_md5(chosen_file)
        project_name = f"{chosen_file.stem}_{md5_hash}"
        logger.info(f"Running --test on random file: {chosen_file} as project {project_name}")
        # Create isolated temp directory
        temp_projects_dir = Path("./temp_projects")
        temp_projects_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = temp_projects_dir / project_name
        temp_dir.mkdir(parents=True, exist_ok=True)
        copied_file = temp_dir / chosen_file.name
        shutil.copy2(chosen_file, copied_file)
        # Create project_data
        project_data = {
            "project_name": project_name,
            "project_path": temp_dir,
            "original_path": chosen_file.parent,
            "sol_file": copied_file,
        }
        # Simulate the processing
        # But to reuse, perhaps call process_projects with a list containing one, but since single_project, but process_projects expects list of roots.
        # Easier to duplicate the logic here.
        logger.info(f"Processing test project: {project_name}...")
        try:
            logger.info(
                "[YELLOW]"
                + f"Configuring project environment for {project_name}..."
            )
            ensure_node_project(temp_dir)
            solc_ver = set_solc_version(temp_dir)
            install_dependencies(temp_dir, solc_ver)
            project_data["solc_version"] = solc_ver
            logger.info("[GREEN]" + f"Project setup completed for {project_name}")
            if not args.setup_only:
                run_slither_analysis(
                    project_data,
                    chosen_file,
                    args.ignore_errors,
                    is_load_checkpoint,
                    args.retry_failed,
                    do_steps,
                )
            logger.info("[GREEN]" + f"Test project processed: {project_name}")
        except Exception as e:
            logger.error(f"Test project failed: {e}")
            traceback.print_exc()
        sys.exit(0)

    if args.cleanup:
        logger.info(
            "--cleanup specified: will remove node_modules and package.json from target project(s)"
        )
        # If single project, validate and cleanup that project only
        if args.project:
            project_path = Path(args.project)
            if not project_path.exists() or not project_path.is_dir():
                logger.error(
                    f"Specified project directory does not exist: {args.project}"
                )
                sys.exit(1)
            cleanup_project(project_path)
        else:
            # Cleanup for all source roots
            for root in SOURCE_CODE_ROOT:
                root_path = Path(root)
                if root_path.exists() and root_path.is_dir():
                    logger.info(f"Cleaning up root: {root}")
                    cleanup_project(root_path)
                else:
                    logger.warning(f"Root does not exist: {root}")

        logger.info("Cleanup step completed.")
        sys.exit(0)

    # If --multi-projects is specified, process each project in the list
    if args.multi_projects:
        project_paths = [p.strip() for p in args.multi_projects.split(",") if p.strip()]
        if not project_paths:
            logger.error(
                "No valid project paths provided in --multi-projects argument."
            )
            sys.exit(1)
        for idx, proj in enumerate(project_paths, start=1):
            project_path = Path(proj)
            if not project_path.exists() or not project_path.is_dir():
                logger.error(
                    f"[{idx}/{len(project_paths)}] Specified project directory does not exist: {proj}"
                )
                continue
            logger.info(f"[{idx}/{len(project_paths)}] Processing project: {proj}")
            logger.info(
                "--multi-projects specified: ignoring checkpoint-based skipping for this run."
            )
            process_projects(
                [str(project_path)],
                ignore_errors=args.ignore_errors,
                load_checkpoint=is_load_checkpoint,
                retry_failed=args.retry_failed,
                single_project=True,
                setup_only=args.setup_only,
                do_steps=do_steps,
            )
    # If --project is specified, process only that project
    elif args.project:
        project_path = Path(args.project)
        if not project_path.exists() or not project_path.is_dir():
            logger.error(f"Specified project directory does not exist: {args.project}")
            sys.exit(1)
        logger.info(f"Processing only specified project: {args.project}")
        logger.info(
            "--project specified: ignoring checkpoint-based skipping for this run."
        )
        process_projects(
            [str(project_path)],
            ignore_errors=args.ignore_errors,
            load_checkpoint=is_load_checkpoint,
            retry_failed=args.retry_failed,
            single_project=True,
            setup_only=args.setup_only,
            do_steps=do_steps,
        )
    else:
        # Default behavior: extract graphs and ABI for all projects
        process_projects(
            SOURCE_CODE_ROOT,
            ignore_errors=args.ignore_errors,
            load_checkpoint=is_load_checkpoint,
            retry_failed=args.retry_failed,
            setup_only=args.setup_only,
            do_steps=do_steps,
        )
