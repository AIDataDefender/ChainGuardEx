import datetime
import logging
import os
import shutil
import traceback
from pathlib import Path
from typing import Optional, Union

from the_utils.logger import setup_logger

RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.environ.get("LOG_DIR")
if not LOG_DIR:
    LOG_DIR = f"Logs/t{RUN_TIMESTAMP}"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.environ["LOG_DIR"] = LOG_DIR

LOG_FILENAME = Path(LOG_DIR) / "file_ops.log"

logger = setup_logger(str(LOG_FILENAME), log_level=logging.INFO)


def safe_move(src: Union[Path, str], dst: Union[Path, str]) -> None:
    """
    Safely move a file from src to dst.

    Args:
        src: Source path.
        dst: Destination path.
    """
    try:
        dst_path = Path(dst)
        if dst_path.exists():
            dst_path.unlink()
        shutil.move(str(src), str(dst))
    except (shutil.Error, OSError) as e:
        logger.error(f"Failed to move {src} to {dst}: {e}")
        traceback.print_exc()


def move_dot_files(
    source_dir: Union[Path, str],
    target_dir: Union[Path, str],
    file_type: Optional[str] = None,
) -> bool:
    """
    Move .dot files from source_dir to target_dir.

    Args:
        source_dir: Source directory.
        target_dir: Target directory.
        file_type: Type of dot files to move (call_graph, cfg, inheritance).

    Returns:
        True if files were moved, False otherwise.
    """
    try:
        dot_files = []
        source_path = Path(source_dir)
        target_path = Path(target_dir)

        if file_type == "call_graph":
            dot_files = list(source_path.glob("*call-graph*.dot"))
        elif file_type == "cfg":
            dot_files = list(source_path.glob("*.dot"))
        elif file_type == "inheritance":
            dot_files = list(source_path.glob("*inheritance*.dot"))
        else:
            dot_files = list(source_path.glob("*.dot"))

        if dot_files:
            logger.info(
                f"Found {len(dot_files)} {file_type if file_type else ''} .dot files to move"
            )
            for f in dot_files:
                logger.info(f" - {f}")
            for dot_file in dot_files:
                target_file_path = target_path / dot_file.name
                safe_move(dot_file, target_file_path)
            logger.info(f"[GREEN]Successfully moved .dot files to {target_dir}")
            return True
        else:
            logger.info(
                f"No {file_type if file_type else ''} .dot files found to move at {source_dir}"
            )
            return False
    except Exception as e:
        error_msg = f"Error moving .dot files: {e}"
        logger.error(error_msg)
        traceback.print_exc()
        return False


def cleanup_project(project_dir_top: Union[Path, str]) -> None:
    """
    Cleanup project directory by removing node_modules and package files.

    Args:
        project_dir_top: Top-level project directory.
    """
    p = Path(project_dir_top)
    if not p.exists() or not p.is_dir():
        logger.error(f"Cleanup: project directory does not exist: {project_dir_top}")
        return

    for root, dirs, files in os.walk(p, topdown=True):
        for d in list(dirs):
            if d == "node_modules":
                nm_path = Path(root) / d
                logger.info(f"Removing {nm_path} ...")
                try:
                    shutil.rmtree(nm_path)
                    logger.info(f"[GREEN]Removed directory: {nm_path}")
                except Exception as e:
                    logger.error(f"Failed to remove {nm_path}: {e}")
                try:
                    dirs.remove(d)
                except ValueError:
                    pass

        if "package.json" in files:
            pkg_path = Path(root) / "package.json"
            if "node_modules" in pkg_path.parts:
                continue
            logger.info(f"Removing {pkg_path} ...")
            try:
                pkg_path.unlink()
                logger.info(f"[GREEN]Removed file: {pkg_path}")
            except Exception as e:
                logger.error(f"Failed to remove {pkg_path}: {e}")
        if "package-lock.json" in files:
            pkg_path = Path(root) / "package-lock.json"
            logger.info(f"Removing {pkg_path} ...")
            try:
                pkg_path.unlink()
                logger.info(f"[GREEN]Removed file: {pkg_path}")
            except Exception as e:
                logger.error(f"Failed to remove {pkg_path}: {e}")
