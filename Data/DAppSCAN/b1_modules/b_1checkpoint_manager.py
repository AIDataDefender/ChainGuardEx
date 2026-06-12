import datetime
import json
import os
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from the_utils.logger import setup_logger

RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.environ.get("LOG_DIR")
if not LOG_DIR:
    LOG_DIR = f"Logs/t{RUN_TIMESTAMP}"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.environ["LOG_DIR"] = LOG_DIR

LOG_FILENAME = Path(LOG_DIR) / "Checkpoint_manager.log"

logger = setup_logger(str(LOG_FILENAME))


class CheckpointManager:
    """Manages analysis checkpoints to enable resuming from failed states."""

    def __init__(
        self,
        all_step: List[str] = ["cg", "cfg", "ir", "abi"],
        file_path: str = "./analysis_checkpoint.json",
    ):
        """
        Initialize the CheckpointManager.

        Args:
            all_step: List of all analysis steps.
            file_path: Path to the checkpoint file.
        """
        self.checkpoint_file = file_path
        self.checkpoint_data: Dict[str, Any] = {
            "analysis_base_path": Path(file_path).resolve().parent.as_posix(),
            "projects": {},
        }
        self.is_single_project = False
        self.all_steps = all_step

    def load_checkpoint(self, single_project: bool = False) -> Dict[str, Any]:
        """
        Load the checkpoint from file.

        Args:
            single_project: Whether running in single project mode.

        Returns:
            The checkpoint data.
        """
        self.is_single_project = single_project
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)
                    logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
                    self.checkpoint_data = data
                    # Ensure "files" key exists in projects
                    for proj in self.checkpoint_data.get("projects", {}):
                        if "files" not in self.checkpoint_data["projects"][proj]:
                            self.checkpoint_data["projects"][proj]["files"] = {}
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load checkpoint: {e}")
                traceback.print_exc()
        return self.checkpoint_data

    def save_checkpoint(self) -> None:
        """Save the checkpoint to file."""
        try:
            existing_data = {}
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, "r") as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = {}
                        traceback.print_exc()

            merged_data = {**existing_data, **self.checkpoint_data}
            with open(self.checkpoint_file, "w") as f:
                json.dump(merged_data, f, indent=2)
            logger.info(f"Checkpoint saved to {self.checkpoint_file}")

        except IOError as e:
            logger.error(f"Failed to save checkpoint: {e}")
            traceback.print_exc()

    def get_project_status(self, project_name: str, step: str) -> int:
        """
        Get the status of a project step.

        Args:
            project_name: The project name.
            step: The step name.

        Returns:
            The status code (0: pending, 1: success, -1: failed).
        """
        if project_name not in self.checkpoint_data["projects"]:
            self.checkpoint_data["projects"][project_name] = {
                "path": "",
                "steps": {},
                "errors": {},
                "files": {},
            }
        return self.checkpoint_data["projects"][project_name]["steps"].get(step, 0)

    def set_project_status(
        self,
        project_name: str,
        step: str,
        status: int,
        error_msg: Optional[str] = None,
        project_path: Optional[str] = None,
    ) -> None:
        """
        Set the status of a project step.

        Args:
            project_name: The project name.
            step: The step name.
            status: The status code.
            error_msg: Optional error message.
            project_path: Optional project path.
        """
        if project_name not in self.checkpoint_data["projects"]:
            self.checkpoint_data["projects"][project_name] = {
                "path": "",
                "steps": {},
                "errors": {},
                "files": {},
            }
        if project_path:
            self.checkpoint_data["projects"][project_name]["path"] = Path(
                project_path
            ).as_posix()
        self.checkpoint_data["projects"][project_name]["steps"][step] = status

        if error_msg and status == -1:
            self.checkpoint_data["projects"][project_name]["errors"][step] = error_msg
        elif (
            status == 1
            and step in self.checkpoint_data["projects"][project_name]["errors"]
        ):
            del self.checkpoint_data["projects"][project_name]["errors"][step]

        self.save_checkpoint()

    def should_skip_step(
        self, project_name: str, step: str, retry_failed: bool = False
    ) -> bool:
        """
        Check if a step should be skipped.

        Args:
            project_name: The project name.
            step: The step name.
            retry_failed: Whether to retry failed steps.

        Returns:
            True if the step should be skipped, False otherwise.
        """
        status = self.get_project_status(project_name, step)
        if self.is_single_project:
            return False
        if retry_failed:
            return status == 1
        else:
            return status in (1, -1)

    def get_file_status(self, project_name: str, file_path: str, step: str) -> int:
        """
        Get the status of a file step.

        Args:
            project_name: The project name.
            file_path: The file path.
            step: The step name.

        Returns:
            The status code.
        """
        if project_name not in self.checkpoint_data["projects"]:
            self.checkpoint_data["projects"][project_name] = {
                "path": "",
                "files": {},
                "steps": {},
                "errors": {},
            }
        if "files" not in self.checkpoint_data["projects"][project_name]:
            self.checkpoint_data["projects"][project_name]["files"] = {}

        if file_path not in self.checkpoint_data["projects"][project_name]["files"]:
            self.checkpoint_data["projects"][project_name]["files"][file_path] = {
                "steps": {},
                "errors": {},
            }
        return self.checkpoint_data["projects"][project_name]["files"][file_path][
            "steps"
        ].get(step, 0)

    def set_file_status(
        self,
        project_name: str,
        file_path: str,
        step: str,
        status: int,
        error_msg: Optional[str] = None,
    ) -> None:
        """
        Set the status of a file step.

        Args:
            project_name: The project name.
            file_path: The file path.
            step: The step name.
            status: The status code.
            error_msg: Optional error message.
        """
        if project_name not in self.checkpoint_data["projects"]:
            self.checkpoint_data["projects"][project_name] = {
                "path": "",
                "files": {},
                "steps": {},
                "errors": {},
            }
        if "files" not in self.checkpoint_data["projects"][project_name]:
            self.checkpoint_data["projects"][project_name]["files"] = {}

        if file_path not in self.checkpoint_data["projects"][project_name]["files"]:
            self.checkpoint_data["projects"][project_name]["files"][file_path] = {
                "steps": {},
                "errors": {},
            }
        self.checkpoint_data["projects"][project_name]["files"][file_path]["steps"][
            step
        ] = status

        if error_msg and status == -1:
            self.checkpoint_data["projects"][project_name]["files"][file_path][
                "errors"
            ][step] = error_msg
        elif (
            status == 1
            and step
            in self.checkpoint_data["projects"][project_name]["files"][file_path][
                "errors"
            ]
        ):
            del self.checkpoint_data["projects"][project_name]["files"][file_path][
                "errors"
            ][step]

        self.save_checkpoint()

    def should_skip_file_step(
        self, project_name: str, file_path: str, step: str, retry_failed: bool = False
    ) -> bool:
        """
        Check if a file step should be skipped.

        Args:
            project_name: The project name.
            file_path: The file path.
            step: The step name.
            retry_failed: Whether to retry failed steps.

        Returns:
            True if the step should be skipped, False otherwise.
        """
        status = self.get_file_status(project_name, file_path, step)
        if self.is_single_project:
            return False
        if retry_failed:
            return status == 1
        else:
            return status in (1, -1)

    def get_pending_steps(
        self, project_name: str, retry_failed: bool = False
    ) -> List[str]:
        """
        Get the list of pending steps for a project.

        Args:
            project_name: The project name.
            retry_failed: Whether to retry failed steps.

        Returns:
            List of pending steps.
        """
        if project_name not in self.checkpoint_data["projects"]:
            return self.all_steps

        all_steps = self.all_steps
        pending = []

        if self.is_single_project:
            return all_steps

        for step in all_steps:
            status = self.get_project_status(project_name, step)
            if status == 0:
                pending.append(step)
            elif status == -1 and retry_failed:
                pending.append(step)
        return pending
