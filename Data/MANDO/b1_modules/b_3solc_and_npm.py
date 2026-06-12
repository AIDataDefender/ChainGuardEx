import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from the_utils.logger import setup_logger

RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.environ.get("LOG_DIR")
if not LOG_DIR:
    LOG_DIR = f"Logs/t{RUN_TIMESTAMP}"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.environ["LOG_DIR"] = LOG_DIR
LOG_FILENAME = Path(LOG_DIR) / "solc_n_npm.log"
logger = setup_logger(str(LOG_FILENAME), log_level=logging.INFO)


DEPENDENCY_MAP = {
    "custom": {
        "openzeppelin-solidity": "openzeppelin-solidity@2.0.0",
        "zeppelin-solidity": "zeppelin-solidity@1.12.0",
        "pancake-swap-core": "@pancakeswap-libs/pancake-swap-core@1.0.1",
        "layerzerolabs": "@layerzerolabs/solidity-examples",
        "aragon": "@aragon/os@4.4.0",
        "0x": "@0x/contracts-utils@4.5.0",
    },
    "0.5": {
        "@openzeppelin/": "@openzeppelin/contracts@2.5.1",
        "@openzeppelin/contracts-upgradeable/": "@openzeppelin/contracts-upgradeable@2.5.1",
        "@openzeppelin/contracts-ethereum-package/": "@openzeppelin/contracts-ethereum-package@2.5.0",
        "@chainlink/": "@chainlink/contracts@0.0.10",
        "@uniswap/": "@uniswap/v2-core@1.0.1",
        "@uniswap/v2-periphery/": "@uniswap/v2-periphery@1.0.0",
        "@uniswap/lib/": "@uniswap/lib@1.1.1",
        "@layerzerolabs/": "@layerzerolabs/solidity-examples",
        "@aragon/": "@aragon/os@4.4.0",
        "@aragon/apps-vault/": "@aragon/apps-vault",
        "@ablack/fundraising-shared-interfaces/": "@ablack/fundraising-shared-interfaces",
        "@0x/": "@0x/contracts-utils@4.5.0",
        "@yield-protocol/": "@yield-protocol/vault-interfaces@0.2.0",
        "@yield-protocol/vault-interfaces/": "@yield-protocol/vault-interfaces@0.2.0",
        "@yield-protocol/utils-v2/": "@yield-protocol/utils-v2@2.0.0",
        "@yield-protocol/yieldspace-interfaces/": "@yield-protocol/yieldspace-interfaces@1.0.0",
        "zeppelin-solidity/": "zeppelin-solidity@1.12.0",
        "openzeppelin-solidity/": "openzeppelin-solidity@2.0.0",
        "@gnosis.pm/safe-contracts/": "@gnosis.pm/safe-contracts",
        "@api3/airnode-protocol/": "@api3/airnode-protocol",
        "@eth-optimism/": "@eth-optimism/contracts",
        "@ablack/": "@ablack/fundraising-shared-interfaces",
        "@ensdomains/": "@ensdomains/ens-contracts",
        "@iexec/": "@iexec/solidity",
        "@pooltogether/": "@pooltogether/v4-core",
        "@nomiclabs/": "@nomiclabs/hardhat-ethers",
        "@opengsn/": "@opengsn/contracts",
        "@pancakeswap/": "@pancakeswap/pancake-swap-core",
        "@airswap/": "@airswap/contracts",
        "@boringcrypto/": "@boringcrypto/boring-solidity",
        "@alium-official/alium-swap-lib/": "@alium-official/alium-swap-lib",
        "@aave/protocol-v2/": "@aave/protocol-v2",
        "@1inch/solidity-utils/": "@1inch/solidity-utils",
        "@aave/aave-stake/": "@aave/aave-stake",
        "@aave/governance-v2/": "@aave/governance-v2",
        "@rari-capital/solmate/": "@rari-capital/solmate@6.0.0",
        "@rari-capital/": "@rari-capital/solmate@6.0.0",
        "@pancakeswap-libs/pancake-swap-core/": "@pancakeswap-libs/pancake-swap-core@0.1.0",
    },
    "0.6": {
        "@openzeppelin/": "@openzeppelin/contracts@3.4.2",
        "@openzeppelin/contracts-upgradeable/": "@openzeppelin/contracts-upgradeable@3.4.2",
        "@openzeppelin/contracts-ethereum-package/": "@openzeppelin/contracts-ethereum-package@3.0.0",
        "@uniswap/": "@uniswap/v2-core@1.0.1",
        "@chainlink/": "@chainlink/contracts@0.1.1",
        "@aave/protocol-v2/": "@aave/protocol-v2@1.0.1",
        "@uniswap/v2-periphery/": "@uniswap/v2-periphery@1.0.0",
        "@uniswap/v3-periphery/": "@uniswap/v3-periphery@1.0.0",
        "@uniswap/lib/": "@uniswap/lib@1.0.1",
        "@layerzerolabs/": "@layerzerolabs/solidity-examples",
        "@aragon/": "@aragon/os@4.4.0",
        "@aragon/apps-vault/": "@aragon/apps-vault",
        "@ablack/fundraising-shared-interfaces/": "@ablack/fundraising-shared-interfaces",
        "@0x/": "@0x/contracts-utils@4.5.0",
        "@yield-protocol/": "@yield-protocol/vault-interfaces@0.2.0",
        "@yield-protocol/vault-interfaces/": "@yield-protocol/vault-interfaces@0.2.0",
        "@yield-protocol/utils-v2/": "@yield-protocol/utils-v2@2.0.0",
        "@yield-protocol/yieldspace-interfaces/": "@yield-protocol/yieldspace-interfaces@1.0.0",
        "zeppelin-solidity/": "zeppelin-solidity@1.12.0",
        "openzeppelin-solidity/": "openzeppelin-solidity@2.3.0",
        "@yearnvaults/": "@yearnvaults/contracts",
        "@sushiswap/core/": "@sushiswap/core-sdk",
        "@yearn/contract-utils/": "@yearn/contract-utils",
        "@lbertenasco/bonded-stealth-tx/": "@lbertenasco/bonded-stealth-tx",
        "@gnosis.pm/safe-contracts/": "@gnosis.pm/safe-contracts",
        "@api3/airnode-protocol/": "@api3/airnode-protocol",
        "@eth-optimism/": "@eth-optimism/contracts",
        "@ablack/fundraising-shared-interfaces/": "@ablack/fundraising-shared-interfaces",
        "@ensdomains/": "@ensdomains/ens-contracts",
        "@iexec/": "@iexec/solidity",
        "@pooltogether/": "@pooltogether/v4-core",
        "@nomiclabs/": "@nomiclabs/hardhat-ethers",
        "@opengsn/": "@opengsn/contracts",
        "@pancakeswap/": "@pancakeswap/pancake-swap-core",
        "@airswap/": "@airswap/contracts",
        "@boringcrypto/": "@boringcrypto/boring-solidity",
        "@alium-official/alium-swap-lib/": "@alium-official/alium-swap-lib",
        "@1inch/solidity-utils/": "@1inch/solidity-utils",
        "@aave/aave-stake/": "@aave/aave-stake",
        "@aave/governance-v2/": "@aave/governance-v2",
        # "@scientix-finance/scientix-contract/": "@scientix-finance/scientix-contract",
        "@rari-capital/solmate/": "@rari-capital/solmate@6.0.0",
        "@rari-capital/": "@rari-capital/solmate@6.0.0",
        "@pancakeswap-libs/pancake-swap-core/": "@pancakeswap-libs/pancake-swap-core@0.1.0",
    },
    "0.7": {
        "@openzeppelin/": "@openzeppelin/contracts@3.4.2",
        "@openzeppelinV3/": "@openzeppelin/contracts@3.4.2",
        "@openzeppelin/contracts-upgradeable/": "@openzeppelin/contracts-upgradeable@3.4.2",
        "@openzeppelin/contracts-ethereum-package/": "@openzeppelin/contracts-ethereum-package@3.0.0",
        "@chainlink/": "@chainlink/contracts@0.2.1",
        "@uniswap/": "@uniswap/v2-core@1.0.1",
        "@uniswap/v2-periphery/": "@uniswap/v2-periphery@1.0.0",
        "@uniswap/lib/": "@uniswap/lib@1.0.1",
        "@layerzerolabs/": "@layerzerolabs/solidity-examples",
        "@aragon/": "@aragon/os@4.4.0",
        "@aragon/apps-vault/": "@aragon/apps-vault",
        "@ablack/fundraising-shared-interfaces/": "@ablack/fundraising-shared-interfaces",
        "@0x/": "@0x/contracts-utils@4.5.0",
        "@yield-protocol/": "@yield-protocol/vault-interfaces@0.2.0",
        "@yield-protocol/vault-interfaces/": "@yield-protocol/vault-interfaces@0.2.0",
        "@yield-protocol/utils-v2/": "@yield-protocol/utils-v2@2.0.0",
        "@yield-protocol/yieldspace-interfaces/": "@yield-protocol/yieldspace-interfaces@1.0.0",
        "zeppelin-solidity/": "zeppelin-solidity@1.12.0",
        "openzeppelin-solidity/": "openzeppelin-solidity@2.3.0",
        "@yearnvaults/": "@yearnvaults/contracts",
        "@sushiswap/core/": "@sushiswap/core-sdk",
        "@yearn/contract-utils/": "@yearn/contract-utils",
        "@lbertenasco/bonded-stealth-tx/": "@lbertenasco/bonded-stealth-tx",
        "@gnosis.pm/safe-contracts/": "@gnosis.pm/safe-contracts",
        "@api3/airnode-protocol/": "@api3/airnode-protocol",
        "@eth-optimism/": "@eth-optimism/contracts",
        "@ablack/fundraising-shared-interfaces/": "@ablack/fundraising-shared-interfaces",
        "@ensdomains/": "@ensdomains/ens-contracts",
        "@iexec/": "@iexec/solidity",
        "@pooltogether/": "@pooltogether/v4-core",
        "@nomiclabs/": "@nomiclabs/hardhat-ethers",
        "@opengsn/": "@opengsn/contracts",
        "@pancakeswap/": "@pancakeswap/pancake-swap-core",
        "@airswap/": "@airswap/contracts",
        "@boringcrypto/": "@boringcrypto/boring-solidity",
        "@alium-official/alium-swap-lib/": "@alium-official/alium-swap-lib",
        "@aave/protocol-v2/": "@aave/protocol-v2",
        "@1inch/solidity-utils/": "@1inch/solidity-utils",
        "@aave/aave-stake/": "@aave/aave-stake",
        "@aave/governance-v2/": "@aave/governance-v2",
        "@pancakeswap-libs/pancake-swap-core/": "@pancakeswap-libs/pancake-swap-core@0.1.0",
    },
    "def": {
        "@openzeppelin/": "@openzeppelin/contracts@4.8.3",
        "@openzeppelin/contracts-upgradeable/": "@openzeppelin/contracts-upgradeable",
        "@openzeppelin/contracts-ethereum-package/": "@openzeppelin/contracts-ethereum-package@4.0.0",
        "@chainlink/": "@chainlink/contracts@0.6.1",
        "@uniswap/": "@uniswap/v3-core@1.0.1",
        "@uniswap/v2-periphery/": "@uniswap/v2-periphery@1.0.0",
        "@uniswap/v3-periphery/": "@uniswap/v3-periphery@1.0.1",
        "@uniswap/lib/": "@uniswap/lib@1.0.1",
        "@layerzerolabs/": "@layerzerolabs/solidity-examples",
        "@aragon/": "@aragon/os@4.4.0",
        "@aragon/apps-vault/": "@aragon/apps-vault",
        "@ablack/fundraising-shared-interfaces/": "@ablack/fundraising-shared-interfaces",
        "@0x/": "@0x/contracts-utils@4.5.0",
        "@yield-protocol/": "@yield-protocol/vault-interfaces@0.2.0",
        "@yield-protocol/vault-interfaces/": "@yield-protocol/vault-interfaces@0.2.0",
        "@yield-protocol/utils-v2/": "@yield-protocol/utils-v2@2.0.0",
        "@yield-protocol/yieldspace-interfaces/": "@yield-protocol/yieldspace-interfaces@1.0.0",
        "zeppelin-solidity/": "zeppelin-solidity@1.12.0",
        "openzeppelin-solidity/": "openzeppelin-solidity@2.3.0",
        "@openzeppelinV2/": "@openzeppelin/contracts@4.8.3",
        "@openzeppelinV3/": "@openzeppelin/contracts@4.8.3",
        "@gnosis.pm/safe-contracts/": "@gnosis.pm/safe-contracts",
        "@api3/airnode-protocol/": "@api3/airnode-protocol",
        "@eth-optimism/": "@eth-optimism/contracts",
        "@ablack/fundraising-shared-interfaces/": "@ablack/fundraising-shared-interfaces",
        "@ensdomains/": "@ensdomains/ens-contracts",
        "@iexec/": "@iexec/solidity",
        "@pooltogether/": "@pooltogether/v4-core",
        "@nomiclabs/": "@nomiclabs/hardhat-ethers",
        "@opengsn/": "@opengsn/contracts",
        "@pancakeswap/": "@pancakeswap/pancake-swap-core",
        "@airswap/": "@airswap/contracts",
        "@boringcrypto/": "@boringcrypto/boring-solidity",
        "@alium-official/alium-swap-lib/": "@alium-official/alium-swap-lib",
        "@aave/protocol-v2/": "@aave/protocol-v2",
        "@1inch/solidity-utils/": "@1inch/solidity-utils",
        "@aave/aave-stake/": "@aave/aave-stake",
        "@aave/governance-v2/": "@aave/governance-v2",
        "@rari-capital/solmate/": "@rari-capital/solmate",
        "@pancakeswap-libs/pancake-swap-core/": "@pancakeswap-libs/pancake-swap-core@0.1.0",
    },
}


def build_remappings(node_modules_dir: str, solc_version: str) -> List[str]:
    """
    Build remappings for solc.

    Args:
        node_modules_dir: Path to node_modules directory.
        solc_version: Solidity version.

    Returns:
        List of remapping strings.
    """
    remappings = []
    solc_major_minor_num = ".".join(solc_version.split(".")[:2])

    # Get map for specific version, fallback to 'def'
    remap_value = DEPENDENCY_MAP.get(solc_major_minor_num)
    if not remap_value:
        remap_value = DEPENDENCY_MAP["def"].copy()
    else:
        remap_value = remap_value.copy()

    remap_value.update(DEPENDENCY_MAP["custom"])

    for prefix, full_pkg_string in remap_value.items():
        # CLEANER LOGIC:
        # 1. Remove version suffix (everything after the last @ if it's a version number)
        # We split by @. If the string starts with @, the first element is empty.
        parts = full_pkg_string.split("@")

        # Handle scoped packages (e.g. @openzeppelin/contracts@3.4.2 -> ['', 'openzeppelin/contracts', '3.4.2'])
        if full_pkg_string.startswith("@"):
            # Reconstruct the package name (e.g. @openzeppelin/contracts)
            # We take everything up to the last part if the last part is a version (starts with digit)
            if parts[-1][0].isdigit():
                package_name = "@" + "@".join(parts[1:-1])
            else:
                # No version specified
                package_name = full_pkg_string
        else:
            # Handle non-scoped (e.g. openzeppelin-solidity@2.0.0)
            package_name = parts[0]

        # 2. Extract the root folder for the remapping
        # If it is @openzeppelin/contracts, we want @openzeppelin/
        # If it is @rari-capital/solmate, we want @rari-capital/solmate/
        if "/" in package_name:
            scope = package_name.split("/")[0]
            if prefix == scope + "/":
                mapping_target = scope
            else:
                mapping_target = package_name
        else:
            mapping_target = package_name

        # Ensure mapping path ends with /
        mapping = f"{prefix}={os.path.join(node_modules_dir, mapping_target)}/"
        remappings.append(mapping)

    return remappings


def collect_imports_from_sol(project_dir: Path) -> List[Tuple[str, str]]:
    """
    Collect all imports from Solidity files in the project.

    Args:
        project_dir: Path to the project directory.

    Returns:
        List of tuples (import_path, file_path).
    """
    imports = []
    IGNORED_IMPORTS = [
        "hardhat/console.sol",
        "forge-std/console.sol",
        "forge-std/Test.sol",
    ]
    import_re = re.compile(r'^\s*import\s+(?:\{[^}]+\}\s+from\s+)?["\']([^"\']+)["\'];')
    for sol_file in project_dir.rglob("*.sol"):
        if any(part == "node_modules" for part in sol_file.parts):
            continue
        try:
            with sol_file.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    m = import_re.match(line)
                    if m:
                        if m.group(1) in IGNORED_IMPORTS:
                            continue
                        imports.append((m.group(1), str(sol_file)))
        except Exception as e:
            logger.error(f"❌  Failed to read {sol_file}: {e}")
    return imports


def resolve_local_import(
    import_path: str, project_dir: Union[Path, str], current_file: Optional[str] = None
) -> Optional[str]:
    """
    Resolve a local import path.

    Args:
        import_path: The import path string.
        project_dir: The project directory.
        current_file: The current file path (for relative imports).

    Returns:
        Resolved absolute path or None.
    """
    if import_path.startswith("./") or import_path.startswith("../"):
        base_dir = os.path.dirname(current_file) if current_file else str(project_dir)
        local_path = os.path.normpath(os.path.join(base_dir, import_path)).replace(
            "\\", "/"
        )
        if os.path.exists(local_path):
            return local_path
    local_path = os.path.join(str(project_dir), import_path).replace("\\", "/")
    if os.path.exists(local_path):
        return local_path
    return None


def resolve_node_import_path(
    import_path: str, project_dir: Union[Path, str], current_file: Optional[str] = None
) -> Optional[str]:
    """
    Resolve a node_modules import path.

    Args:
        import_path: The import path string.
        project_dir: The project directory.
        current_file: The current file path.

    Returns:
        Resolved absolute path or None.
    """
    node_path = os.path.join(str(project_dir), "node_modules", import_path).replace(
        "\\", "/"
    )
    if os.path.exists(node_path):
        return node_path
    parts = import_path.split("/")
    if len(parts) >= 2 and parts[0].startswith("@"):
        file_path = os.path.join(str(project_dir), "node_modules", *parts).replace(
            "\\", "/"
        )
        if os.path.exists(file_path):
            return file_path
    return None


def get_project_pragma_version(project_dir: Path) -> str:
    """
    Get the most compatible Solidity version for the project.

    Args:
        project_dir: The project directory.

    Returns:
        Solidity version string.
    """
    versions = []
    sol_files = [f for f in project_dir.rglob("*.sol") if "node_modules" not in f.parts]

    if not sol_files:
        logger.info(
            f"No Solidity files found in project {project_dir}, use default 0.8.0"
        )
        return "0.8.0"

    for sol_file in sol_files:
        version = extract_pragma_version(sol_file)
        if version:
            versions.append(version)

    if not versions:
        solc_version = "0.8.0"
        logger.info(
            f"[INFO] Could not detect Solidity version, using default {solc_version}"
        )
    else:

        def version_key(v):
            parts = v.split(".")
            while len(parts) < 3:
                parts.append("0")
            return [int(x) for x in parts]

        versions.sort(key=version_key)
        solc_version = versions[-1]
        logger.info(
            f"[INFO] Using Solidity version {solc_version} for dependency installation"
        )
    return solc_version


def extract_pragma_version(sol_file: Union[Path, str]) -> Optional[str]:
    """
    Extract the Solidity version from a file's pragma.

    Args:
        sol_file: Path to the Solidity file.

    Returns:
        Version string or None.
    """
    try:
        if not os.path.exists(sol_file):
            return None
        with open(sol_file, "r", encoding="utf-8") as f:
            content = f.read()
        # Find pragma solidity lines, ignoring comments
        pragma_matches = re.findall(
            r"^\s*pragma solidity\s+([^;]+);", content, re.MULTILINE
        )
        if not pragma_matches:
            return None
        # Collect all version numbers from all pragmas
        all_versions = []
        for pragma_content in pragma_matches:
            versions = re.findall(r"\d+\.\d+(?:\.\d+)?", pragma_content)
            all_versions.extend(versions)
        if not all_versions:
            return None
        # Sort versions and take the lowest (most compatible)
        all_versions.sort(key=lambda v: [int(x) for x in v.split(".")])
        return all_versions[0]
    except (IOError, UnicodeDecodeError):
        logger.error(traceback.print_exc())
        return None


def set_solc_version(project_dir: Path) -> Optional[str]:
    """
    Set the solc version for the project using solc-select.

    Args:
        project_dir: The project directory.

    Returns:
        The set version string or None on failure.
    """
    version = get_project_pragma_version(project_dir)
    try:
        try:
            current_solc = subprocess.run(
                ["solc", "--version"], capture_output=True, text=True
            )
            logger.info(f"Current solc: {current_solc.stdout.strip()}")
        except Exception:
            logger.error(
                "Could not detect local 'solc' binary; solc may not be installed or not in PATH"
            )
        try:
            result = subprocess.run(
                ["solc-select", "versions"], capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError:
            logger.info(
                "[BLUE]"
                + f"solc-select not initialized, installing default solc 0.8.0..."
            )
            subprocess.run(
                ["solc-select", "install", "0.8.0"],
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                ["solc-select", "use", "0.8.0"],
                capture_output=True,
                text=True,
                check=True,
            )
            result = subprocess.run(
                ["solc-select", "versions"], capture_output=True, text=True, check=True
            )
        logger.info("[BLUE]" + f"Attempting to set solc versions via solc-select...")

        if version not in result.stdout:
            install_result = subprocess.run(
                ["solc-select", "install", version], capture_output=True, text=True
            )
            logger.info("[BLUE]" + f"Installing solc {version} via solc-select...")
            if install_result.returncode != 0:
                logger.error(
                    f"❌ Failed to install solc {version} via solc-select: {install_result.stderr}"
                )
                return None
            logger.info("[GREEN]" + f"Installed solc {version} via solc-select")

        use_result = subprocess.run(
            ["solc-select", "use", version], capture_output=True, text=True
        )
        logger.info("[BLUE]" + f"Setting solc {version} via solc-select...")
        if use_result.returncode != 0:
            logger.error(
                f"❌ Failed to use solc-select for version {version}: {use_result.stderr}"
            )
            return None
        logger.info("[GREEN]" + f"Using solc {version} via solc-select")
        return version

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to use solc-select for version {version}: {e}")
        traceback.print_exc()
        return None


def get_current_solc_version() -> Optional[str]:
    """
    Get the currently active solc version.

    Returns:
        Version string or None.
    """
    try:
        res = subprocess.run(["solc", "--version"], capture_output=True, text=True)
        out = res.stdout or res.stderr
        m = re.search(r"Version:\s*(\d+\.\d+\.\d+)", out)
        if m:
            return m.group(1)
        res2 = subprocess.run(
            ["solc-select", "versions"], capture_output=True, text=True
        )
        m2 = re.search(r"->\s*(\d+\.\d+\.\d+)", res2.stdout)
        if m2:
            return m2.group(1)
    except Exception:
        return None
    return None


def ensure_solc_for_file(sol_file: Union[Path, str]) -> bool:
    """
    Ensure the active solc version matches the file's pragma.

    Args:
        sol_file: Path to the Solidity file.

    Returns:
        True if successful, False otherwise.
    """
    file_version = extract_pragma_version(sol_file)
    if not file_version:
        return True

    current = get_current_solc_version()
    if current:
        if ".".join(current.split(".")[:2]) == ".".join(file_version.split(".")[:2]):
            return True

    try:
        versions_res = subprocess.run(
            ["solc-select", "versions"], capture_output=True, text=True
        )
        available = re.findall(r"\d+\.\d+\.\d+", versions_res.stdout)
        install_target = file_version

        if install_target not in available:
            logger.info(
                "[BLUE]"
                + f"Attempting to install solc {install_target} to match {sol_file}"
            )
            install_result = subprocess.run(
                ["solc-select", "install", install_target],
                capture_output=True,
                text=True,
            )
            if install_result.returncode != 0:
                logger.error(
                    f"Failed to install solc {install_target}: {install_result.stderr}"
                )
                return False

        use_result = subprocess.run(
            ["solc-select", "use", install_target], capture_output=True, text=True
        )
        if use_result.returncode != 0:
            logger.error(
                f"Failed to switch solc to {install_target}: {use_result.stderr}"
            )
            return False
        active = get_current_solc_version()
        if active:
            logger.info(
                "[GREEN]"
                + f"Using solc {install_target} (selected). Active solc: {active} (for {sol_file})"
            )
        else:
            logger.info(
                "[GREEN]" + f"Using solc {install_target} (selected) for {sol_file}"
            )
        return True
    except Exception as e:
        logger.error(f"Error setting solc to match {sol_file}: {e}")
        traceback.print_exc()
        return False


def ensure_node_project(project_dir_top: Union[Path, str]) -> None:
    """
    Ensure the project has a package.json, creating one if needed.

    Args:
        project_dir_top: Top-level project directory.
    """
    pkg_file = Path(project_dir_top) / "package.json"
    if pkg_file.exists():
        return
    npm_path = shutil_which_npm()
    if not npm_path:
        logger.info(
            "[BLUE]"
            + f"'npm' not found in PATH. Creating minimal package.json in {project_dir_top} as fallback."
        )
        sys.exit(1)
    logger.info(
        "[BLUE]"
        + f"Initializing npm project in {project_dir_top} using npm at {npm_path}..."
    )
    try:
        res = subprocess.run(
            ["npm", "init", "-y"],
            cwd=str(project_dir_top),
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(res.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"ailed to npm init: {e.stderr}")


def shutil_which_npm() -> Optional[str]:
    """
    Abstracted shutil.which('npm') for easier testing.

    Returns:
        Path to npm executable or None.
    """
    import shutil

    return shutil.which("npm")


def get_installed_version(
    project_dir: Union[Path, str], package_name: str
) -> Optional[str]:
    """
    Reads the package.json of an installed dependency to find its version.

    Args:
        project_dir: Project directory.
        package_name: Package name.

    Returns:
        The version string (e.g., '3.4.2') or None if not found.
    """
    # Handle scoped packages (e.g. @openzeppelin/contracts)
    package_json_path = os.path.join(
        str(project_dir), "node_modules", package_name, "package.json"
    )

    if not os.path.exists(package_json_path):
        return None

    try:
        with open(package_json_path, "r") as f:
            data = json.load(f)
            return data.get("version")
    except Exception:
        return None


def install_dependencies(
    project_dir_top: Union[Path, str], solc_version: str = "0.8.0"
) -> None:
    """
    Smartly installs dependencies:
    1. Checks if dependency is mapped.
    2. Checks if it is installed.
    3. Checks if the INSTALLED version matches the REQUIRED version.
    4. Re-installs if versions mismatch.

    Args:
        project_dir_top: Top-level project directory.
        solc_version: Solidity version.
    """
    # --------------------------
    # Step 1. Collect Solidity versions
    # --------------------------
    logger.info(f"[BLUE]SOLC {solc_version}")
    solc_major_minor_num = ".".join(solc_version.split(".")[:2])

    # --------------------------
    # Step 2. Define dependency maps
    # --------------------------
    if solc_major_minor_num in DEPENDENCY_MAP:
        dependency_map = DEPENDENCY_MAP[solc_major_minor_num]
    else:
        dependency_map = DEPENDENCY_MAP["def"]

    # Combine with custom
    full_map = {**dependency_map, **DEPENDENCY_MAP["custom"]}

    # --------------------------
    # Step 3. Identify required packages
    # --------------------------
    dependencies_to_install = set()
    unresolved_imports = []

    all_imports = collect_imports_from_sol(Path(project_dir_top))

    for imp, filepath in all_imports:
        local_resolved = resolve_local_import(imp, project_dir_top, filepath)
        if local_resolved:
            continue

        matched = False
        for prefix, pkg_string in full_map.items():
            if imp.startswith(prefix) or imp.endswith(prefix):
                matched = True

                # Parse package string to get name and version
                # Example: "@openzeppelin/contracts@3.4.2" or "ds-test"
                parts = pkg_string.rsplit("@", 1)
                if len(parts) == 2 and parts[1][0].isdigit():
                    pkg_name = parts[0]
                    desired_version = parts[1]
                else:
                    pkg_name = pkg_string
                    desired_version = None

                # --------------------------
                # SMART CHECK: Compare Installed vs Desired
                # --------------------------
                installed_version = get_installed_version(project_dir_top, pkg_name)

                if installed_version is None:
                    # Not installed -> Install
                    dependencies_to_install.add(pkg_string)
                elif desired_version and installed_version != desired_version:
                    # Mismatch -> Re-install
                    logger.info(
                        f"[YELLOW]Version mismatch for {pkg_name}: found {installed_version}, need {desired_version}. Reinstalling."
                    )
                    dependencies_to_install.add(pkg_string)
                else:
                    # Correct version exists -> Do nothing
                    pass

                break

        if not matched:
            # Only verify existence for unmapped imports (rare)
            if not resolve_node_import_path(imp, project_dir_top):
                unresolved_imports.append({"import": imp, "file": filepath})

    # --------------------------
    # Step 5. Install in batch
    # --------------------------
    if dependencies_to_install:
        deps_list = [dep for dep in dependencies_to_install if dep]
        logger.info("Installing dependencies:\n- " + "\n- ".join(deps_list))
        try:
            # --legacy-peer-deps is crucial for older projects mixing new/old tools
            subprocess.run(
                ["npm", "install", "--legacy-peer-deps"] + deps_list,
                cwd=str(project_dir_top),
                check=True,
                capture_output=True,
                timeout=600,  # Increased timeout for big installs
            )
            logger.info("[GREEN]" + f"[SUCCESS] Dependencies synced successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] npm install failed: {e.stderr.decode('utf-8')}")
        except subprocess.TimeoutExpired:
            logger.error(f"[ERROR] npm install timed out")
    else:
        logger.info("[GREEN]Dependencies already up to date.")

    # --------------------------
    # Step 6. Report unresolved
    # --------------------------
    if unresolved_imports:
        logger.info(f"[WARNING] Missing deps for {project_dir_top}:")
        for imp in unresolved_imports:
            # Simplified logging
            logger.info(f"Missing: {imp.get('import')} in {imp.get('file')}")
