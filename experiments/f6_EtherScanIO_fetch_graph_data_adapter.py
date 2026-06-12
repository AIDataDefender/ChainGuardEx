import csv
import os
import pickle
import sys
import traceback
from pathlib import Path

from colorama import Fore, Style
import json
import networkx as nx
from tqdm import tqdm


# Ensure local packages (e.g., c2_build_CPG_modules) remain importable when depickling
DATA_DIR = Path(r"/mnt/d/KLTN2/DatasetEtherScanio/EtherScanio/").resolve()
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))


def print_error(message):
    """Print error messages in red"""
    print(f"{Fore.RED}{message}{Style.RESET_ALL}")


PROCESSED_DATA_DIR = Path(
    "D:/KLTN2/DatasetEtherScanio/EtherScanio/ProcessedData/success").as_posix()


def f6_fetch_EtherScanIO_data(root=None, is_test=False):
    try:
        if root:
            global PROCESSED_DATA_DIR
            PROCESSED_DATA_DIR = root
        cpg_graphs = []
        vuln_maps = []
        project_list = [p for p in os.listdir(
            PROCESSED_DATA_DIR) if p != "benign"]
        if is_test:
            # Limit to first 5 projects for testing
            project_list = project_list[1:3]
        for project_name in tqdm(project_list, desc="Fetching projects"):
            project_path = os.path.join(PROCESSED_DATA_DIR, project_name)
            if not os.path.isdir(project_path):
                continue
            try:
                cpg_graph = pickle.load(
                    open(os.path.join(project_path, "cpg_graph.gpickle"), "rb")
                )
            except Exception as e:
                print_error(
                    f"Failed to load CPG graph for project {project_name}: {e}"
                )
                continue
            vuln_mapping_dict = json.load(
                open(os.path.join(project_path, "vuln_node_mappings.json"), "r")
            )

            cpg_graphs.append((project_name, cpg_graph))
            vuln_maps.append((project_name, vuln_mapping_dict))
        return cpg_graphs, vuln_maps
    except Exception as e:
        print_error(f"Error in f6_fetch_DAppSCAN_data: {e}")
        traceback.print_exc()
        return [], []
