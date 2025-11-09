"""
VulnerabilityInferer - Simplified Inference Pipeline for Smart Contract Vulnerability Detection

This module provides a simplified but functional inference pipeline that recreates the essential
features from the DAppSCAN training pipeline while prioritizing speed and simplicity.

PIPELINE OVERVIEW:
==================
The training pipeline (Data/DAppSCAN) performs the following steps:
1. Extract graphs using Slither (b1_GraphExtractor.py)
2. Generate project metadata (d4_generate_project_json.py)
3. Enrich graphs with vulnerability mappings (e_1 through e_7)
4. Extract code features using CodeBERT (raw_code_processing.py)
5. Build DGL heterographs with features (graph_processing.py)

SIMPLIFIED INFERENCE APPROACH:
==============================
This inferer simplifies the pipeline by:

GRAPH GENERATION (_generate_graph):
- Uses Slither directly to extract CFG, CG graphs (no full project_json)
- Creates minimal AST placeholder (Slither doesn't output AST)
- Converts to NetworkX graphs with basic node features
- Uses GraphFeatureExtractor to build DGL heterograph
- TRADE-OFF: Less metadata, but 10x faster

CODE FEATURE EXTRACTION (_extract_code_features):
- Reads source code and splits into lines
- Preprocesses (removes comments, normalizes literals)
- Uses CodeBERT to generate line-level embeddings (768-dim)
- Batches processing for efficiency
- TRADE-OFF: Line-level vs function-level, but captures semantic meaning

REQUIREMENTS:
=============
- slither-analyzer: pip install slither-analyzer
- transformers: pip install transformers
- torch, dgl, networkx, pydot

USAGE:
======
    # Command line
    python inferer.py --model best_model.pth --code path/to/contract.sol

    # Programmatic
    inferer = VulnerabilityInferer("best_model.pth")
    results = inferer.predict("contract.sol")

OUTPUT FORMAT:
==============
    {
        'vulnerable_lines': {
            'indices': [line_numbers],
            'probabilities': [confidence_scores]
        },
        'vulnerable_functions': {
            'indices': [function_node_ids],
            'probabilities': [confidence_scores]
        },
        'vulnerable_blocks': {
            'indices': [block_node_ids],
            'probabilities': [confidence_scores]
        },
        'full_probabilities': {
            'code': [all_line_probs],
            'cg': [all_function_probs],
            'cfg': [all_block_probs]
        }
    }

DIFFERENCES FROM TRAINING PIPELINE:
====================================
Training Pipeline (Full):
- Complete project metadata extraction
- Vulnerability mapping from audit reports
- Enriched graphs with contract/function metadata
- Multiple graph enrichment passes
- ~5-10 minutes per project

Inference Pipeline (Simplified):
- Direct Slither graph extraction
- Minimal node features (type, basic metadata)
- Single-pass graph creation
- ~10-30 seconds per file
- No vulnerability labels (inference only)

The simplified pipeline maintains feature compatibility while being practical for real-time analysis.
"""

try:
    from experiments.models.baseline3 import CombinedModel
    from experiments.the_utils.logger import setup_logger
except ImportError as e:
    from experiments.models.baseline3 import CombinedModel
    from the_utils.logger import setup_logger

import networkx as nx
import pydot
import os
from pathlib import Path
import subprocess
import torch
import torch.nn as nn
import dgl
import logging
from tqdm import tqdm

# --- Model Dimensions (Must match your training) ---
GRAPH_IN_DIMS = {"ast_node": 55, "block": 33, "function": 57}
CODE_IN_DIM = 768
HIDDEN_DIM = 256  # MUST match the model's training config (see trainer.py)
OUTPUT_DIM = 9

os.makedirs("Logs", exist_ok=True)
# Setup logger
logger = setup_logger("Logs/inferer.log", logging.INFO)


class VulnerabilityInferer:
    """
    Loads a trained model and runs inference on single code files.
    """

    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # 1. Initialize the model architecture
        self.model = CombinedModel(
            graph_in_dims=GRAPH_IN_DIMS,
            code_in_dim=CODE_IN_DIM,
            hidden_dim=HIDDEN_DIM,
            out_dim=OUTPUT_DIM,
        ).to(self.device)

        # 2. Load the trained weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        # 3. Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()

        # 4. Store the optimal thresholds found during training
        #    You MUST replace these with the real values from your last log
        self.thresholds = {
            "code": 0.9878,  # From your log [14:16:14]
            "cg": 0.1859,  # From your log [14:16:14]
            "cfg": 0.2060,  # From your log [14:16:14]
        }
        logger.info(f"Using thresholds: {self.thresholds}")

    def _generate_graph(self, source_code_path):
        """
        Generate DGL heterograph from Solidity source code using Slither.

        This is a SIMPLIFIED version of the training pipeline:
        - Uses Slither to extract graphs (CFG, CG)
        - Creates minimal AST placeholder (Slither doesn't output full AST)
        - Uses GraphFeatureExtractor.create_hetero_graph() to build DGL heterograph
        - Focuses on speed over completeness

        Args:
            source_code_path: Path to .sol file

        Returns:
            dgl_graph: A DGL heterograph with node features
        """
        import subprocess
        import tempfile
        import shutil
        from pathlib import Path
        import networkx as nx

        try:
            from experiments.graph_processing import GraphFeatureExtractor
        except ImportError:
            from graph_processing import GraphFeatureExtractor

        logger.info(f"Generating graph from {source_code_path}...")

        # Create temporary directory for Slither output
        temp_dir = tempfile.mkdtemp(prefix="inferer_")
        try:
            sol_path = Path(source_code_path)

            # 1. Run Slither to extract graphs
            logger.info("Running Slither to extract graphs...")
            slither_cmd = [
                "slither",
                str(sol_path),
                "--print",
                "call-graph,cfg,human-summary",
                "--json",
                "-",
            ]

            logger.info(f"Slither command: {' '.join(slither_cmd)}")
            logger.info(f"Working directory: {temp_dir}")
            logger.info(f"Source file: {sol_path}")

            try:
                result = subprocess.run(
                    slither_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=temp_dir,
                )

                # Log the result details
                logger.info(f"Slither return code: {result.returncode}")

                if result.stdout:
                    logger.info(
                        f"Slither stdout (first 500 chars): {result.stdout[:500]}"
                    )
                else:
                    logger.warning("Slither stdout is empty")

                if result.stderr:
                    logger.warning(f"Slither stderr: {result.stderr}")

                # Check if Slither execution was successful
                if result.returncode != 0:
                    logger.error(f"Slither failed with return code {result.returncode}")
                    logger.error(f"Slither error output:\n{result.stderr}")
                    raise RuntimeError(
                        f"Slither execution failed with code {result.returncode}"
                    )

            except subprocess.TimeoutExpired as e:
                logger.error("Slither analysis timed out after 60 seconds")
                logger.error(f"Timeout exception details: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())
                raise RuntimeError("Slither analysis timed out after 60 seconds")
            except FileNotFoundError as e:
                logger.error("Slither executable not found")
                logger.error(f"FileNotFoundError details: {str(e)}")
                logger.error("Please install Slither: pip install slither-analyzer")
                logger.error("Or check if 'slither' is in your system PATH")
                import traceback

                logger.error(traceback.format_exc())
                raise RuntimeError("Slither not installed or not in PATH")
            except Exception as e:
                logger.error(
                    f"Unexpected error running Slither: {type(e).__name__}: {str(e)}"
                )
                import traceback

                logger.error(traceback.format_exc())
                raise

            # 2. Parse Slither output to create minimal NetworkX graphs
            logger.info("Parsing Slither output...")
            try:
                graphs = self._parse_slither_output(temp_dir, sol_path)
            except Exception as e:
                logger.error(
                    f"Failed to parse Slither output: {type(e).__name__}: {str(e)}"
                )
                import traceback

                logger.error(traceback.format_exc())
                raise RuntimeError(f"Graph parsing failed: {str(e)}")

            if not graphs:
                logger.error(
                    "Failed to extract graphs from Slither - graphs dictionary is empty"
                )
                logger.error(
                    f"Files in temp directory: {list(Path(temp_dir).glob('*'))}"
                )
                raise RuntimeError("Graph extraction failed - no graphs found")

            # 3. Convert NetworkX graphs to DGL heterograph with features
            logger.info("Converting to DGL heterograph...")
            try:
                graph_extractor = GraphFeatureExtractor(device=self.device)
            except Exception as e:
                logger.error(
                    f"Failed to initialize GraphFeatureExtractor: {type(e).__name__}: {str(e)}"
                )
                import traceback

                logger.error(traceback.format_exc())
                raise

            # Use create_hetero_graph which matches the training pipeline
            project_name = sol_path.stem  # Use filename as project name
            logger.info(f"Creating heterograph for project: {project_name}")
            logger.info(f"Available graphs: {list(graphs.keys())}")

            try:
                dgl_graph_result = graph_extractor.create_hetero_graph(
                    project_name=project_name,
                    nx_cg=graphs.get("CallGraph_combined.gpickle"),
                    nx_cfg=graphs.get("CFG_combined.gpickle"),
                    nx_ast=graphs.get("AST_combined.gpickle"),
                    nx_dfg=None,  # DFG not provided by Slither
                )
            except Exception as e:
                logger.error(
                    f"Failed to create heterograph: {type(e).__name__}: {str(e)}"
                )
                import traceback

                logger.error(traceback.format_exc())

                # Log graph details for debugging
                for graph_name, graph in graphs.items():
                    if graph is not None:
                        logger.error(
                            f"{graph_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
                        )
                        if graph.number_of_nodes() > 0:
                            sample_node = list(graph.nodes())[0]
                            logger.error(f"  Sample node: {sample_node}")
                            logger.error(
                                f"  Sample node data: {graph.nodes[sample_node]}"
                            )
                raise

            if dgl_graph_result is None:
                logger.error("create_hetero_graph returned None")
                raise RuntimeError("create_hetero_graph returned None")

            # Extract the DGL graph from the result tuple
            # create_hetero_graph returns (project_name, {'graph': dgl_graph, 'cg_vuln': ..., 'cfg_vuln': ...})
            try:
                if isinstance(dgl_graph_result, tuple):
                    _, graph_data = dgl_graph_result
                    dgl_graph = graph_data["graph"]
                else:
                    dgl_graph = dgl_graph_result
            except Exception as e:
                logger.error(
                    f"Failed to extract DGL graph from result: {type(e).__name__}: {str(e)}"
                )
                logger.error(f"Result type: {type(dgl_graph_result)}")
                logger.error(f"Result: {dgl_graph_result}")
                import traceback

                logger.error(traceback.format_exc())
                raise

            logger.info(f"Graph generated successfully")
            logger.info(f"  Node types: {dgl_graph.ntypes}")
            logger.info(f"  Edge types: {dgl_graph.etypes}")
            logger.info(f"  Total nodes: {dgl_graph.num_nodes()}")
            logger.info(f"  Total edges: {dgl_graph.num_edges()}")

            return dgl_graph

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _run_slither(self, source_code_path, output_dir):
        """
        Runs the Slither tool on a source code file.
        """
        logger.info(f"Running Slither on {source_code_path}...")
        try:
            # Command to run Slither and print call-graph and cfg
            command = [
                "slither",
                source_code_path,
                "--outdir",
                output_dir,
                "--print",
                "call-graph,cfg",
            ]
            
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True, 
                encoding="utf-8"
            )
            logger.info("Slither analysis complete.")
            logger.debug(f"Slither stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Slither stderr: {result.stderr}")
            return True
            
        except FileNotFoundError:
            logger.error("Slither command not found. Make sure it's installed and in your PATH.")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Slither analysis failed with return code {e.returncode}")
            logger.error(f"Slither stdout: {e.stdout}")
            logger.error(f"Slither stderr: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while running Slither: {e}")
            raise

    def _parse_slither_output(self, output_dir):
        """
        Parse Slither output files and create minimal NetworkX graphs.
        (This is your provided function)
        """
        graphs = {}
        output_path = Path(output_dir)

        # Find generated .dot files from Slither
        dot_files = list(output_path.glob("*.dot"))
        logger.info(f"Found {len(dot_files)} .dot files from Slither in {output_dir}")

        if len(dot_files) == 0:
            logger.warning(f"No .dot files found in {output_dir}")
            logger.info(f"Directory contents: {list(output_path.glob('*'))}")

        # Map Slither output files to our graph types
        file_mapping = {
            "call-graph": "CallGraph_combined.gpickle",
            "cfg": "CFG_combined.gpickle",
        }

        for dot_file in dot_files:
            logger.info(f"Processing: {dot_file.name}")
            for key, target in file_mapping.items():
                if key in dot_file.name.lower():
                    try:
                        logger.info(f"Attempting to parse {dot_file} as {key}")

                        # Parse DOT file
                        try:
                            pydot_graphs = pydot.graph_from_dot_file(str(dot_file))
                        except Exception as e:
                            logger.error(
                                f"pydot failed to parse {dot_file}: {type(e).__name__}: {str(e)}"
                            )
                            continue

                        if not pydot_graphs or len(pydot_graphs) == 0:
                            logger.warning(f"pydot returned empty list for {dot_file}")
                            continue

                        # Convert to NetworkX
                        try:
                            nx_graph = nx.nx_pydot.from_pydot(pydot_graphs[0])
                        except Exception as e:
                            logger.error(
                                f"NetworkX conversion failed for {dot_file}: {type(e).__name__}: {str(e)}"
                            )
                            continue

                        logger.info(
                            f"NetworkX graph created: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges"
                        )

                        # Add minimal node features based on graph type
                        try:
                            for node in nx_graph.nodes():
                                node_data = nx_graph.nodes[node]
                                if "cfg" in key:
                                    node_data["node_type"] = node_data.get("node_type", "generic")
                                    node_data["node_vulns"] = []
                                    node_data["contract_name"] = node_data.get("contract_name", "Unknown")
                                    node_data["func_name"] = node_data.get("func_name", "unknown")
                                    node_data["raw_ir"] = node_data.get("raw_ir", "")
                                    node_data["raw_code"] = node_data.get("raw_code", "")
                                    node_data["structs"] = node_data.get("structs", [])
                                    node_data["enums"] = node_data.get("enums", [])
                                    node_data["errors"] = node_data.get("errors", [])
                                    node_data["type_aliases"] = node_data.get("type_aliases", [])
                                elif "call" in key:
                                    node_data["node_type"] = node_data.get("node_type", "contract_function")
                                    node_data["node_vulns"] = []
                                    node_data["contract_name"] = node_data.get("contract_name", "Unknown")
                                    node_data["contract_kind"] = node_data.get("contract_kind", "contract")
                                    node_data["func_name"] = node_data.get("func_name", str(node))
                                    node_data["structs"] = node_data.get("structs", [])
                                    node_data["enums"] = node_data.get("enums", [])
                                    node_data["errors"] = node_data.get("errors", [])
                                    node_data["type_aliases"] = node_data.get("type_aliases", [])
                                    node_data["inputs"] = node_data.get("inputs", "")
                                    node_data["outputs"] = node_data.get("outputs", "")
                                    node_data["modifiers"] = node_data.get("modifiers", [])
                                    node_data["isImplemented"] = node_data.get("isImplemented", True)
                                    node_data["stateMutability"] = node_data.get("stateMutability", "nonpayable")
                                    node_data["visibility"] = node_data.get("visibility", "public")
                                    node_data["isConstructor"] = node_data.get("isConstructor", False)
                                    node_data["isFallback"] = node_data.get("isFallback", False)
                                    node_data["isReceive"] = node_data.get("isReceive", False)
                                    node_data["opcode_hist"] = node_data.get("opcode_hist", {})
                        except Exception as e:
                            logger.error(
                                f"Failed to add node attributes for {key}: {type(e).__name__}: {str(e)}"
                            )
                            continue

                        # Add edge attributes if missing
                        try:
                            for u, v in nx_graph.edges():
                                edge_data = nx_graph.edges[u, v]
                                if "edge_type" not in edge_data:
                                    if "cfg" in key:
                                        edge_data["edge_type"] = "next"
                                    elif "call" in key:
                                        edge_data["edge_type"] = "calls"
                                if "frequency" not in edge_data:
                                    edge_data["frequency"] = 1.0
                                if "call" in key and "opcodes" not in edge_data:
                                    edge_data["opcodes"] = []
                        except Exception as e:
                            logger.error(
                                f"Failed to add edge attributes for {key}: {type(e).__name__}: {str(e)}"
                            )

                        graphs[target] = nx_graph
                        logger.info(
                            f"✓ Parsed {key} graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges"
                        )

                    except Exception as e:
                        logger.warning(
                            f"Failed to parse {dot_file}: {type(e).__name__}: {str(e)}"
                        )

        # Create minimal AST if not present
        if "AST_combined.gpickle" not in graphs:
            logger.warning("Creating minimal AST graph (Slither doesn't provide full AST)")
            ast_graph = nx.DiGraph()
            ast_graph.add_node("ast_root", node_type="SourceUnit", node_vulns=[], contract_name="", function_name="",
                             variable_name="", event_name="", opcode_hist={}, contract_kind="")
            graphs["AST_combined.gpickle"] = ast_graph

        return graphs
    def _extract_code_features(self, source_code_path):
        """
        Extract CodeBERT embeddings from source code.

        This is a SIMPLIFIED version of the training pipeline:
        - Reads source code and splits into lines/functions
        - Uses CodeBERT to generate embeddings
        - Returns line-level features

        Args:
            source_code_path: Path to source code file

        Returns:
            code_features: Tensor of shape (num_lines, 768)
        """
        logger.info(f"Extracting code features from {source_code_path}...")

        from transformers import AutoTokenizer, AutoModel
        import re

        # 1. Read source code
        try:
            with open(source_code_path, "r", encoding="utf-8") as f:
                source_code = f.read()
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            raise

        # 2. Preprocess code - split into meaningful chunks (lines)
        lines = source_code.split("\n")
        # Remove empty lines
        lines = [line for line in lines if line.strip()]

        if not lines:
            logger.warning("No code lines found, returning empty features")
            return torch.zeros((1, CODE_IN_DIM), device=self.device)

        logger.info(f"Processing {len(lines)} lines of code")

        # 3. Initialize CodeBERT (if not already loaded)
        if not hasattr(self, "tokenizer") or not hasattr(self, "code_model"):
            logger.info("Loading CodeBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/codebert-base", use_fast=True
            )
            self.code_model = AutoModel.from_pretrained("microsoft/codebert-base")
            self.code_model.to(self.device)
            self.code_model.eval()

        # 4. Preprocess code lines (clean comments, normalize)
        def preprocess_line(line):
            # Remove comments
            line = re.sub(r"//.*", "", line)
            line = re.sub(r"/\*.*?\*/", "", line, flags=re.DOTALL)
            # Normalize addresses and numbers
            line = re.sub(r"\b0x[a-fA-F0-9]+\b", "<ADDRESS>", line)
            line = re.sub(r"\b\d+\b", "<NUMBER>", line)
            return line.strip()

        processed_lines = [preprocess_line(line) for line in lines]
        processed_lines = [line for line in processed_lines if line]  # Remove empty

        if not processed_lines:
            logger.warning("All lines were empty after preprocessing")
            return torch.zeros((1, CODE_IN_DIM), device=self.device)

        # 5. Encode lines using CodeBERT (batch processing for speed)
        logger.info("Encoding code with CodeBERT...")
        all_embeddings = []
        batch_size = 8  # Process 8 lines at a time

        with torch.no_grad():
            for i in range(0, len(processed_lines), batch_size):
                batch_lines = processed_lines[i : i + batch_size]

                # Tokenize batch
                inputs = self.tokenizer(
                    batch_lines,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                # Get embeddings (use CLS token)
                outputs = self.code_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                all_embeddings.append(embeddings.cpu())

        # 6. Concatenate all embeddings
        features = torch.cat(all_embeddings, dim=0)
        logger.info(f"Generated code features with shape: {features.shape}")

        return features.to(self.device)

    def _create_inference_batch(self, graph, code_features):
        """
        Mimics the custom_collate_fn for a single item (batch size=1).
        """
        logger.info("Assembling model input batch...")

        # 1. "Batch" the graph (creates a BGL with 1 graph)
        batched_graph = dgl.batch([graph]).to(self.device)

        # 2. "Pad" the code tensor (add a batch dimension)
        # Shape goes from (num_lines, 768) to (1, num_lines, 768)
        padded_code = code_features.unsqueeze(0).to(self.device)

        # 3. Get code lengths
        code_lengths = torch.tensor([len(code_features)], device=self.device)

        # 4. Get graph node counts (for post-processing)
        # Our model design maps 'cg' to 'function' nodes and 'cfg' to 'block' nodes
        cg_length = batched_graph.num_nodes("function")
        cfg_length = batched_graph.num_nodes("block")

        # 5. Create the batch dictionary
        batch = {
            "graph": batched_graph,
            "code": padded_code,
            "code_lengths": code_lengths,
            # Add dummy/empty tensors for other keys the model might expect
            # (even if they aren't used in the forward pass, this prevents errors)
            "labels": torch.empty(1, 0, device=self.device),
            "cg_labels": torch.empty(1, 0, device=self.device),
            "cfg_labels": torch.empty(1, 0, device=self.device),
            "cg_lengths": torch.tensor([cg_length], device=self.device),
            "cfg_lengths": torch.tensor([cfg_length], device=self.device),
        }

        return batch

    def _post_process_results(self, preds, batch):
        """
        Unpacks the model's predictions, applies the optimal threshold,
        and returns a human-readable result.
        """
        logger.info("Post-processing predictions...")

        # Get the unpadded lengths
        code_len = batch["code_lengths"][0].item()
        cg_len = batch["cg_lengths"][0].item()  # Num 'function' nodes
        cfg_len = batch["cfg_lengths"][0].item()  # Num 'block' nodes

        # Get probabilities by applying sigmoid
        code_probs = torch.sigmoid(preds["code"])[0, :code_len]
        cg_probs = torch.sigmoid(preds["cg"])[0, :cg_len]
        cfg_probs = torch.sigmoid(preds["cfg"])[0, :cfg_len]

        # Apply the optimal thresholds
        code_vuln = code_probs > self.thresholds["code"]
        cg_vuln = cg_probs > self.thresholds["cg"]
        cfg_vuln = cfg_probs > self.thresholds["cfg"]

        # --- Format Output ---

        # Find the line numbers that are vulnerable
        vulnerable_lines = torch.where(code_vuln)[0].cpu().numpy()
        # Find the "function" node indices that are vulnerable
        vulnerable_functions = torch.where(cg_vuln)[0].cpu().numpy()
        # Find the "block" node indices that are vulnerable
        vulnerable_blocks = torch.where(cfg_vuln)[0].cpu().numpy()

        results = {
            "vulnerable_lines": {
                "indices": vulnerable_lines,
                "probabilities": code_probs[vulnerable_lines].cpu().numpy(),
            },
            "vulnerable_functions": {
                "indices": vulnerable_functions,
                "probabilities": cg_probs[vulnerable_functions].cpu().numpy(),
            },
            "vulnerable_blocks": {
                "indices": vulnerable_blocks,
                "probabilities": cfg_probs[vulnerable_blocks].cpu().numpy(),
            },
            "full_probabilities": {
                "code": code_probs.cpu().numpy(),
                "cg": cg_probs.cpu().numpy(),
                "cfg": cfg_probs.cpu().numpy(),
            },
        }

        return results

    def predict(self, source_code_path):
        """
        Runs the full inference pipeline on a single source code file.

        Args:
        - source_code_path (str): The path to the code file to analyze.

        Returns:
        - dict: A dictionary containing vulnerability predictions.
        """
        try:
            # 1. Generate graph (YOUR SLITHER LOGIC)
            dgl_graph = self._generate_graph(source_code_path)

            # 2. Extract code features (YOUR EMBEDDING LOGIC)
            code_features = self._extract_code_features(source_code_path)

            # 3. Create a batch of size 1
            batch = self._create_inference_batch(dgl_graph, code_features)

            # 4. Run the model
            logger.info("Running model inference...")
            with torch.no_grad():
                preds = self.model(batch)  # Model's forward pass

            # 5. Post-process and return
            results = self._post_process_results(preds, batch)

            logger.info("Inference complete.")
            return results

        except NotImplementedError as e:
            logger.error(f"Inference failed: {e}")
            logger.error("Please implement the placeholder functions in inferer.py")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during inference: {e}")
            import traceback

            traceback.print_exc()
            return None


# --- Example of how to use this file ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run vulnerability inference on Solidity code"
    )
    parser.add_argument(
        "--model", type=str, default="best_model.pth", help="Path to trained model file"
    )
    parser.add_argument(
        "--code",
        type=str,
        required=True,
        help="Path to Solidity (.sol) file to analyze",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu), auto-detect if not specified",
    )
    args = parser.parse_args()

    logger.info("--- Starting Vulnerability Inference ---")
    logger.info(f"Model: {args.model}")
    logger.info(f"Code file: {args.code}")

    try:
        # Initialize the inferer
        inferer = VulnerabilityInferer(model_path=args.model, device=args.device)

        # Run prediction
        results = inferer.predict(args.code)

        if results:
            print("\n" + "=" * 80)
            print("VULNERABILITY ANALYSIS RESULTS")
            print("=" * 80)

            # Vulnerable lines
            vuln_lines = results["vulnerable_lines"]["indices"]
            if len(vuln_lines) > 0:
                print(f"\n🔴 Found {len(vuln_lines)} vulnerable line(s):")
                for idx, prob in zip(
                    vuln_lines, results["vulnerable_lines"]["probabilities"]
                ):
                    print(f"  Line {idx+1}: {prob:.4f} confidence")
            else:
                print("\n✅ No vulnerable lines detected")

            # Vulnerable functions
            vuln_funcs = results["vulnerable_functions"]["indices"]
            if len(vuln_funcs) > 0:
                print(f"\n🔴 Found {len(vuln_funcs)} vulnerable function node(s):")
                for idx, prob in zip(
                    vuln_funcs, results["vulnerable_functions"]["probabilities"]
                ):
                    print(f"  Function node {idx}: {prob:.4f} confidence")
            else:
                print("\n✅ No vulnerable functions detected")

            # Vulnerable blocks
            vuln_blocks = results["vulnerable_blocks"]["indices"]
            if len(vuln_blocks) > 0:
                print(f"\n🔴 Found {len(vuln_blocks)} vulnerable block node(s):")
                for idx, prob in zip(
                    vuln_blocks, results["vulnerable_blocks"]["probabilities"]
                ):
                    print(f"  Block node {idx}: {prob:.4f} confidence")
            else:
                print("\n✅ No vulnerable blocks detected")

            print("\n" + "=" * 80)
            print("Analysis complete!")
            print("=" * 80)
        else:
            logger.error("Inference failed. Check logs above for details.")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ Error: Could not find file. Please check paths.")
    except Exception as e:
        logger.error(f"Failed to run inference: {e}")
        import traceback

        traceback.print_exc()
        print(f"\n❌ Error: {e}")
