#!/usr/bin/env python3
"""
Single File Pipeline Adapter
Wraps the full DAppSCAN pipeline for single Solidity file analysis

Pipeline Flow:
1. Setup workspace → Copy .sol file
2. b1_GraphExtractor → Extract CFG/IR/ABI
3. e5_preprocess_data → Build CPG (raw heterograph, NO embeddings)
4. dataset.process_single_graph_for_detection:
   → CPG_Processor: Add CodeBERT embeddings + structural features
   → Standardize: Ensure consistent DGL schema
5. Model inference → Vulnerability predictions
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent paths - DAppSCAN is 2 levels up
# Path: python-backend/../.. = Extension/
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, 'DAppSCAN'))

from DAppSCAN.b1_GraphExtractor import process_projects
from DAppSCAN.e5_preprocess_data import process_project_folder
from seed_utils import set_global_seed


class SingleFileAnalyzer:
    """
    Adapter to run the full DAppSCAN pipeline on a single Solidity file
    """
    
    def __init__(self):
        self.temp_dir = None
        self.raw_graph = None  # Store NetworkX graph for node lookup
        self.project_name = None  # Store project name for JSON lookup
        self.dappscan_dir = None  # Store DAppSCAN directory path
    
    def analyze_file(
        self,
        file_path: str,
        model_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run detection pipeline on single file (NO ground truth needed):
        1. Setup temp workspace
        2. Run b1_GraphExtractor → Extract CFG, IR, ABI
        3. Run e5_preprocess_data → Build CPG
        4. Run model inference → Detect vulnerabilities
        
        Returns:
            Dictionary with detected vulnerabilities in VSCode format
        """
        try:
            file_path = Path(file_path).resolve()

            if seed is not None:
                set_global_seed(int(seed), deterministic=True)
                print(f"[Seed] Using seed={seed}", file=sys.stderr)
            
            if not file_path.exists():
                return self._error_response(f"File not found: {file_path}")
            
            # Step 1: Create isolated workspace
            self.temp_dir = Path(tempfile.mkdtemp(prefix="sol_analysis_"))
            project_name = file_path.stem
            
            # Create project structure
            project_dir = self.temp_dir / "contracts" / project_name
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            dest_file = project_dir / file_path.name
            shutil.copy2(file_path, dest_file)
            
            print(f"[Setup] Workspace: {self.temp_dir}", file=sys.stderr)
            print(f"[Setup] Project: {project_name}", file=sys.stderr)
            
            # Step 2: Extract graphs
            print("[Step 1/3] Extracting graphs (CFG, IR, ABI)...", file=sys.stderr)
            
            # Change to DAppSCAN directory for proper imports
            # DAppSCAN is at workspace_root/DAppSCAN (2 levels up from python-backend/)
            original_cwd = os.getcwd()
            workspace_root = Path(__file__).parent.parent.parent  # python-backend/../.. = Extension/
            dappscan_dir = workspace_root / "DAppSCAN"
            
            if not dappscan_dir.exists():
                raise FileNotFoundError(f"DAppSCAN directory not found at {dappscan_dir}")
            
            os.chdir(str(dappscan_dir))
            
            try:
                # Run graph extraction
                process_projects(
                    source_code_dir=str(self.temp_dir / "contracts"),
                    ignore_errors=True,
                    load_checkpoint=False,
                    retry_failed=False,
                    single_project=False,
                    setup_only=False,
                    do_steps=["cfg", "ir", "abi"]
                )
                print("[Step 1/3] ✓ Graph extraction complete", file=sys.stderr)
            finally:
                os.chdir(original_cwd)
            
            # Step 3: Preprocess data (Build CPG)
            print("[Step 2/4] Building CPG...", file=sys.stderr)
            
            extracted_graphs_dir = dappscan_dir / "Extracted_Graphs" / project_name
            
            if not extracted_graphs_dir.exists():
                raise FileNotFoundError(f"Extracted graphs not found at {extracted_graphs_dir}")
            
            os.chdir(str(dappscan_dir))
            try:
                raw_graph = process_project_folder(
                    project_folder=extracted_graphs_dir,
                    to_pydot=False
                )
                print("[Step 2/4] ✓ CPG built successfully", file=sys.stderr)
                
                # Store for later node lookup
                self.raw_graph = raw_graph
                self.project_name = project_name
                self.dappscan_dir = dappscan_dir
            finally:
                os.chdir(original_cwd)
            
            # Step 3: Process graph through CPG_Processor (add embeddings) + dataset.py
            print("[Step 3/4] Processing graph through CPG_Processor + dataset.py...", file=sys.stderr)
            #print(f"[Graph] Raw CPG: {raw_graph.num_nodes()} nodes, {raw_graph.num_edges()} edges", file=sys.stderr)
            
            # Import dataset processor
            from dataset import process_single_graph_for_detection
            
            # Process graph through FULL pipeline:
            # 1. CPG_Processor: Add CodeBERT embeddings + structural features
            # 2. Standardize: Ensure consistent schema for model
            print("[Graph] Running CPG_Processor to add embeddings...", file=sys.stderr)
            dgl_graph, local_to_global = process_single_graph_for_detection(
                raw_graph=raw_graph,
                project_name=project_name,
                rand_seed=seed,
            )
            print(f"[Graph] local_to_global mapping: {len(local_to_global) if local_to_global else 0} entries", file=sys.stderr)
            print(f"[Graph] ✓ Final graph: {dgl_graph.num_nodes()} nodes, {dgl_graph.num_edges()} edges", file=sys.stderr)
            print(f"[Graph] Node types: {dgl_graph.ntypes}", file=sys.stderr)
            print(f"[Graph] Edge types: {len(dgl_graph.canonical_etypes)}", file=sys.stderr)
            if dgl_graph.ntypes:
                print(f"[Graph] Has embeddings: {'feat' in dgl_graph.nodes[dgl_graph.ntypes[0]].data}", file=sys.stderr)

            print(f"[Step 3/4] ✓ Graph ready with embeddings for model", file=sys.stderr)
            
            # Step 5: Run Model Detection (REQUIRED for detection mode)
            if not model_path or not os.path.exists(model_path):
                return self._error_response("Model path required for vulnerability detection")
            
            print(f"[Step 4/4] Running ML detection...", file=sys.stderr)
            vulnerabilities = self._run_model_detection(
                dgl_graph, dappscan_dir, project_name, file_path, model_path, raw_graph, local_to_global, seed
            )
            print("Data mapped", vulnerabilities)
            print(f"[Step 4/4] ✓ Detected {len(vulnerabilities)} vulnerabilities", file=sys.stderr)
            
            return {
                "status": "success",
                "file": str(file_path),
                "vulnerabilities": vulnerabilities
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return self._error_response(str(e))
        finally:
            # Cleanup temp directory
            self._cleanup()
    
    def _run_model_detection(
        self,
        dgl_graph: Any,
        dappscan_dir: Path, 
        project_name: str, 
        original_file: Path,
        model_path: str,
        raw_graph: Any = None,
        local_to_global: Any = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run ML model DETECTION (no ground truth needed)
        Pipeline: Standardized DGL graph → model → vulnerability predictions
        """
        import torch
        import dgl
        
        try:
            if seed is not None:
                set_global_seed(int(seed), deterministic=True)

            # Import model architecture
            workspace_root = Path(__file__).parent.parent.parent
            model_dir = workspace_root / "model"
            sys.path.insert(0, str(model_dir))
            
            from proto_3 import CascadedHeteroModel
            
            # Validate input graph
            if not hasattr(dgl_graph, 'ntypes') or not hasattr(dgl_graph, 'canonical_etypes'):
                raise TypeError(f"Invalid DGL graph: {type(dgl_graph)}")
            
            print(f"[ML] Input graph: {dgl_graph.num_nodes()} nodes, {dgl_graph.num_edges()} edges", file=sys.stderr)
            
            # Initialize model with architecture
            print(f"[ML] Initializing model architecture...", file=sys.stderr)
            
            # Extract relation names from graph
            rel_names = [('cfg_node', 'cf_false', 'cfg_node'), ('cfg_node', 'return_call', 'cfg_node'), ('cfg_node', 'df', 'cfg_node'), ('ast_node', 'ast_to_cfg', 'cfg_node'), ('cfg_node', 'cf', 'cfg_node'), ('ast_node', 'ast_child', 'ast_node'), ('cfg_node', 'call', 'cfg_node')]
            print(f"[ML] Graph has {len(rel_names)} relation types", file=sys.stderr)
            
            # Model hyperparameters (must match training config)
            node_dims = {'ast_node': 834, 'cfg_node': 806}
            edge_dims = {rel: 768 for rel in rel_names}  # Dynamic edge dims
            hidden_dim = 128
            out_dim = 8  # Number of vulnerability classes
            stage = "3"  # Node-level classification
            
            model = CascadedHeteroModel(
                node_dims=node_dims,
                edge_dims=edge_dims,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                rel_names=rel_names,
                stage=stage
            )
            
            # Load trained weights
            print(f"[ML] Loading weights from {model_path}", file=sys.stderr)
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            state_dict = None
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load with strict=False to handle edge type mismatches
            # Model trained with different edge types (cf_false, return_call, etc.)
            # Current graph may have different edge types (cf, call, df, etc.)
            print(f"[ML] Loading state dict (strict=False to handle edge type differences)...", file=sys.stderr)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"[ML] Warning: {len(missing_keys)} missing keys (will use random init)", file=sys.stderr)
            if unexpected_keys:
                print(f"[ML] Warning: {len(unexpected_keys)} unexpected keys in checkpoint (ignored)", file=sys.stderr)
            
            model.eval()
            print(f"[ML] ✓ Model loaded successfully", file=sys.stderr)
            
            # Run model inference with pre-loaded DGL graph
            print("[ML] Running model inference...", file=sys.stderr)
            with torch.no_grad():
                # Prepare batch dict as model expects
                batch_dict = {
                    "graph": dgl_graph
                }
                
                print("batch", batch_dict)
                # Forward pass
                predictions = model(batch_dict)
            
            print(f"[ML] ✓ Inference complete")
            print("Predictions:", predictions)
            print("length of predictions:", len(predictions))
            for k, v in predictions.items():
                print(f" - {k}: {torch.sigmoid(v)}")
            # Convert predictions to vulnerabilities
            vulnerabilities = self._predictions_to_vulnerabilities(predictions, local_to_global, original_file, raw_graph, dappscan_dir, project_name)  
            
            return vulnerabilities
            
        except Exception as e:
            print(f"[ML] Detection error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
    
    def match_local_to_global(self, local_idx, ln_type, local_to_global):
        """
        Map DGL local index to NetworkX global ID.
        
        Args:
            local_idx: DGL local node index (tensor or int)
            ln_type: Node type (e.g., 'cfg_node', 'ast_node')
            local_to_global: Mapping dict {local_idx: {node_type: global_id}}
        
        Returns:
            global_id (int) or None if not found
        """
        # Convert tensor to int if needed
        if hasattr(local_idx, 'item'):
            local_idx = int(local_idx.item())
        else:
            local_idx = int(local_idx)
        
        if local_to_global is None or not local_to_global:
            print(f"[Mapping] WARNING: No local_to_global mapping available, using local_idx={local_idx} as fallback", file=sys.stderr)
            return local_idx
        
        if local_idx not in local_to_global:
            print(f"[Mapping] ERROR: local_idx={local_idx} not found in mapping", file=sys.stderr)
            return None
        
        node_type_map = local_to_global[local_idx]
        if ln_type in node_type_map:
            global_id = node_type_map[ln_type]
            print(f"[Mapping] ✓ Mapped local_idx={local_idx} ({ln_type}) → global_id={global_id}", file=sys.stderr)
            return global_id
        
        print(f"[Mapping] ERROR: Node type '{ln_type}' not found for local_idx={local_idx}. Available types: {list(node_type_map.keys())}", file=sys.stderr)
        return None
    
    def _fetch_node_data_by_global_id(
        self,
        global_idx: int,
        node_type: str,
        raw_graph: Any,
        dappscan_dir: Path,
        project_name: str
    ) -> Dict[str, Any]:
        """
        Given a global index:
        1. Map it back to the NetworkX node with global_id
        2. Fetch all node data from NetworkX graph
        3. Match to the JSON index (ast_cfg_cpg_index.json)
        4. Return compiled data in JSON format
        """
        compiled_data = {
            "global_idx": int(global_idx),
            "node_type": node_type,
            "nx_node_data": None,
            "json_index_match": None,
            "error": None
        }
        
        try:
            # Step 1: Find NetworkX node with matching global_id
            if raw_graph is None:
                compiled_data["error"] = "Raw graph (NetworkX) not available"
                return compiled_data
            
            # NetworkX graph stores nodes with 'global_id' attribute
            matching_node = None
            for node_id, node_data in raw_graph.nodes(data=True):
                if node_id == global_idx and node_data.get('node_type') == node_type:
                    matching_node = (node_id, node_data)
                    break
            
            if matching_node is None:
                compiled_data["error"] = f"No NetworkX node found with global_id={global_idx}, node_type={node_type}"
                return compiled_data
            
            node_id, node_data = matching_node
            
            # Step 2: Fetch all node data
            # Convert non-serializable data to strings
            nx_data_serializable = {}
            for key, value in node_data.items():
                try:
                    json.dumps(value)  # Test if serializable
                    nx_data_serializable[key] = value
                except (TypeError, ValueError):
                    nx_data_serializable[key] = str(value)
            
            compiled_data["nx_node_data"] = {
                "node_id": str(node_id),
                "attributes": nx_data_serializable
            }
            
            # Step 3: Match to JSON index
            json_index_path = dappscan_dir / "ProcessedData" / "benign" / project_name / "ast_cfg_cpg_index.json"
            
            if json_index_path.exists():
                with open(json_index_path, 'r') as f:
                    json_index = json.load(f)
                
                # Match by node type and ID
                matching_records = []
                for key, record in json_index.items():
                    # Check if this record matches our node
                    if node_type == 'ast_node' and 'ast_id' in record:
                        # Try to match by ast_id
                        if node_data.get('id') == record['ast_id']:
                            matching_records.append({
                                "key": key,
                                "record": record,
                                "match_type": "ast_id"
                            })
                    elif node_type == 'cfg_node' and 'cfg_id' in record:
                        # Try to match by cfg_id
                        if node_data.get('id') == record['cfg_id']:
                            matching_records.append({
                                "key": key,
                                "record": record,
                                "match_type": "cfg_id"
                            })
                
                compiled_data["json_index_match"] = {
                    "index_file": str(json_index_path),
                    "matches_found": len(matching_records),
                    "matches": matching_records
                }
            else:
                compiled_data["json_index_match"] = {
                    "index_file": str(json_index_path),
                    "exists": False,
                    "error": "JSON index file not found"
                }
        
        except Exception as e:
            import traceback
            compiled_data["error"] = str(e)
            compiled_data["traceback"] = traceback.format_exc()
        
        return compiled_data

    def _extract_location_from_node_data(
        self,
        node_data: Dict[str, Any],
        source_file: Path,
        raw_graph: Any = None,
        global_idx: int = None
    ) -> tuple:
        """
        Extract precise location information from node data.
        For AST nodes: use 'src' field directly
        For CFG nodes: find linked AST node via 'ast_to_cfg' edge
        Returns: (line_from, line_to, snippet, contract_name, function_name)
        """
        line_from = 1
        line_to = 1
        snippet = ""
        contract_name = None
        function_name = None
        
        try:
            # Read source file to map byte offsets to lines
            if source_file.exists():
                with open(source_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                    source_lines = source_code.split('\n')
            else:
                source_code = ""
                source_lines = []
            
            # Extract from NetworkX node data
            nx_data = node_data.get('nx_node_data', {})
            if nx_data:
                attributes = nx_data.get('attributes', {})
                node_type = attributes.get('node_type', '')
                
                # Extract contract and function names 
                contract_name = attributes.get('contract')
                function_name = attributes.get('function')
                
                # For CFG nodes: try to find linked AST node OR use expression field
                if 'cfg' in str(node_type).lower():
                    # Try method 1: Find linked AST node via 'ast_to_cfg' edge
                    if raw_graph is not None and global_idx is not None:
                        print(f"[Location] CFG node detected, searching for linked AST node...", file=sys.stderr)
                        for source_id, target_id, edge_data in raw_graph.edges(data=True):
                            edge_type = edge_data.get('edge_type', '')
                            # Check if this edge links to our CFG node
                            if 'ast_to_cfg' in str(edge_type).lower() and str(target_id) == str(global_idx):
                                # Found AST node that links to this CFG node
                                ast_node_data = raw_graph.nodes.get(source_id, {})
                                ast_src = ast_node_data.get('src', '')
                                if ast_src and ':' in ast_src:
                                    print(f"[Location] Found linked AST node {source_id} with src={ast_src}", file=sys.stderr)
                                    attributes['src'] = ast_src  # Use AST node's src
                                    break
                    
                    # Method 2: If no AST link found, try to search by expression in source code
                    if 'src' not in attributes or not attributes.get('src'):
                        expression = attributes.get('expression', '').strip()
                        if expression and expression != '' and source_lines:
                            print(f"[Location] Searching source code for expression: '{expression[:50]}...'", file=sys.stderr)
                            # Clean up expression for matching (remove extra spaces, normalize)
                            expr_cleaned = ' '.join(expression.split())
                            
                            # Search in source code
                            for line_num, line in enumerate(source_lines, start=1):
                                line_cleaned = ' '.join(line.split())
                                if expr_cleaned in line_cleaned or line_cleaned in expr_cleaned:
                                    line_from = line_num
                                    line_to = line_num
                                    snippet = line.strip()
                                    print(f"[Location] Found expression at line {line_num}: '{snippet[:50]}...'", file=sys.stderr)
                                    break
                
                # Parse 'src' field: "start:length:fileId" format
                src = attributes.get('src', '')
                if src and ':' in src:
                    parts = src.split(':')
                    if len(parts) >= 2:
                        try:
                            byte_start = int(parts[0])
                            byte_length = int(parts[1])
                            byte_end = byte_start + byte_length
                            
                            # Convert byte offsets to line numbers
                            line_from = source_code[:byte_start].count('\n') + 1
                            line_to = source_code[:byte_end].count('\n') + 1
                            
                            # Extract snippet (limit to 3 lines for readability)
                            if line_from <= len(source_lines):
                                snippet_lines = source_lines[line_from-1:min(line_to, line_from+2)]
                                snippet = '\n'.join(snippet_lines).strip()
                                # Limit snippet length
                                if len(snippet) > 150:
                                    snippet = snippet[:147] + '...'
                        except (ValueError, IndexError) as e:
                            print(f"[Location] Error parsing src '{src}': {e}", file=sys.stderr)
                
                # Fallback: use line field if available
                if line_from == 1 and 'line' in attributes:
                    try:
                        line_from = int(attributes['line'])
                        line_to = line_from
                        if line_from <= len(source_lines):
                            snippet = source_lines[line_from-1].strip()
                    except (ValueError, IndexError):
                        pass
        
        except Exception as e:
            print(f"[Location] Error extracting location: {e}", file=sys.stderr)
        
        return line_from, line_to, snippet, contract_name, function_name


    def _predictions_to_vulnerabilities(
        self,
        predictions: Any,
        local_to_global,
        original_file: Path,
        raw_graph: Any = None,
        dappscan_dir: Path = None,
        project_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Convert model predictions to vulnerability list.
        Handles heterograph predictions with multiple node types (cfg_node, ast_node).
        """
        import torch
        
        vulnerabilities = []
        
        # Map prediction keys to actual NetworkX node types
        NODE_TYPE_MAPPING = {
            'ast_logits': 'ast_node',
            'cfg_logits': 'cfg_node',
        }
        
        try:
            # predictions is a dict: {node_type: tensor([num_nodes, num_classes])}
            print(f"[ML] Processing predictions for {len(predictions)} node types", file=sys.stderr)
            
            # Threshold for detection (confidence > 0.5)
            threshold = 0.5
            
            # Use the local_to_global mapping passed from dataset
            if local_to_global is not None:
                print(f"[ML] local_to_global type: {type(local_to_global)}", file=sys.stderr)
                if isinstance(local_to_global, dict):
                    print(f"[ML] Using local_to_global mapping with {len(local_to_global)} entries", file=sys.stderr)
                    # Show first few entries for debugging
                    sample_items = list(local_to_global.items())[:3]
                    for idx, mapping in sample_items:
                        print(f"[ML]   local_idx={idx} -> {mapping}", file=sys.stderr)
                else:
                    print(f"[ML] ERROR: local_to_global is not a dict: {local_to_global}", file=sys.stderr)
            else:
                print(f"[ML] WARNING: No local_to_global mapping available", file=sys.stderr)
            
            for node_type, logits in predictions.items():
                print(f"[ML] Processing {node_type}: shape {logits.shape}", file=sys.stderr)
                
                # Map to actual NetworkX node type
                nx_node_type = NODE_TYPE_MAPPING.get(node_type, node_type)
                print(f"[ML] Mapped {node_type} -> {nx_node_type} for NetworkX lookup", file=sys.stderr)
                
                # Debug: Check if we have mapping for this node type (only if dict)
                if local_to_global and isinstance(local_to_global, dict):
                    try:
                        mapped_for_this_type = sum(1 for idx_map in local_to_global.values() if isinstance(idx_map, dict) and nx_node_type in idx_map)
                        print(f"[ML] Found {mapped_for_this_type} {nx_node_type} entries in local_to_global", file=sys.stderr)
                        # Show max local_idx for this type
                        max_local_idx = max((idx for idx, type_map in local_to_global.items() if isinstance(type_map, dict) and nx_node_type in type_map), default=-1)
                        print(f"[ML] Max local_idx for {nx_node_type}: {max_local_idx}, but model has {logits.shape[0]} nodes", file=sys.stderr)
                    except Exception as e:
                        print(f"[ML] Error checking mapping stats: {e}", file=sys.stderr)
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                
                # Get max probability and predicted class for each node
                max_probs, predicted_classes = probs.max(dim=1)
                
                # Find nodes with confidence above threshold
                detected_mask = max_probs > threshold
                detected_indices = detected_mask.nonzero(as_tuple=True)[0]
                
                print(f"[ML] {node_type}: {len(detected_indices)} detections above threshold", file=sys.stderr)
                
                for local_idx in detected_indices:
                    confidence = max_probs[local_idx].item()
                    vuln_class = predicted_classes[local_idx].item()
                    global_idx = self.match_local_to_global(local_idx, nx_node_type, local_to_global)
                    
                    # Initialize node_location_data with default empty dict
                    node_location_data = {}
                    
                    # === REQUESTED FUNCTIONALITY AT LINE 354 ===
                    # Given global_idx: map back to NX node, fetch data, match JSON, save to test_location.json
                    if global_idx is not None and raw_graph is not None and dappscan_dir is not None:
                        node_location_data = self._fetch_node_data_by_global_id(
                            global_idx=global_idx,
                            node_type=nx_node_type,  # Use mapped node type
                            raw_graph=raw_graph,
                            dappscan_dir=dappscan_dir,
                            project_name=project_name
                        )
                        
                        # Save to test_location.json
                        test_location_file = dappscan_dir / "test_location.json"
                        try:
                            # Load existing data or create new
                            if test_location_file.exists():
                                with open(test_location_file, 'r') as f:
                                    test_data = json.load(f)
                            else:
                                test_data = {"detections": []}
                            
                            # Append this detection's location data
                            detection_entry = {
                                "detection_timestamp": str(Path(original_file).name),
                                "local_idx": int(local_idx.item()) if hasattr(local_idx, 'item') else int(local_idx),
                                "global_idx": int(global_idx),
                                "node_type": node_type,  # Original prediction key
                                "nx_node_type": nx_node_type,  # Mapped NetworkX type
                                "confidence": float(confidence),
                                "vuln_class": int(vuln_class),
                                "location_data": node_location_data
                            }
                            test_data["detections"].append(detection_entry)
                            
                            # Save back to file
                            with open(test_location_file, 'w') as f:
                                json.dump(test_data, f, indent=2)
                            
                            print(f"[ML] ✓ Saved node location data to {test_location_file}", file=sys.stderr)
                        except Exception as e:
                            print(f"[ML] Warning: Could not save to test_location.json: {e}", file=sys.stderr)
                    # === END REQUESTED FUNCTIONALITY ===
                    
                    # Skip if mapping failed (no global_idx found)
                    if global_idx is None:
                        print(f"[ML] Skipping vulnerability - failed to map local_idx={local_idx} to global_idx", file=sys.stderr)
                        continue
                    
                    # Map class to vulnerability type
                    vuln_info = self._class_to_vulnerability_info(vuln_class)
                    
                    # Extract precise location from node data (pass raw_graph and global_idx for CFG nodes)
                    line_from, line_to, snippet, contract_name, function_name = self._extract_location_from_node_data(
                        node_location_data, original_file, raw_graph, global_idx
                    )
                    
                    vuln = {
                        'file': original_file.name,
                        'contract': contract_name or 'Unknown',
                        'function': function_name or 'Unknown',
                        'line_from': line_from,
                        'line_to': line_to,
                        'snippet': snippet or '',
                        'owasp_id': vuln_info['swc_id'],
                        'owasp_name': vuln_info['name'],
                        'suggestion': vuln_info['suggestion'],
                        'node_type': nx_node_type,
                        'confidence': confidence
                    }
                    vulnerabilities.append(vuln)
            
            print(f"[ML] ✓ Converted to {len(vulnerabilities)} total vulnerabilities")
        
        except Exception as e:
            print(f"[ML] Error converting predictions: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        
        return vulnerabilities
    
    def _class_to_vulnerability_info(self, class_idx: int) -> Dict[str, str]:
        """Map model class index to vulnerability information"""
        # Based on OWASP vulnerability classes from training
        VULNERABILITY_MAP = {
            0: {
                'swc_id': 'SC01:2025',
                'name': 'Access Control',
                'suggestion': 'Add proper access control modifiers (onlyOwner, onlyRole) and validate caller permissions'
            },
            1: {
                'swc_id': 'SC03:2025',
                'name': 'Logic Errors',
                'suggestion': 'Review business logic carefully, add comprehensive tests, and validate state transitions'
            },
            2: {
                'swc_id': 'SC04:2025',
                'name': 'Lack of Input Validation',
                'suggestion': 'Validate all inputs with require() statements, check bounds, and sanitize external data'
            },
            3: {
                'swc_id': 'SC05:2025',
                'name': 'Reentrancy Attack',
                'suggestion': 'Follow checks-effects-interactions pattern, use ReentrancyGuard, and update state before external calls'
            },
            4: {
                'swc_id': 'SC06:2025',
                'name': 'Unchecked External Calls',
                'suggestion': 'Always check return values of external calls using require() or handle failure cases'
            },
            5: {
                'swc_id': 'SC08:2025',
                'name': 'Integer Overflow and Underflow',
                'suggestion': 'Use SafeMath library or Solidity 0.8+ with built-in overflow checks'
            },
            6: {
                'swc_id': 'SC09:2025',
                'name': 'Insecure Randomness',
                'suggestion': 'Use Chainlink VRF or commit-reveal schemes instead of block.timestamp or blockhash'
            },
            7: {
                'swc_id': 'SC10:2025',
                'name': 'Denial of Service',
                'suggestion': 'Avoid unbounded loops, limit gas consumption, and use pull-over-push payment patterns'
            }
        }
        
        return VULNERABILITY_MAP.get(class_idx, {
            'swc_id': 'SWC-UNKNOWN',
            'name': f'Unknown Vulnerability (Class {class_idx})',
            'suggestion': 'Review the code carefully and follow Solidity best practices'
        })
    
    def _cleanup(self):
        """Clean up temporary directory"""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                print(f"[Cleanup] Removed temp directory", file=sys.stderr)
            except Exception as e:
                print(f"[Warning] Failed to cleanup {self.temp_dir}: {e}", file=sys.stderr)
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "status": "error",
            "error": message,
            "vulnerabilities": []
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Single File Pipeline Analyzer")
    parser.add_argument('--file', required=True, help='Path to Solidity file')
    parser.add_argument('--model', default=None, help='Path to trained model')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible initialization')
    
    args = parser.parse_args()
    
    analyzer = SingleFileAnalyzer()
    result = analyzer.analyze_file(args.file, model_path=args.model, seed=args.seed)
    
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
