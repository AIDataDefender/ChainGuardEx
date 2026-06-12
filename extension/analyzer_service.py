#!/usr/bin/env python3
"""
Solidity Vulnerability Analyzer Service
Bridge between VSCode Extension and Python ML pipeline
"""

import sys
import os
import json
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import redirect_stdout

# Add parent directory to path to import existing modules
# Path: python-backend/../.. = Extension/
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, workspace_root)

# Redirect all stdout from imports to stderr to keep JSON clean
import io
_original_stdout = sys.stdout
sys.stdout = sys.stderr

from single_file_adapter import SingleFileAnalyzer
from seed_utils import parse_seed, set_global_seed

# Restore stdout after imports
sys.stdout = _original_stdout


class VulnerabilityAnalyzer:
    """
    Main analyzer class that orchestrates the detection pipeline:
    1. Extract graphs (b1_GraphExtractor)
    2. Preprocess data (e5_preprocess_data)
    3. Load dataset (dataset.py)
    4. Run model inference
    """
    
    def __init__(self, model_path: Optional[str] = None, seed: Optional[int] = None):
        self.model_path = model_path
        self.seed = seed
        self.model = None
        self._load_model()
        
        # Initialize single file analyzer (uses your full pipeline)
        self.pipeline_analyzer = SingleFileAnalyzer()
    
    def _load_model(self):
        """Load the trained model"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                import torch
                self.model = torch.load(self.model_path, map_location='cpu')
                self.model.eval()
                print(f"Model loaded from {self.model_path}", file=sys.stderr)
            except Exception as e:
                print(f"Failed to load model: {e}", file=sys.stderr)
        else:
            print("No model provided - using rule-based detection", file=sys.stderr)
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single Solidity file
        
        Args:
            file_path: Path to .sol file
            
        Returns:
            Dictionary containing vulnerability results
        """
        try:
            file_path = Path(file_path).resolve()
            
            if not file_path.exists():
                return self._error_response(f"File not found: {file_path}")
            
            if not file_path.suffix == '.sol':
                return self._error_response(f"Not a Solidity file: {file_path}")
            
            # Add ~/.local/bin to PATH for slither and other tools
            local_bin = os.path.expanduser("~/.local/bin")
            if local_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH', '')}"
                print(f"[Setup] Added {local_bin} to PATH", file=sys.stderr)
            
            # Use the integrated pipeline adapter
            print("=== Running Full DAppSCAN Pipeline ===", file=sys.stderr)
            print(f"File: {file_path}", file=sys.stderr)
            print(f"Mode: {'ML Model' if self.model else 'Rule-based'}", file=sys.stderr)
            
            # Run your complete pipeline: b1_GraphExtractor → e5_preprocess_data
            result = self.pipeline_analyzer.analyze_file(
                str(file_path),
                model_path=self.model_path,
                seed=self.seed,
            )
            
            return result
            
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return self._error_response(str(e))
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "status": "error",
            "error": message,
            "vulnerabilities": []
        }


def main():
    parser = argparse.ArgumentParser(description="Solidity Vulnerability Analyzer Service")
    parser.add_argument('--file', required=True, help='Path to Solidity file to analyze')
    parser.add_argument('--model', help='Path to trained model')
    parser.add_argument('--output', default='json', choices=['json', 'text'], help='Output format')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible pipeline/model initialization (overrides SOLIDITY_SECURITY_SEED if set)'
    )
    
    args = parser.parse_args()

    # Seed as early as possible to stabilize any random initialization.
    seed = args.seed
    if seed is None:
        seed = parse_seed(os.getenv('SOLIDITY_SECURITY_SEED'))
    if seed is not None:
        set_global_seed(int(seed), deterministic=True)
        print(f"[Seed] Using seed={seed}", file=sys.stderr)
    
    # Initialize analyzer
    analyzer = VulnerabilityAnalyzer(model_path=args.model, seed=seed)
    
    # Redirect stdout to stderr during analysis to keep JSON output clean
    _stdout_backup = sys.stdout
    sys.stdout = sys.stderr
    
    # Analyze file
    result = analyzer.analyze_file(args.file)
    
    # Restore stdout for JSON output
    sys.stdout = _stdout_backup
    
    # Output result
    if args.output == 'json':
        # Ensure only JSON goes to stdout, suppress any prints from analysis
        sys.stdout = _original_stdout
        print(json.dumps(result, indent=2))
    else:
        if result['status'] == 'error':
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)
        
        print(f"\nAnalysis Results for: {result['file']}")
        print("=" * 80)
        
        vulns = result['vulnerabilities']
        if not vulns:
            print("✅ No vulnerabilities found!")
        else:
            print(f"Found {len(vulns)} vulnerabilities:\n")
            for i, vuln in enumerate(vulns, 1):
                print(f"{i}. [{vuln['severity']}] {vuln['swc_id']}: {vuln['swc_name']}")
                print(f"   Lines: {vuln['line_from']}-{vuln['line_to']}")
                if vuln.get('function_name'):
                    print(f"   Function: {vuln['function_name']}")
                print(f"   {vuln['description']}")
                if vuln.get('suggestion'):
                    print(f"   💡 {vuln['suggestion']}")
                print()


if __name__ == '__main__':
    main()
