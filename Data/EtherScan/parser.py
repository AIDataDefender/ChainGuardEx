import os
import json
import pandas as pd
from tqdm import tqdm
from packaging import version
import re
from collections import defaultdict

# ========== SOLIDITY FLATTENER ==========

class SolidityFlattener:
    """Handles flattening of multi-file Solidity contracts"""
    
    def __init__(self, compiler_version: str):
        self.compiler_version = compiler_version
        self.pragma_version = self._extract_version(compiler_version)
        
    def _extract_version(self, compiler_version: str) -> str:
        """Extract version like '0.8.17' from 'v0.8.17+commit.xxx'"""
        if not compiler_version:
            return "0.8.0"
        match = re.search(r'v?(\d+\.\d+\.\d+)', compiler_version)
        return match.group(1) if match else "0.8.0"
    
    def flatten(self, source_code: str) -> str:
        """Main entry point for flattening"""
        if not source_code or not source_code.strip():
            return ""
        
        source_code = source_code.strip()
        
        # Try to parse as JSON (multi-file format)
        if source_code.startswith('{{') or source_code.startswith('{'):
            # Remove double wrapping {{...}}
            if source_code.startswith('{{') and source_code.endswith('}}'):
                source_code = source_code[1:-1]
            
            try:
                data = json.loads(source_code)
                if isinstance(data, dict):
                    sources = data.get('sources', data)
                    if sources and isinstance(sources, dict):
                        return self._flatten_multi_file(sources)
            except:
                pass
        
        # Single file - just clean it
        return self._clean_single_file(source_code)
    
    def _flatten_multi_file(self, sources: dict) -> str:
        """Flatten multiple Solidity files preserving dependencies"""
        if not sources:
            return ""
        
        # Extract content from each file
        files = {}
        for path, data in sources.items():
            content = data.get('content', data) if isinstance(data, dict) else data
            if content:
                files[path] = content
        
        if not files:
            return ""
        
        # Build dependency order with transitive closure
        deps = self._build_dependencies(files)
        ordered = self._topological_sort_with_dfs(deps)
        
        # Detect conflicts
        all_defs = self._collect_all_definitions(files, ordered)
        conflicts = {name: paths for name, paths in all_defs.items() if len(paths) > 1}
        
        # Process each file
        result_parts = []
        non_sol_pragmas = set()
        
        for filepath in ordered:
            if filepath not in files:
                continue
            
            content = files[filepath]
            
            # Extract non-solidity pragmas
            for pragma in self._extract_pragmas(content, exclude_solidity=True):
                non_sol_pragmas.add(pragma)
            
            # Clean and extract definitions
            cleaned = self._extract_code_only(content)
            if not cleaned:
                continue
            
            # Rename conflicts
            if conflicts:
                prefix = self._get_prefix(filepath)
                cleaned = self._rename_conflicts(cleaned, conflicts, filepath, prefix)
            
            result_parts.append(cleaned)
        
        # Build final output
        output = ["// SPDX-License-Identifier: MIT"]
        output.append(f"pragma solidity {self.pragma_version};")
        output.extend(sorted(non_sol_pragmas))
        output.append("")
        output.extend(result_parts)
        
        return '\n\n'.join(output)
    
    def _clean_single_file(self, code: str) -> str:
        """Clean a single file"""
        # Remove existing pragmas and SPDX
        code = re.sub(r'pragma\s+solidity\s+[^;]+;', '', code)
        spdx_match = re.search(r'//\s*SPDX-License-Identifier:\s*\S+', code)
        spdx = spdx_match.group(0) if spdx_match else "// SPDX-License-Identifier: MIT"
        code = re.sub(r'//\s*SPDX-License-Identifier:[^\n]*\n?', '', code)
        
        return f"{spdx}\npragma solidity {self.pragma_version};\n\n{code.strip()}"
    
    def _extract_code_only(self, content: str) -> str:
        """Extract only definitions (contracts, libraries, interfaces, structs, enums, etc.)"""
        # Remove imports
        content = re.sub(r'import\s*\{[^}]*\}\s*from\s*["\'].*?["\'];', '', content, flags=re.DOTALL)
        content = re.sub(r'import\s+["\'].*?["\'];', '', content)
        content = re.sub(r'import\s+.*?\s+from\s+["\'].*?["\'];', '', content)
        
        # Remove code fences
        content = re.sub(r'```[^\n]*\n', '', content)
        content = re.sub(r'```', '', content)
        
        lines = content.split('\n')
        result = []
        in_def = False
        current = []
        brace_count = 0
        paren_count = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Skip file-level noise
            if not in_def:
                if not stripped or stripped.startswith('pragma ') or 'SPDX' in stripped:
                    continue
                if stripped.startswith('//') or stripped.startswith('/*'):
                    continue
            
            # Check for definition start
            is_global_def = re.match(r'^(struct|enum|error|type|using)\s+', stripped)
            is_contract_def = re.match(r'^(contract|library|interface|abstract\s+contract)\s+\w+', stripped)
            is_constant = re.match(r'^\w+\s+constant\s+\w+', stripped)
            
            if not in_def and (is_global_def or is_contract_def or is_constant):
                in_def = True
                current = [line]
                brace_count = line.count('{') - line.count('}')
                paren_count = line.count('(') - line.count(')')
                
                # Single line definition
                if ';' in line and '{' not in line:
                    result.extend(current)
                    result.append('')
                    current = []
                    in_def = False
                elif brace_count == 0 and '{' in line:
                    result.extend(current)
                    result.append('')
                    current = []
                    in_def = False
                continue
            
            if in_def:
                current.append(line)
                brace_count += line.count('{') - line.count('}')
                paren_count += line.count('(') - line.count(')')
                
                # Check completion
                if brace_count == 0 and paren_count == 0 and ('{' in '\n'.join(current) or ';' in line):
                    result.extend(current)
                    result.append('')
                    current = []
                    in_def = False
        
        # Add any incomplete definition
        if current:
            result.extend(current)
        
        return '\n'.join(result).strip()
    
    def _extract_pragmas(self, content: str, exclude_solidity: bool = False) -> list:
        """Extract pragma statements"""
        if exclude_solidity:
            pattern = r'pragma\s+(?!solidity)(\w+(?:\s+\w+)*)\s*;'
        else:
            pattern = r'pragma\s+(\w+(?:\s+\w+)*)\s*;'
        
        matches = re.findall(pattern, content, re.IGNORECASE)
        return [f"pragma {m};" for m in matches]
    
    def _get_prefix(self, filepath: str) -> str:
        """Get file prefix for renaming"""
        filename = os.path.basename(filepath)
        prefix = os.path.splitext(filename)[0]
        return re.sub(r'[^\w]', '_', prefix)
    
    def _collect_all_definitions(self, files: dict, ordered: list) -> dict:
        """Collect all definition names across files"""
        all_defs = defaultdict(list)
        
        for filepath in ordered:
            if filepath not in files:
                continue
            
            content = self._extract_code_only(files[filepath])
            names = self._extract_definition_names(content)
            
            for name in names:
                all_defs[name].append(filepath)
        
        return dict(all_defs)
    
    def _extract_definition_names(self, code: str) -> list:
        """Extract names of all definitions"""
        names = []
        patterns = [
            r'^(contract|library|interface|abstract\s+contract)\s+(\w+)',
            r'^(struct|enum|error|type)\s+(\w+)',
            r'^\w+\s+constant\s+(\w+)'
        ]
        
        for line in code.split('\n'):
            stripped = line.strip()
            for pattern in patterns:
                match = re.match(pattern, stripped)
                if match:
                    # Get the last captured group (the name)
                    name = match.groups()[-1]
                    names.append(name)
                    break
        
        return names
    
    def _rename_conflicts(self, content: str, conflicts: dict, filepath: str, prefix: str) -> str:
        """Rename conflicting definitions"""
        renames = {}
        
        for def_name, paths in conflicts.items():
            if filepath in paths:
                renames[def_name] = f"{prefix}_{def_name}"
        
        if not renames:
            return content
        
        # Apply renames
        for old, new in renames.items():
            pattern = r'\b' + re.escape(old) + r'\b'
            content = re.sub(pattern, new, content)
        
        return content
    
    def _build_dependencies(self, files: dict) -> dict:
        """Build import dependency graph"""
        deps = {}
        all_paths = set(files.keys())
        
        for filepath, content in files.items():
            imports = []
            
            # Extract imports
            patterns = [
                r'import\s+["\']([^"\']+)["\']',
                r'import\s+.*?\s+from\s+["\']([^"\']+)["\']',
                r'import\s*\{[^}]*\}\s*from\s*["\']([^"\']+)["\']'
            ]
            
            for pattern in patterns:
                for match in re.findall(pattern, content):
                    resolved = self._resolve_import(match, filepath, all_paths)
                    if resolved and resolved != filepath:  # Avoid self-loops
                        imports.append(resolved)
            
            deps[filepath] = list(set(imports))  # Remove duplicates
        
        return deps
    
    def _resolve_import(self, import_path: str, current_file: str, all_paths: set) -> str:
        """Resolve import path to actual file"""
        # Remove ./
        if import_path.startswith('./'):
            import_path = import_path[2:]
        
        # Remove ../
        while import_path.startswith('../'):
            import_path = import_path[3:]
        
        # Exact match
        if import_path in all_paths:
            return import_path
        
        # Match by filename
        import_name = os.path.basename(import_path)
        for path in all_paths:
            if os.path.basename(path) == import_name:
                return path
        
        # Match by ending
        for path in all_paths:
            if path.endswith(import_path):
                return path
        
        return None
    
    def _topological_sort_with_dfs(self, deps: dict) -> list:
        """
        Improved topological sort using DFS to handle transitive dependencies.
        This ensures files are ordered so dependencies come before dependents.
        """
        visited = set()
        temp_mark = set()
        result = []
        
        def visit(node):
            if node in temp_mark:
                # Circular dependency detected - skip to avoid infinite loop
                return
            if node in visited:
                return
            
            temp_mark.add(node)
            
            # Visit all dependencies first
            for dep in deps.get(node, []):
                if dep in deps:  # Only visit if it's in our graph
                    visit(dep)
            
            temp_mark.remove(node)
            visited.add(node)
            result.append(node)
        
        # Visit all nodes
        for node in sorted(deps.keys()):  # Sort for deterministic output
            if node not in visited:
                visit(node)
        
        return result
    
    def _topological_sort(self, deps: dict) -> list:
        """
        DEPRECATED: Old implementation - kept for compatibility.
        Use _topological_sort_with_dfs instead.
        """
        in_degree = {node: 0 for node in deps}
        
        for node, dependencies in deps.items():
            in_degree[node] = len([d for d in dependencies if d in deps])
        
        queue = sorted([node for node, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Update dependent nodes
            for other in deps:
                if node in deps[other]:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)
                        queue.sort()
        
        # Add any remaining (circular deps)
        remaining = [n for n in deps if n not in result]
        result.extend(sorted(remaining))
        
        return result


# ========== PRAGMA NORMALIZATION ==========

def normalize_pragma(pragma_str: str, compiler_version: str) -> str:
    """Extract major.minor.x from compiler version"""
    if isinstance(compiler_version, str) and compiler_version.strip():
        match = re.search(r'v?(\d+\.\d+)\.\d+', compiler_version)
        if match:
            return f"{match.group(1)}.x"
    
    if not isinstance(pragma_str, str) or not pragma_str.strip():
        return "unknown"
    
    matches = re.findall(r"\d+\.\d+", pragma_str)
    if matches:
        return f"{matches[0]}.x"
    
    return "unknown"


# ========== MAIN PARSER ==========

def parse_contracts(csv_path: str, output_dir: str, test_rows: int = None):
    """Parse contracts and generate grouped outputs"""
    os.makedirs(output_dir, exist_ok=True)
    raw_code_dir = os.path.join(output_dir, "raw_code")
    os.makedirs(raw_code_dir, exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_path)
    
    if test_rows and test_rows > 0:
        print(f"🧪 TEST MODE: Processing {test_rows} rows")
        df = df.head(test_rows)
    
    required_cols = {'address', 'file_path', 'pragma', 'optimization_used', 
                     'runs', 'compiler_version', 'license_type', 'contract_name'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {required_cols - set(df.columns)}")

    # Normalize versions
    tqdm.pandas(desc="Normalizing versions")
    df["pragma_group"] = df.progress_apply(
        lambda r: normalize_pragma(r["pragma"], r["compiler_version"]), axis=1
    )

    grouped_contracts = {}

    # Process each contract
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing contracts"):
        json_path = row["file_path"]
        
        if not os.path.exists(json_path):
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            continue

        # Parse ABI
        abi = data.get("ABI", "[]")
        try:
            abi_parsed = json.loads(abi) if isinstance(abi, str) else abi
        except:
            abi_parsed = []

        # Flatten source code
        source_code = data.get("SourceCode", "")
        compiler_ver = data.get("CompilerVersion", row["compiler_version"])
        sol_file_path = None
        
        if source_code and source_code.strip():
            try:
                flattener = SolidityFlattener(compiler_ver)
                flattened = flattener.flatten(source_code)
                
                if flattened:
                    contract_name = row["contract_name"] or "Contract"
                    safe_name = re.sub(r'[^\w\-]', '_', contract_name)
                    sol_filename = f"{row['address']}_{safe_name}.sol"
                    full_path = os.path.join(raw_code_dir, sol_filename)
                    
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(flattened)
                    
                    sol_file_path = os.path.join("raw_code", sol_filename)
            except Exception as e:
                print(f"Error flattening {row['address']}: {e}")

        # Build contract data
        contract_data = {
            "contract_name": row["contract_name"],
            "file_path": row["file_path"],
            "sol_file_path": sol_file_path,
            "contract_address": row["address"],
            "language": "Solidity",
            "source_code": source_code,
            "abi": abi_parsed,
            "compiler_version": compiler_ver,
            "optimization_used": bool(int(data.get("OptimizationUsed", row["optimization_used"]))),
            "runs": int(data.get("Runs", row["runs"])) if str(row["runs"]).isdigit() else 200,
            "constructor_arguments": data.get("ConstructorArguments", ""),
            "evm_version": data.get("EVMVersion", "Default"),
            "library": data.get("Library", ""),
            "license_type": data.get("LicenseType", row["license_type"]),
            "proxy": bool(int(data.get("Proxy", "0"))),
            "implementation": data.get("Implementation", ""),
            "swarm_source": data.get("SwarmSource", ""),
        }

        pragma_cat = row["pragma_group"]
        grouped_contracts.setdefault(pragma_cat, []).append(contract_data)

    # Write grouped JSONs
    for pragma_cat, contracts in grouped_contracts.items():
        out_file = os.path.join(output_dir, f"pragma_{pragma_cat.replace('.', '_')}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(contracts, f, indent=2, ensure_ascii=False)

    # Generate summary
    summary = {
        "total_contracts": len(df),
        "by_version": {cat: len(contracts) for cat, contracts in grouped_contracts.items()},
        "by_major_version": {}
    }

    for pragma_cat, count in summary["by_version"].items():
        if pragma_cat == "unknown":
            major = "unknown"
        else:
            match = re.match(r"(\d+\.\d+)", pragma_cat)
            major = match.group(1) if match else "unknown"
        summary["by_major_version"][major] = summary["by_major_version"].get(major, 0) + count

    summary["by_version"] = dict(sorted(summary["by_version"].items()))
    summary["by_major_version"] = dict(sorted(summary["by_major_version"].items()))

    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done!")
    print(f"📝 {len([f for f in os.listdir(raw_code_dir) if f.endswith('.sol')])} .sol files in {raw_code_dir}")
    print(f"📊 Summary: {summary_file}")
    print("\n=== Distribution ===")
    for ver, count in summary["by_major_version"].items():
        print(f"  {ver}: {count} contracts")


# ========== ENTRY POINT ==========

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse and flatten Solidity contracts")
    parser.add_argument("--csv", required=True, help="Path to metadata CSV")
    parser.add_argument("--outdir", default="./grouped_output", help="Output directory")
    parser.add_argument("--test", type=int, metavar="N", help="Test with first N rows")

    args = parser.parse_args()
    parse_contracts(args.csv, args.outdir, args.test)