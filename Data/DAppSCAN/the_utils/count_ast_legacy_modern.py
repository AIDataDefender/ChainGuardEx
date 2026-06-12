import os
import json
from typing import Dict, Any
from Data.DAppSCAN.d_1ast_normalizer import is_legacy_ast

# Folder to traverse
AST_ROOT = r"f:\ChainGuardv2\ChainGuard\Data\DAppSCAN\Processed_Data"

legacy_count = []
modern_count = []
error_count = 0
files_checked = 0

for dirpath, dirnames, filenames in os.walk(AST_ROOT):
    for fname in filenames:
        if fname.endswith("ast_compiled.json"):
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    ast_data = json.load(f)
                files_checked += 1
                
                # Extract project name from path
                # Assuming structure: AST_ROOT/project_name/...
                relative_path = os.path.relpath(fpath, AST_ROOT)
                project_name = relative_path.split(os.sep)[0]
                
                if is_legacy_ast(ast_data):
                    legacy_count.append(project_name)
                else:
                    modern_count.append(project_name)
            except Exception as e:
                print(f"[ERROR] Could not process {fpath}: {e}")
                error_count += 1

# Save the legacy project list to JSON
output_file = "./ast_list.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"legacy_projects": legacy_count, "modern_projects": modern_count}, f, indent=2)

print(f"Checked {files_checked} AST files.")
print(f"Legacy ASTs: {len(legacy_count)}")
print(f"Modern ASTs: {len(modern_count)}")
print(f"Errors: {error_count}")
print(f"Saved legacy project list to {output_file}")
