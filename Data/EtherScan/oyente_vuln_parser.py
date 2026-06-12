#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ============================================================
# In-scope findings ONLY
# ============================================================

FINDINGS = {
    "Callstack Depth Attack Vulnerability",
    "Transaction-Ordering Dependency",
    "Timestamp Dependency",
    "Re-Entrancy Vulnerability",
    "Integer Overflow",
    "Integer Underflow",
    "Parity Multisig Bug 2",
}

# ============================================================
# Regexes (STRICT)
# ============================================================

LOCATION = re.compile(
    r"^(?:INFO:symExec:)?(?P<file>/.+\.sol):(?P<line>\d+):(?P<col>\d+): Warning: (?P<name>.*?)\."
)

SNIPPET = re.compile(r"^\s+(?P<code>.+)$")
FUNC = re.compile(r"\bfunction\b")

# ============================================================
# Solidity helpers (unchanged logic)
# ============================================================


def read_lines(sol_file: Path) -> List[str]:
    return sol_file.read_text(encoding="utf-8", errors="ignore").splitlines()


def find_enclosing_function_start(
    lines: List[str], line_no: int
) -> Optional[int]:
    i = line_no - 1
    while i >= 0:
        if FUNC.search(lines[i]):
            return i + 1
        i -= 1
    return None


def extract_function_block(
    sol_file: Path, start_line: int
) -> Tuple[str, int]:
    lines = read_lines(sol_file)
    total = len(lines)

    i = start_line - 1
    collected = []
    brace_depth = 0
    seen_brace = False

    # collect signature
    while i < total:
        line = lines[i]
        collected.append(line)

        if "{" in line:
            seen_brace = True
            brace_depth += line.count("{")
            brace_depth -= line.count("}")
            i += 1
            break

        i += 1

    if not seen_brace:
        return "\n".join(collected), start_line + len(collected) - 1

    # collect body
    while i < total:
        line = lines[i]
        collected.append(line)

        brace_depth += line.count("{")
        brace_depth -= line.count("}")

        if brace_depth == 0:
            return "\n".join(collected), i + 1

        i += 1

    return "\n".join(collected), i

def complete_parentheses(
    lines: List[str],
    start_line: int,
    initial: str,
) -> Tuple[str, int]:
    """
    Complete a single Solidity statement by balancing parentheses,
    but STOP at statement boundary (not entire blocks).
    """
    open_p = initial.count("(")
    close_p = initial.count(")")

    collected = [initial]
    i = start_line

    # if already balanced, return immediately
    if open_p <= close_p:
        return initial, start_line

    while i < len(lines):
        line = lines[i]
        collected.append(line)

        open_p += line.count("(")
        close_p += line.count(")")

        stripped = line.strip()

        # ✅ stop conditions
        if open_p <= close_p:
            # common Solidity statement endings
            if (
                stripped.endswith(");")
                or stripped.endswith(")")
                or stripped.endswith("{")
                or ") {" in stripped
                or "){" in stripped
            ):
                return "\n".join(collected), i + 1

        # 🔴 hard safety stop: never consume more than ~10 lines
        if len(collected) > 10:
            return "\n".join(collected), i + 1

        i += 1

    return "\n".join(collected), i


# ============================================================
# Oyente parser (CLEAN)
# ============================================================


def parse_oyente(log: List[str], source_root: Path) -> Dict:
    findings: List[Dict] = []
    i = 0

    while i < len(log):
        line = log[i].rstrip("\n")
        m = LOCATION.match(line)
        if not m:
            i += 1
            continue

        name = m.group("name").strip()
        if name not in FINDINGS:
            i += 1
            continue

        sol_path = Path(m.group("file"))
        start_line = int(m.group("line"))

        snippet = None
        if i + 1 < len(log):
            sm = SNIPPET.match(log[i + 1])
            if sm:
                snippet = sm.group("code").rstrip()

        sol_file = sol_path if sol_path.exists() else source_root / sol_path.name
        if not sol_file.exists():
            i += 2
            continue

        lines = read_lines(sol_file)
        line_end = start_line
        final_snippet = snippet

        # function-level extraction
        if snippet and FUNC.search(snippet):
            func_start = find_enclosing_function_start(lines, start_line)
            if func_start:
                block, end = extract_function_block(sol_file, func_start)
                final_snippet = block
                start_line = func_start
                line_end = end
        elif snippet:
            completed, end = complete_parentheses(lines, start_line, snippet)
            final_snippet = completed
            line_end = end

        findings.append({
            "name": name,
            "filename": str(sol_file),
            "line": start_line,
            "line_end": line_end,
            "snippet": final_snippet,
        })

        i += 2

    return {
        "errors": [],
        "fails": [],
        "infos": [],
        "findings": findings,
    }


# ============================================================
# CLI (unchanged behavior)
# ============================================================

def find_result_logs(result_dir: Path) -> List[Path]:
    logs = []
    for root, _, files in os.walk(result_dir):
        if "result.log" in files:
            logs.append(Path(root) / "result.log")
    return logs


def main():
    ap = argparse.ArgumentParser("Oyente vulnerability reparser")
    ap.add_argument("--result-dir", required=True, type=Path)
    ap.add_argument("--source-root", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--name", default="result_reparse.json")
    args = ap.parse_args()

    logs = find_result_logs(args.result_dir)
    if not logs:
        raise FileNotFoundError("No result.log found")

    for log_path in logs:
        log_lines = log_path.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()

        parsed = parse_oyente(log_lines, args.source_root)

        if args.output_dir:
            out_dir = args.output_dir / log_path.parent.name
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = log_path.parent

        out_file = out_dir / args.name
        out_file.write_text(json.dumps(parsed, indent=4))
        print(f"[+] Wrote {out_file}")


if __name__ == "__main__":
    main()
