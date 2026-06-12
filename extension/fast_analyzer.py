#!/usr/bin/env python3
"""
Single File Analyzer - Optimized version for VSCode Extension
Fast analysis without full pipeline overhead
"""

import sys
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any

# Simple regex-based patterns for common vulnerabilities
VULNERABILITY_PATTERNS = {
    'SWC-101': {
        'name': 'Integer Overflow and Underflow',
        'pattern': r'(\+\+|--|\+|\-|\*|/|\*\*)\s*(?!SafeMath)',
        'severity': 'High',
        'description': 'Arithmetic operation without SafeMath protection',
        'suggestion': 'Use SafeMath library or Solidity 0.8+ with built-in overflow checks'
    },
    'SWC-105': {
        'name': 'Unprotected Ether Withdrawal',
        'pattern': r'\.transfer\(|\.send\(',
        'severity': 'Critical',
        'description': 'Potential unprotected withdrawal',
        'suggestion': 'Add proper access control (onlyOwner, require checks)'
    },
    'SWC-107': {
        'name': 'Reentrancy',
        'pattern': r'\.call\{value:|\.call\.value\(',
        'severity': 'Critical',
        'description': 'Potential reentrancy vulnerability with external call',
        'suggestion': 'Follow checks-effects-interactions pattern, use ReentrancyGuard'
    },
    'SWC-112': {
        'name': 'Delegatecall to Untrusted Callee',
        'pattern': r'delegatecall\(',
        'severity': 'High',
        'description': 'Delegatecall can be dangerous if callee is not trusted',
        'suggestion': 'Ensure delegatecall target is trusted and validated'
    },
    'SWC-115': {
        'name': 'Authorization through tx.origin',
        'pattern': r'tx\.origin',
        'severity': 'Medium',
        'description': 'Using tx.origin for authorization is insecure',
        'suggestion': 'Use msg.sender instead of tx.origin for authorization'
    },
    'SWC-120': {
        'name': 'Weak Sources of Randomness',
        'pattern': r'block\.(timestamp|number|difficulty|blockhash)',
        'severity': 'Medium',
        'description': 'Block properties are not secure sources of randomness',
        'suggestion': 'Use Chainlink VRF or other secure randomness sources'
    },
    'SWC-131': {
        'name': 'Presence of unused variables',
        'pattern': r'(?:uint|int|bool|address|string)\s+(\w+)\s*;',
        'severity': 'Low',
        'description': 'Unused variable detected',
        'suggestion': 'Remove unused variables to optimize gas'
    }
}


def analyze_solidity_file(file_path: str) -> Dict[str, Any]:
    """
    Fast regex-based analysis for immediate feedback
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'status': 'error', 'error': 'File not found', 'vulnerabilities': []}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        lines = code.split('\n')
        vulnerabilities = []
        
        # Track function contexts
        current_function = None
        function_stack = []
        
        for line_num, line in enumerate(lines, start=1):
            # Track function context
            func_match = re.search(r'function\s+(\w+)', line)
            if func_match:
                current_function = func_match.group(1)
            
            # Check for vulnerabilities
            for swc_id, pattern_data in VULNERABILITY_PATTERNS.items():
                if re.search(pattern_data['pattern'], line):
                    # Avoid duplicates
                    vuln = {
                        'file': str(file_path.name),
                        'line_from': line_num,
                        'line_to': line_num,
                        'column_from': 0,
                        'column_to': len(line),
                        'severity': pattern_data['severity'],
                        'swc_id': swc_id,
                        'owasp_id': _map_swc_to_owasp(swc_id),
                        'swc_name': pattern_data['name'],
                        'function_name': current_function or 'global',
                        'description': pattern_data['description'],
                        'suggestion': pattern_data['suggestion'],
                        'confidence': 0.75  # Regex-based lower confidence
                    }
                    vulnerabilities.append(vuln)
        
        return {
            'status': 'success',
            'file': str(file_path),
            'vulnerabilities': vulnerabilities
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'vulnerabilities': []
        }


def _map_swc_to_owasp(swc_id: str) -> str:
    """Map SWC to OWASP ID"""
    mapping = {
        'SWC-101': 'A9',
        'SWC-105': 'A5',
        'SWC-107': 'A9',
        'SWC-112': 'A9',
        'SWC-115': 'A5',
        'SWC-120': 'A9',
        'SWC-131': 'A10'
    }
    return mapping.get(swc_id, 'Unknown')


def main():
    if len(sys.argv) < 3:
        print(json.dumps({
            'status': 'error',
            'error': 'Usage: fast_analyzer.py --file <path>',
            'vulnerabilities': []
        }))
        sys.exit(1)
    
    if sys.argv[1] != '--file':
        print(json.dumps({
            'status': 'error',
            'error': 'Invalid argument',
            'vulnerabilities': []
        }))
        sys.exit(1)
    
    file_path = sys.argv[2]
    result = analyze_solidity_file(file_path)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
