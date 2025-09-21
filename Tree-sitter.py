
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}


# Load language từ file .so đã build
PY_LANGUAGE = Language('parser/my-languages.so', 'python')

# Khởi tạo Parser
parser = Parser()
parser.set_language(PY_LANGUAGE)

# Ví dụ parse một đoạn code Python
source_code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
"""

tree = parser.parse(bytes(source_code, 'utf8'))
root_node = tree.root_node
print(f'tree {tree}\n')
print(f'root {root_node}\n')

# Duyệt qua AST
def traverse(node, level=0):
    print('  ' * level + f"{node.type}: {node.text.decode('utf8')}")
    for child in node.children:
        traverse(child, level + 1)

traverse(root_node)