import re
import os

def remove_extra_whitespace(inputFile, outputFile):
    with open(inputFile, 'r', encoding="utf-8") as fdr:
        content = fdr.read()

    # Remove leading and trailing whitespace from each line
    content = re.sub(r'^[ \t]+|[ \t]+$', '', content, flags=re.M)
    
    # Replace multiple spaces or tabs between words with a single space
    content = re.sub(r'[ \t]+', ' ', content)
    
    # Remove multiple empty lines
    content = re.sub(r'\n\s*\n', '\n', content)

    with open(outputFile, 'w', encoding="utf-8") as fdw:
        fdw.write(content)

# Example usage
if __name__ == '__main__':
    original_dir = "../Data/dataclean/"
    output_dir = "../Data/dataset/"

    dir = os.listdir(original_dir)
    for i in dir:
        print(i)
        remove_extra_whitespace(original_dir + i, output_dir + i)
