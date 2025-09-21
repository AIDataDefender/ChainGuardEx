import os
import json

def extract_sol_code_to_jsonl(sourcecode_directory,bytecode_directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        sourcecode_filenames = sorted(os.listdir(sourcecode_directory))
        bytecode_filenames = sorted(os.listdir(bytecode_directory))
        for sourcecode_filename, bytecode_filename in zip(sourcecode_filenames,bytecode_filenames):
            if sourcecode_filename.endswith('.sol') and bytecode_filename.endswith('.sol'):
                source_filepath = os.path.join(sourcecode_directory, sourcecode_filename)
                byte_filepath = os.path.join(bytecode_directory, bytecode_filename)
                with open(source_filepath, 'r', encoding='utf-8') as source_file, \
                     open(byte_filepath,'r', encoding='utf-8') as byte_file:
                        sourcecode = source_file.read()
                        bytecode = byte_file.read()
                        base_filename = os.path.splitext(sourcecode_filename)[0]
                        json_object = {
                            "source": sourcecode,
                            "byte" : bytecode,
                            "idx": base_filename
                        }
                        jsonl_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')


bytecode_directory = '../Data/bytecode'
sourcode_directory = '../Data/dataset'
output_file = '../Data/data.jsonl'


extract_sol_code_to_jsonl(sourcode_directory,bytecode_directory, output_file)
