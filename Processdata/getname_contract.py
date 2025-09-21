import os
import re

def extract_number_from_filename(filename):
    # Find all numbers in the filename
    numbers = re.findall(r'\d+', filename)
    # Convert to integer, if found, else return a very large number (so files without numbers go last)
    return int(numbers[0]) if numbers else float('inf')

def list_files_in_directory(directory_path, output_file):
    # Get the list of files in the directory
    file_names = os.listdir(directory_path)
    
    # Filter only files (exclude directories)
    file_names = [f for f in file_names if os.path.isfile(os.path.join(directory_path, f))]
    
    # Sort the files based on extracted numbers
    file_names.sort(key=extract_number_from_filename)
    
    # Write the file names to the output file
    with open(output_file, 'w') as f:
        for file_name in file_names:
            f.write(file_name + '\n')

# Usage
directory_path = 'data\\timestamp\\sourcecode_clean'  # Replace with your directory path
output_file = 'data\\name.txt'  # Replace with your desired output file name
list_files_in_directory(directory_path, output_file)
