def merge_files(file1, file2, output_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as out:
        for line1, line2 in zip(f1, f2):
            out.write(line2.rstrip() + ' ' + line1)

file1 = 'label.txt'
file2 = '../Data/data.txt'
output_file = 'output.txt'

merge_files(file1, file2, output_file)
