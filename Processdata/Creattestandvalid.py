import random

# Đường dẫn đến file train.txt
input_file = '../Data/data.txt'

# Đường dẫn đến các file đầu ra
valid_file = '../Data/valid.txt'
test_file = '../Data/test.txt'

# Đọc nội dung từ file train.txt
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Random shuffle the lines
random.shuffle(lines)

# Tính toán số lượng mẫu cho mỗi file
total_lines = len(lines)
split_index = total_lines // 2

# Chia nội dung thành hai phần
valid_lines = lines[:split_index]
test_lines = lines[split_index:]

# Ghi nội dung vào file valid.txt
with open(valid_file, 'w', encoding='utf-8') as f:
    f.writelines(valid_lines)

# Ghi nội dung vào file test.txt
with open(test_file, 'w', encoding='utf-8') as f:
    f.writelines(test_lines)

print(f"Đã chia file thành '{valid_file}' và '{test_file}' thành công.")
