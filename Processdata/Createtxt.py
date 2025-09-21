import os


sourcecode_dir = '../Data/label_reentrancy'


output_file = 'label.txt'

if not os.path.exists(sourcecode_dir):
    print(f"Thư mục '{sourcecode_dir}' không tồn tại.")
else:
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for idx, filename in enumerate(sorted(os.listdir(sourcecode_dir))):
            if filename.endswith('.sol'):
                file_path = os.path.join(sourcecode_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as sol_file:
                    content = sol_file.read()
                    out_file.write(f'{content}\n')

    print(f"Đã trích xuất và lưu trữ nội dung vào file '{output_file}' thành công.")
