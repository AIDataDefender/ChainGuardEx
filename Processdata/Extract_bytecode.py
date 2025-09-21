import pandas as pd
import os


df = pd.read_csv('combined_df_mapped_ver3.csv')  

bytecode_dir = '../Data/bytecode'



if not os.path.exists(bytecode_dir):
    print(f"Thư mục '{bytecode_dir}' không tồn tại.")
else:
    for idx, bytecode in enumerate(df['bytecode']):
        bytecode = str(bytecode)
        if(bytecode.startswith('[SEP]')):
            bytecode = bytecode[len('[SEP]'):].strip()
        file_path = os.path.join(bytecode_dir, f'{idx+1}.sol')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(bytecode))

    print("Đã tách và lưu trữ các mã nguồn thành công.")
