import pandas as pd
import os


df = pd.read_csv('combined_df_mapped_ver3.csv')  


sourcecode_dir = '../Data/sourcecode'

output_json = 'Data/dat'


if not os.path.exists(sourcecode_dir):
    print(f"Thư mục '{sourcecode_dir}' không tồn tại.")
else:
    for idx, source_code in enumerate(df['sourcecode']):
        file_path = os.path.join(sourcecode_dir, f'{idx+1}.sol')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(source_code)

    print("Đã tách và lưu trữ các mã nguồn thành công.")
