import pandas as pd
import os

# Đọc file CSV
df = pd.read_csv('combined_df_mapped_ver3.csv')  # Sửa dấu gạch chéo để phù hợp với các hệ điều hành khác nhau

# Đường dẫn đến thư mục đã có sẵn
sourcecode_dir = '../Data/label_timestamp'



# Kiểm tra xem thư mục có tồn tại không
if not os.path.exists(sourcecode_dir):
    print(f"Thư mục '{sourcecode_dir}' không tồn tại.")
else:
    # Lưu từng hàng của cột 'source code' vào file .sol trong thư mục đã có sẵn
    for idx, source_code in enumerate(df['access_control']):
        file_path = os.path.join(sourcecode_dir, f'{idx+1}.sol')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(source_code))

    print("Đã tách và lưu trữ các mã nguồn thành công.")
