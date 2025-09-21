import json

# Mở file JSONL và đọc từng dòng
with open('../Data/data.jsonl', 'r', encoding='utf-8') as file:
    data = []
    for line in file:
        data.append(json.loads(line))

# Lấy 1000 dòng đầu tiên
datatest = data[:1000]

# Lưu lại 1000 dòng đầu tiên vào file JSONL mới, mỗi đối tượng trên một dòng
with open('../Data/datatest.jsonl', 'w', encoding='utf-8') as file:
    for entry in datatest:
        file.write(json.dumps(entry, ensure_ascii=False) + '\n')
