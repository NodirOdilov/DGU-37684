import os
import json
import pandas as pd
import msgpack

# Функция для объединения JSON-файлов
def merge_json_files(directory, output_file):
    # Создаем список, в который будем добавлять содержимое каждого JSON-файла
    json_list = []

    # Проходимся по всем папкам и файлам внутри указанной директории
    for root, dirs, files in os.walk(directory):
        # Проходимся по всем файлам внутри текущей папки
        for file in files:
            # Проверяем, что текущий файл является JSON-файлом
            if file.endswith('.json'):
                # Открываем файл, считываем содержимое и добавляем его в список
                with open(os.path.join(root, file)) as f:
                    content = json.load(f)
                    json_list.append(content)

    # Создаем выходной файл и записываем в него все JSON-объекты из списка
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_list, f, ensure_ascii=False)

# Указываем директорию, в которой находятся JSON-файлы
directory = './waf-bypass/payload/'

# Указываем имя выходного файла
output_file = 'output_file.json'

# Вызываем функцию для объединения JSON-файлов
merge_json_files(directory, output_file)

# Указываем имя файла для чтения и записи
input_file = 'output_file.json'
output_file = 'test.json'

# Загружаем JSON-объект из файла
with open(input_file, 'r', encoding='utf-8') as f:
    json_list = json.load(f)

# Создаем список для хранения отфильтрованных данных
filtered_data = []

# Проходимся по каждому объекту в списке и извлекаем необходимые данные
for json_obj in json_list:
    for payload in json_obj.get("payload", []):
        if payload.get("URL"):
            blocked_content = payload.get("URL")
            injection = payload.get("BLOCKED")
            
            # Формируем словарь с данными
            data = {
                "payload": blocked_content,
                "injection": injection
            }
            
            # Добавляем словарь в список отфильтрованных данных
            filtered_data.append(data)

# Создаем выходной файл и записываем в него отфильтрованные данные в формате JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

# Чтение данных из msgpack
with open('train.msgpack', 'rb') as f:
    data_msgpack = msgpack.unpack(f, raw=False)

# Конвертирование данных из msgpack в DataFrame
df_msgpack = pd.DataFrame.from_dict(data_msgpack)

# Чтение данных из CSV
df_csv = pd.read_csv('train_info.csv')

df_msgpack = df_msgpack.rename(columns={0: 'id', 1: 'payload'})

df_merged = pd.merge(df_csv, df_msgpack, on='id')

# Создаем список для хранения данных
json_list = []

# Проходимся по каждой строке в DataFrame
for index, row in df_merged.iterrows():
    payload = row['payload']
    injection = row['injection']
    
    # Формируем словарь с данными
    data = {
        "payload": payload,
        "injection": injection
    }
    
    # Добавляем словарь в список данных
    json_list.append(data)

# Создаем выходной файл и записываем в него данные в формате JSON
with open('train.json', 'w', encoding='utf-8') as f:
    json.dump(json_list, f, ensure_ascii=False, indent=4)

# Вывод информации
print("JSON-файлы успешно объединены и отфильтрованы.")
print("Данные из msgpack и CSV успешно объединены и сохранены в JSON.")
