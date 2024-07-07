import pandas as pd
import os

# Создаем список файлов в директории ./data/
file_paths = ['./data/' + i for i in os.listdir('./data/')]

# Определяем столбцы, которые необходимо удалить
columns_to_drop = [
    'src_ip',
    'dst_ip',
    'dst_port',
    'src_port',
    'timestamp',
    'fwd_psh_flags',
    'bwd_psh_flags',
    'fwd_urg_flags',
    'bwd_urg_flags',
    'flow_byts_s', 
    'flow_pkts_s'
]

# Создаем пустой DataFrame
df_dataset = pd.DataFrame()

# Проходим по каждому файлу в списке
for file_path in file_paths:
    df_data = pd.read_csv(file_path)
    
    # Присваиваем метки в зависимости от названия файла
    if file_path in ['./data/BENIGN.csv', './data/BENIGN1.csv']:
        df_data['Label'] = 0
    elif file_path == './data/http.csv':
        df_data['Label'] = 1
    elif file_path == './data/udp.csv':
        df_data['Label'] = 2
    
    # Добавляем данные в общий DataFrame
    df_dataset = df_dataset.append(df_data, ignore_index=True)

# Удаляем ненужные столбцы
df_dataset = df_dataset.drop(columns=columns_to_drop)

# Сохраняем обработанные данные в файл
df_dataset.to_csv('live_data.csv', index=False)

# Печатаем уникальные значения столбца Label
print(list(df_dataset["Label"].unique()))
