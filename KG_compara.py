import json
import os
from collections import Counter
import pandas as pd

## for SPOKE
folder_path = "/Users/apple/Downloads/compare_KG_data/"
df_types = pd.DataFrame()
for file in os.listdir(folder_path):
    if file.startswith("SPOKE_") and file.endswith(".cyjs"):
        file_path = os.path.join(folder_path, file)
        # 读取CYJS文件
        with open(file_path, 'r') as f:
            cyjs_data = json.load(f)
        # 提取类型信息
        types = [node['data']['neo4j_type'] for node in cyjs_data['elements']['nodes']]
        # 计算每种类型的数量
        type_counts = Counter(types)
        # 提取疾病名称
        disease_name = file.replace("SPOKE_", "").replace(".cyjs", "")
        df_types[disease_name] = pd.Series(type_counts)

df_types = df_types.T
df_types_filled = df_types.fillna(0).applymap(lambda x: int(x) if isinstance(x, (int, float)) else 0)
df_types_filled = df_types_filled.iloc[:, :-1]
print(df_types_filled)
df_types_filled.to_csv("/Users/apple/Downloads/compare_KG_data/SPOKE_counts.csv")

## for GENA
df = pd.read_csv('/Users/apple/Downloads/compare_KG_data/GENA.csv')
keywords = ['major depressive', 'anxiety', 'bipolar', 'schizophrenia', 'depression']

final_results = pd.DataFrame()
for key_word in keywords:
    depression_rows = df[df['E1'].str.contains(key_word, case=False, na=False) |
                         df['E2'].str.contains(key_word, case=False, na=False)]

    unique_depression_pairs_df = depression_rows.drop_duplicates(subset=['E1', 'E2'])

    type_E1_counts = {}
    type_E2_counts = {}

    for index, row in unique_depression_pairs_df.iterrows():
        if key_word in row['E1'].lower() and pd.notnull(row['Type_E2']):
            type_E1_counts[row['Type_E2']] = type_E1_counts.get(row['Type_E2'], 0) + 1
        elif key_word in row['E2'].lower() and pd.notnull(row['Type_E1']):
            type_E2_counts[row['Type_E1']] = type_E2_counts.get(row['Type_E1'], 0) + 1

    # 合并两个字典的计数
    combined_counts = {}
    for category in set(type_E1_counts.keys()).union(type_E2_counts.keys()):
        combined_counts[category] = type_E1_counts.get(category, 0) + type_E2_counts.get(category, 0)

    # 将当前关键词的统计结果添加到最终结果 DataFrame 中
    final_results[key_word] = pd.Series(combined_counts)

# 转置结果，使每个关键词成为一行，每种类型成为一列
final_results = final_results.transpose()

# 输出最终结果
print(final_results)
final_results.to_csv('/Users/apple/Downloads/compare_KG_data/GENA_counts.csv')

## for PrimeKG
df = pd.read_csv('/Users/apple/Downloads/compare_KG_data/Primekg.csv')
# 定义所有关键词
keywords = ['major depressive', 'anxiety', 'bipolar', 'schizophrenia', 'depression']

# 初始化最终结果的 DataFrame
final_results = pd.DataFrame()

# 遍历每个关键词
for key_word in keywords:
    # 筛选包含关键词的行
    depression_rows = df[df['x_name'].str.contains(key_word, case=False, na=False) |
                         df['y_name'].str.contains(key_word, case=False, na=False)]

    # 移除重复的 'x_name' 和 'y_name' 组合
    unique_depression_pairs_df = depression_rows.drop_duplicates(subset=['x_name', 'y_name'])

    # 初始化统计字典
    type_E1_counts = {}
    type_E2_counts = {}

    # 遍历唯一组合的行
    for index, row in unique_depression_pairs_df.iterrows():
        if key_word in row['x_name'].lower() and pd.notnull(row['y_type']):
            type_E1_counts[row['y_type']] = type_E1_counts.get(row['y_type'], 0) + 1
        elif key_word in row['y_name'].lower() and pd.notnull(row['x_type']):
            type_E2_counts[row['x_type']] = type_E2_counts.get(row['x_type'], 0) + 1

    # 合并两个类型的计数
    combined_counts = {category: type_E1_counts.get(category, 0) + type_E2_counts.get(category, 0)
                       for category in set(type_E1_counts) | set(type_E2_counts)}

    # 将合并后的计数添加到最终结果中
    final_results[key_word] = pd.Series(combined_counts)

# 调整最终结果的格式
final_results = final_results.fillna(0).astype(int)  # 填充空值为0并转换为整数
final_results = final_results.transpose()  # 转置DataFrame，使关键词成为行索引

# 显示最终结果
print(final_results)

# 保存结果到 CSV 文件
final_results.to_csv('/Users/apple/Downloads/compare_KG_data/PrimeKG_counts.csv')