import pandas as pd

# 读取 XLSX 文件
file_path = "Steam Games 2024_Filled with MC and Twitch.xlsx"  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 保留 AppID 列
app_ids = df['AppID']

# 对 Categories 列做 one-hot encoding
categories = df['Categories'].str.get_dummies(sep=',')

# 对 Genres 列做 one-hot encoding
genres = df['Genres'].str.get_dummies(sep=',')

# 对 Genres 列做 one-hot encoding
Supported_languages = df['Supported languages'].str.get_dummies(sep=',')

# 合并原始 AppID 与编码结果
df_encoded = pd.concat([app_ids, Supported_languages, categories, genres], axis=1)

# 保存结果到新文件 (XLSX 格式)
output_path = "Steam Games 2024_Filled with MC and Twitch_Encoded.xlsx"
df_encoded.to_excel(output_path, index=False)

print(f"One-hot encoding 完成！结果已保存到 {output_path}")