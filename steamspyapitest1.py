import requests
import pandas as pd
import time

# 从 Excel 文件中读取 AppID 列表
input_file = 'steam_review_read.xlsx'
df_input = pd.read_excel(input_file)

# 假设 AppID 在第一列，列名为 'appid'
app_ids = df_input['appid'].tolist()

# 存储结果的列表
results = []

# 遍历每个 AppID
for app_id in app_ids:
    response = requests.get(f'https://steamspy.com/api.php?request=appdetails&appid={app_id}')

    if response.status_code == 200:
        data = response.json()
        # 提取 positive 和 negative 值
        positive = data.get('positive', 0)
        negative = data.get('negative', 0)
        results.append({'appid': app_id, 'positive': positive, 'negative': negative})

        # 创建 DataFrame
        df_results = pd.DataFrame(results)

        # 保存到 Excel 文件
        df_results.to_excel('steam_review.xlsx', index=False)

        print(f"数据已保存到 steam_review.xlsx (AppID: {app_id})")

        # 暂停一秒
        time.sleep(1)
    else:
        print(f"Error fetching data for AppID {app_id}")

print("所有数据处理完成。")