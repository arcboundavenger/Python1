import requests
import pandas as pd

# 从 API 获取 JSON 数据
url = "http://api.steampowered.com/ISteamApps/GetAppList/v2/"
response = requests.get(url)

# 检查请求是否成功
if response.status_code == 200:
    data = response.json()

    # 提取 apps 列表
    apps = data['applist']['apps']

    # 将 apps 列表转换为 DataFrame
    df = pd.DataFrame(apps)

    # 将 DataFrame 写入 Excel 文件
    df.to_excel('steam_apps.xlsx', index=False)

    print("Excel 文件已生成：steam_apps.xlsx")
else:
    print("请求失败，状态码：", response.status_code)