import requests
import json
import time
import csv
import pandas as pd

df = pd.read_csv('GameList 2.csv', encoding='latin-1')
appid_list = df['appid']
list_temp = []
api_key = "EGJlJay6LK8qx7LoU9iDux3w8MApXUd0"
headers = {
    'api-key': api_key,
}

for i in range(len(appid_list)):
    url = "https://api.gamalytic.com/game/" + str(int(appid_list[i]))
    params = {
        'format': 'json'
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        # Parse the response content as a JSON object
        data = response.json()

        # 提取所需字段
        price = data.get('price', None)
        reviews = data.get('reviews', None)
        review_score = data.get('reviewScore', None)
        copies_sold = data.get('copiesSold', None)
        revenue = data.get('revenue', None)
        followers = data.get('followers', None)

        # 获取游戏名称和标签
        tags = data.get('tags', [])
        name = data.get('name', 'Unknown')

        # 创建字典并添加到列表
        dict_temp = {
            'appid': str(appid_list[i]),
            'name': name,
            'tags': tags,
            'price': price,
            'reviews': reviews,
            'reviewScore': review_score,
            'copiesSold': copies_sold,
            'revenue': revenue,
            'followers': followers
        }
        list_temp.append(dict_temp)

        # 写入 CSV 文件
        with open('GameList_2_new_gama.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list_temp[0].keys())
            writer.writeheader()
            writer.writerows(list_temp)

        print(f"Request succeed for appid: {dict_temp['appid']}")
    else:
        print(f"Request failed with status code {response.status_code}")

    time.sleep(1)
