import requests
import json
import time
import csv
import pandas as pd

df = pd.read_csv('GameList 2.csv', encoding='latin-1')
appid_list = df['appid']
list_temp = []
api_key = ""
headers = {
    'api-key': api_key,
}
for i in range(len(appid_list)):
    url = "https://api.gamalytic.com/game/" + str(appid_list[i])
    params = {
        'format': 'json'
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        # Parse the response content as a JSON object
        data = response.json()
        json_data = json.dumps(data, indent=4)
        with open("SteamGaData.json", "w") as f:
            # Write the JSON string to the txt file
            f.write(json_data)
        # 读取JSON文件
        with open('SteamGaData.json') as f:
            data = json.load(f)

        # 获取所有价格的值
        tags = data['tags']
        name = data['name']
        copiesSold = data['copiesSold']
        totalRevenue= data['totalRevenue']

        # 获取最小值
        dict_temp = {'appid': str(appid_list[i]),'name': name, 'tags': tags, 'copiesSold': copiesSold, 'totalRevenue': totalRevenue}
        list_temp.append(dict_temp)
        with open('GameList_2_new_gama.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list_temp[0].keys())
            writer.writeheader()
            writer.writerows(list_temp)
        print("Request succeed")
    else:
        # Print an error message if the request failed
        print(f"Request failed with status code {response.status_code}")
    time.sleep(1)

print(list_temp)

