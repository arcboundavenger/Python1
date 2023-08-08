import requests
import json
import time
import csv
import pandas as pd

df = pd.read_csv('GameList 2.csv')
appid_list = df['appid']
list_temp = []
for i in range(len(appid_list)):
    url = "https://api.gamalytic.com/game/" + str(appid_list[i])
    params = {
        'format': 'json'
    }
    response = requests.get(url, headers={}, params=params)
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
        prices = [record['price'] for record in data['history']]
        players = [record['players'] for record in data['history']]
        # 获取最小值
        min_price = min(prices)
        max_players = int(max(players))
        dict_temp = {'appid': str(appid_list[i]), 'price_min': min_price, 'max_players': max_players}
        list_temp.append(dict_temp)
    else:
        # Print an error message if the request failed
        print(f"Request failed with status code {response.status_code}")
    time.sleep(1)

print(list_temp)

with open('GameList_2_gama.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list_temp[0].keys())
    writer.writeheader()
    writer.writerows(list_temp)