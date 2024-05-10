import requests
import json
import csv
import pandas as pd

# Define the URL of the API
url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"

# Define the query parameters
params = {
    'format':'json',
}

# Make a GET request to the URL with the parameters
response = requests.get(url, params=params)
if response.status_code == 200:
    data=response.json()
    json_data=json.dumps(data)
    with open("data.json", "w") as f:
        # Write the JSON string to the txt file
        f.write(json_data)
    # 将JSON数据转换为DataFrame对象
    df = pd.DataFrame(data['applist']['apps'])
    # 将DataFrame对象保存为CSV文件
    df.to_csv('steam_app_list.csv', index=False)

else:
    # Print an error message if the request failed
    print(f"Request failed with status code {response.status_code}")

