# Import the requests library
import requests
import json
import time
import csv
import pandas as pd

df = pd.read_csv('GameList 2.csv')
appid_list = df['appid']
list_temp = []
for i in range(len(appid_list)):

    # Define the URL of the API
    url = "https://store.steampowered.com/api/appdetails"
    url2 = 'https://steamspy.com/api.php?request=appdetails'

    # Define the query parameters
    params = {
        'format': 'json',
        'appids': str(appid_list[i]),
        'appid': str(appid_list[i])
    }

    # Find the value you want to extract
    def find_value(data, target_key):
        if isinstance(data, dict):
            for key, value in data.items():
                if key == target_key:
                    return value
                else:
                    result = find_value(value, target_key)
                    if result is not None:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = find_value(item, target_key)
                if result is not None:
                    return result


    # Make a GET request to the URL with the parameters
    response = requests.get(url, headers={}, params=params)
    if response.status_code == 200:
        # Parse the response content as a JSON object
        data = response.json()
        json_data = json.dumps(data, indent=4, sort_keys=True)
        with open("SteamData.json", "w") as f:
            # Write the JSON string to the txt file
            f.write(json_data)
        with open('SteamData.json') as f:
            data = json.load(f)
    else:
        # Print an error message if the request failed
        print(f"Request failed with status code {response.status_code}")

    # Extract the value you want
    target_key = 'categories'  # Replace with the key you want to extract
    value0 = find_value(data, target_key)
    dict0 = {target_key: value0}
    value = ''
    category = None
    for category in dict0['categories']:
        if category['id'] == 2:
            value = 'Single-player'
        if category['id'] == 9:
            value = 'Co-op'
        if category['id'] == 38:
            value = 'Online Co-op'
        if category['id'] == 49:
            value = 'PvP'
        if category['id'] == 36:
            value = 'Online PvP'
        if category['id'] == 20:
            value = 'MMO'

    # Extract the value you want
    target_key2 = 'score'  # Replace with the key you want to extract
    value2 = find_value(data, target_key2)

    time.sleep(1)

    # Make a GET request to the URL with the parameters
    response = requests.get(url2, headers={}, params=params)
    if response.status_code == 200:
        # Parse the response content as a JSON object
        data = response.json()
        json_data = json.dumps(data, indent=4, sort_keys=True)
        with open("SteamspyData.json", "w") as f:
            # Write the JSON string to the txt file
            f.write(json_data)
        with open('SteamspyData.json') as f:
            data = json.load(f)
    else:
        # Print an error message if the request failed
        print(f"Request failed with status code {response.status_code}")

    target_key3 = 'initialprice'  # Replace with the key you want to extract
    value3 = find_value(data, target_key3)

    target_key4 = 'languages'  # Replace with the key you want to extract
    value4 = find_value(data, target_key4)
    dict_temp = {'appid': str(appid_list[i]), target_key: value, target_key2: value2, target_key3: value3,
                 target_key4: value4}
    list_temp.append(dict_temp)
    time.sleep(1)

print(list_temp)

with open('GameList_2_full.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list_temp[0].keys())
    writer.writeheader()
    writer.writerows(list_temp)
