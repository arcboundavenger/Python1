from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta
import pandas as pd
from google_auth_oauthlib.flow import InstalledAppFlow
import time
import csv

# 加载客户端凭据
flow = InstalledAppFlow.from_client_secrets_file('client_secret_438963477050-h5o5n7ho4nifiglijmini53ee1rs4ecu.apps.googleusercontent.com.json', scopes=['https://www.googleapis.com/auth/youtube.force-ssl'])
credentials = flow.run_local_server(port=0)

# 构建YouTube Data API客户端
youtube = build('youtube', 'v3', credentials=credentials)

# 读取Excel表
df = pd.read_excel('NewGames2.xlsx')

# 定义要查询的日期范围
start_date = 'YYYY-MM-DD'
end_date = 'YYYY-MM-DD'

# 初始化视频总数
total_video_count = 0
list_temp = []

# 遍历Excel表中的每一行
for index, row in df.iterrows():
    game_name = row['Game']
    start_date = row['Start'].strftime('%Y-%m-%d')
    end_date = row['End'].strftime('%Y-%m-%d')

    # 构建查询参数
    search_params = {
        'q': game_name,
        'publishedAfter': str(start_date) + 'T00:00:00Z',
        'publishedBefore': str(end_date) + 'T00:00:00Z',
        'type': 'video',
        'videoCategoryId': 20,
        'part': 'snippet'
    }

    # 发起搜索请求
    search_response = youtube.search().list(**search_params).execute()

    # 获取搜索结果中的视频数量
    video_count = search_response['pageInfo']['totalResults']
    time.sleep(1)
    dict_temp = {'game': game_name, 'video_count': video_count}
    list_temp.append(dict_temp)

print(list_temp)

with open('GameList_2_youtube.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list_temp[0].keys())
    writer.writeheader()
    writer.writerows(list_temp)