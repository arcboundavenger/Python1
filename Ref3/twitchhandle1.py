import pandas as pd

# 尝试使用不同的编码格式读取CSV文件
df = pd.read_csv("Twitch_game_data.csv", encoding='ISO-8859-1')

# 转换数据类型（如果需要）
df['Peak_viewers'] = df['Peak_viewers'].astype(int)
df['Peak_channels'] = df['Peak_channels'].astype(int)

# 找到每个游戏的最高峰观众数和峰频道数
max_viewers = df.loc[df.groupby('Game')['Peak_viewers'].idxmax()]
max_channels = df.loc[df.groupby('Game')['Peak_channels'].idxmax()]

# 合并这两个DataFrame，去除重复的游戏
result = pd.concat([max_viewers, max_channels]).drop_duplicates(subset=['Game'])

# 保存结果到CSV文件
result.to_csv("top_games_peak_viewers_channels.csv", index=False)

# 输出结果
print("结果已保存到 top_games_peak_viewers_channels.csv")