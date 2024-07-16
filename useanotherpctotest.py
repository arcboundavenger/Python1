import pandas as pd

# 定义类别列表
main_genre_list = ['MMORPG', 'Racing', 'Sports', 'Shooter', 'RPG', 'Strategy', 'Simulation', 'Adventure', 'Casual', 'Action']
sub_genre_list = ['Extraction Shooter', 'Social Deduction', 'Gambling', 'Escape Room', 'Asynchronous Multiplayer', 'Narrative', '4X', 'Open World Survival Craft', 'Colony Sim', 'Grand Strategy', 'Match 3', 'Music', 'Fighting', 'Idler', 'Beat \'em up', 'Flight', 'Metroidvania', 'City Builder', 'Driving', 'Board Game', 'Tower Defense', 'Detective', 'Turn-Based', 'RTS', 'Clicker', 'Life Sim', 'Card Game', 'Third-Person Shooter', 'Stealth', 'Dating Sim', 'Base Building', 'Walking Simulator', 'JRPG', 'Dungeon Crawler', 'Resource Management', 'Hack and Slash', 'Survival Horror', 'Turn-Based Strategy', 'Crafting', 'Logic', '3D Platformer', 'Roguelite', 'Tactical', 'Building', 'Roguelike', 'Side Scroller', 'Action RPG', 'Sandbox', 'Point & Click', 'FPS', 'VR', 'Linear', '2D Platformer', 'Sexual Content', 'Visual Novel', 'Survival', 'Open World', 'Horror', 'Platformer', 'Action-Adventure', 'Arcade', 'Puzzle']
art_style_list = ['Gothic', 'Comic Book', 'Abstract', 'Text-Based', '2.5D', 'Cartoon', 'Old School', 'Hand-drawn', 'Minimalist', 'Cartoony', 'Realistic', 'Pixel Graphics', 'Colorful', '3D', '2D']
game_mode_list = ['Massively Multiplayer', 'Local Co-Op', 'Local Multiplayer', 'Online Co-Op', 'PvE', 'Co-op', 'PvP', 'Multiplayer', 'Singleplayer']
point_of_view_list = ['Isometric', 'Top-Down', 'Third Person', 'First-Person']


# 读取原始xlsx文件
df = pd.read_excel('steam_games_continued_202407_temp.xlsx')

# 创建新的DataFrame来存储结果
new_df = pd.DataFrame(columns=['appid', 'Main Genre', 'Sub-Genre', 'Niche Genre', 'Art Style', 'Theme', 'Game Mode', 'POV'])

# 遍历每一行
for index, row in df.iterrows():
    tags = row['Tags']
    categories = {}

    # 检查tags是否为空值
    if isinstance(tags, str):
        # 按逗号分割tags内容，并转换为列表
        tag_list = tags.split(', ')

        # 搜索标签所属的类别
        categories['Main Genre'] = next((tag for tag in tag_list if tag in main_genre_list), None)
        categories['Sub-Genre'] = next((tag for tag in tag_list if tag in sub_genre_list), None)
        # categories['Niche Genre'] = next((tag for tag in tag_list if tag in niche_genre_list), None)
        categories['Art Style'] = next((tag for tag in tag_list if tag in art_style_list), None)
        # categories['Theme'] = next((tag for tag in tag_list if tag in main_theme_list), None)
        categories['Game Mode'] = next((tag for tag in tag_list if tag in game_mode_list), None)
        categories['POV'] = next((tag for tag in tag_list if tag in point_of_view_list), None)

    # 将结果添加到新的DataFrame中
    if any(categories.values()):
        categories['appid'] = row['appid']
        new_df = pd.concat([new_df, pd.DataFrame(categories, index=[0])], ignore_index=True)

# 将结果保存到新的xlsx文件
new_df.to_excel('output_steam_games_continued_202407_temp.xlsx', index=False)
print('Done')