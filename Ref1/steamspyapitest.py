import requests
import pandas as pd
import time
import random


def get_median_playtime(app_ids):
    median_playtimes = {}

    for app_id in app_ids:
        # 构造 API 请求的 URL
        url = f"https://steamspy.com/api.php?request=appdetails&appid={app_id}"

        try:
            response = requests.get(url)
            response.raise_for_status()  # 检查请求是否成功
            data = response.json()

            # 提取中位数游戏时间
            if 'median_forever' in data:
                median_playtimes[app_id] = data['median_forever'] / 60  # 转换为小时
            else:
                median_playtimes[app_id] = None  # 如果没有找到中位数

            # 创建 DataFrame
            df = pd.DataFrame(list(median_playtimes.items()), columns=['App ID', 'Median Playtime (Hours)'])

            # 保存为 Excel 文件
            output_file = 'median_playtime.xlsx'
            df.to_excel(output_file, index=False)
            print(f"Current results saved to {output_file}")

            # 等待随机 1 到 2 秒
            time.sleep(random.uniform(1, 2))

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for app ID {app_id}: {e}")
            median_playtimes[app_id] = None

    return median_playtimes


# 示例 appid 列表
app_ids = [
    2140330, 1190970, 1962660, 1942660, 934700, 1324130, 2440510,
    1594320, 1288320, 1065310, 1069660, 1608070, 2375550, 1875830,
    1812450, 2291060, 601050, 1129540, 1547000, 1335790, 768200,
    694280, 1849250, 1498570, 1271700, 482400, 1468720, 1295920,
    2058190, 1003590, 973230, 1737100, 1372110, 2446550, 2058180,
    1176470, 2495450, 1840080, 1272320, 1811990, 1593030, 1509510,
    1970580, 1222370, 1635450, 1624540, 1009560, 898750, 1229240,
    2162800, 1957780, 1680880, 1414850, 2140020, 2849080, 824550,
    1874490, 1342330, 1281590, 1432050, 820520, 1843760, 1177980,
    2060130, 1382070, 1963370, 491540, 1321440, 1157390, 2144740,
    1426450, 1000360, 1408610, 2198150, 1585220, 1677770, 1477590,
    1732180, 1239020, 1434950, 1114150, 2638370, 1097200, 1562420,
    1205520, 2096600, 1110910, 1944430, 1343240, 973810, 915810,
    1606180, 1088710, 2132850, 2015270, 1594940, 2653790, 1732190,
    1109570, 1701520, 1249970, 900040, 1150760, 1293160, 2190290,
    2103140, 1578650, 2786680, 674140, 1618540, 1497440, 1372810,
    454120, 1138660, 1388590, 1179080, 1220010, 2458530, 1272160,
    1801110, 2429860, 1043810, 1173770, 1359980, 1330470, 2187290,
    1972440, 1328840, 927350, 1504500, 1328350, 1206610, 1473350,
    2666510, 2095300, 2194530, 2068280, 1458100, 1745510, 1443200,
    2067920, 2113850, 1307580, 1718870, 1866180, 1684930, 2276930,
    1316680, 1846170, 2291850, 1161170, 1461810, 1250760, 978460,
    1615290, 1816570, 1634150, 1764390, 1506980, 2901520, 2378620,
    615530, 2707940, 1557990, 2878980, 2515020
] # 这里可以替换为你自己的 appid 列表
median_playtimes = get_median_playtime(app_ids)
