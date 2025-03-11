import matplotlib.pyplot as plt
import pandas as pd

# 数据
data = {
    'DateTime': [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Games': [70, 135, 242, 379, 274, 280, 302, 436, 1714, 2823, 4658, 6923, 5936, 2793, 3114, 3336, 3531, 3924, 4143],
    'Limited Games': [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 3, 13, 2945, 5298, 6564, 7957, 8831, 10349, 14784]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 绘制堆叠柱状图，设置dpi为300，并调整figsize以保持比例
plt.figure(figsize=(12, 6))  # 调整figsize为(10, 5)，确保宽高比例合适
plt.bar(df['DateTime'], df['Games'], label='Games')
plt.bar(df['DateTime'], df['Limited Games'], bottom=df['Games'], label='Limited Games')

# 添加竖虚线并标注
plt.axvline(x=2012.5, color='g', linestyle='--')  # Steam Greenlight 使用绿色
plt.text(2012.5 + 0.2, max(df['Games'] + df['Limited Games']) * 0.7, 'Steam Greenlight', color='g', rotation=90, verticalalignment='center')

plt.axvline(x=2017.5, color='r', linestyle='--')  # Steam Direct 使用红色
plt.text(2017.5 + 0.2, max(df['Games'] + df['Limited Games']) * 0.7, 'Steam Direct', color='r', rotation=90, verticalalignment='center')

# 设置横坐标为整数
plt.xticks(df['DateTime'])

# 添加标签和标题
plt.xlabel('Year')
plt.ylabel('Number of Games')

plt.legend()

# 保存图像为PNG文件
plt.savefig('SteamGameReleasebyYear.png', dpi=1200, bbox_inches='tight')  # 保存为PNG文件，DPI为300

# 显示图表
plt.show()