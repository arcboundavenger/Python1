import matplotlib.pyplot as plt
import pandas as pd

# 从 Excel 文件中读取数据
file_path = 'IHLIDXUSTPSOFTDEVE.xlsx'  # 文件路径
df = pd.read_excel(file_path)  # 读取 Excel 文件

# 确保日期列是日期格式
df['observation_date'] = pd.to_datetime(df['observation_date'])

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(df['observation_date'], df['IHLIDXUSTPSOFTDEVE'], color='blue', linestyle='-')

# 设置纵坐标轴标签
plt.ylabel('Index Feb, 1 2020=100')

# 保存图像，dpi=1200
output_file = 'Software Development Job Postings on Indeed in the United States.png'
plt.savefig(output_file, dpi=1200, bbox_inches='tight')

# 显示图表
plt.show()