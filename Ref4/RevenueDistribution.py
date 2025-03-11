import matplotlib.pyplot as plt
import pandas as pd

# 读取 Excel 文件
file_path = 'VGI - Average Sales by Percentile.xlsx'
df = pd.read_excel(file_path)

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(df['Percentile'], df['Average Sales ($)'], color='blue', linestyle='-')

# 设置纵坐标为对数刻度
plt.yscale('log')

# 设置坐标轴标签
plt.xlabel('Percentile')
plt.ylabel('Gross Revenue (USD)')
# 保存图像，dpi=1200
output_file = 'average_sales_by_percentile_log_scale.png'
plt.savefig(output_file, dpi=1200, bbox_inches='tight')
# 显示图表
plt.show()