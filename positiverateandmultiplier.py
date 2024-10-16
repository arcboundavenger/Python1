import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.api as sm

# Extract the data from 'Sheet1'
x_data = [40, 60, 70, 80, 90]
y_data = [2.4, 3.6, 7.2, 10.1, 21.0]

# Since we are fitting an exponential model, we'll have to transform our data
# by taking the logarithm of the 'Multiplier' to make it a linear model.
log_y_data = np.log(y_data)

# Add a constant term for the intercept
X = sm.add_constant(x_data)

# Construct the ordinary least squares (OLS) model
model = sm.OLS(log_y_data, X)

# Fit the model to the training set
results = model.fit()

# Get the confidence intervals for the parameters
conf_int = results.conf_int(alpha=0.05)  # 95% confidence interval

# 在图像上绘制
x_fit = np.linspace(0, 100)
y_fit = np.exp(results.params[0]) * np.exp(results.params[1] * x_fit)

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Original Data', color='blue')
plt.plot(x_fit, y_fit, label='Fitted Curve: y = a * x + e ^ b', linestyle='--', color='g')
plt.xlabel('Positive Rate')
plt.ylabel('Multiplier')
# 调整坐标轴刻度
plt.xlim(0, 100)
plt.ylim(0, 30)

plt.legend(loc='lower right', fontsize=8)
plt.grid(True)

# Print the fitted exponential function expression and confidence intervals
a_lower = np.exp(conf_int[0, 0])
a_upper = np.exp(conf_int[0, 1])
b_lower = conf_int[1, 0]
b_upper = conf_int[1, 1]
a = np.exp(results.params[0])
b = results.params[1]


print("\nFitted Exponential Function Expression and Confidence Intervals:")
print(f"Fitted Expression: y = {np.exp(results.params[0])} * e^({results.params[1]} * x)")
# Print the R-squared value
print("\nR-squared Value:", results.rsquared)
print(f"Confidence Interval for 'a': {a_lower} to {a_upper}")
print(f"Confidence Interval for 'b': {b_lower} to {b_upper}")

# 添加表格
table_data = [['Params', 'Value', '95% CI', 'R2'],
              ['a', f'{a:.4f}', f'({a_lower:.4f}, {a_upper:.4f})', f'{results.rsquared:.4f}'],
              ['b', f'{b:.4f}', f'({b_lower:.4f}, {b_upper:.4f})', '']]

table = plt.table(cellText=table_data,
                  colLabels=None,
                  loc='upper left',
                  cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(0.7, 1)


# Display the plot
plt.show()
