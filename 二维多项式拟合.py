import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 设置中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 假设你有一组二维数据点
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([5, 3.9, 2, 8.5, 18.2, 10, 13, 18, 30, 60])

# 目标输出 v
v = np.array([15, 12, 8, 22, 36, 24, 31, 40, 60, 90])

# 自定义拟合函数，假设我们使用二次多项式进行拟合
def model_func(X, a, b, c, d, e, f):
    x, y = X
    return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

# 拟合数据
params, covariance = curve_fit(model_func, (x, y), v)

# 输出拟合参数
print("拟合参数：", params)

# 使用拟合函数计算每个数据点的拟合值
v_fit = model_func((x, y), *params)

# 计算误差（原始值 - 拟合值）
residuals = v - v_fit

# 生成网格数据用于展示拟合结果
x_range = np.linspace(min(x), max(x), 100)
y_range = np.linspace(min(y), max(y), 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# 计算网格上每个点的拟合值
V_grid = model_func((X_grid, Y_grid), *params)

# 绘制灰度图
plt.figure(figsize=(8, 6))
plt.contourf(X_grid, Y_grid, V_grid, 50, cmap='gray')  # 50个灰度级
plt.colorbar(label="V (拟合值)")

# 绘制原始数据点，并显示误差条
plt.scatter(x, y, color='red', label='原始数据')
plt.errorbar(x, y, yerr=np.abs(residuals), fmt='o', color='red', ecolor='blue', capsize=5, label='误差')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('拟合结果的灰度图及误差')
plt.legend()
plt.show()
