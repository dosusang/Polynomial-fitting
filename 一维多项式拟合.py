import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 假设你有一组二维数据点
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 3.9, 2, 8.5, 18.2])

# 选择多项式的阶数，这里我们选择拟合一个二次多项式 (degree = 2)
degree = 2
coeffs = np.polyfit(x, y, degree)

# 打印拟合的多项式系数
print("多项式系数：", coeffs)

# 使用拟合得到的多项式进行预测
p = np.poly1d(coeffs)
y_fit = p(x)

# 绘图查看拟合效果
plt.scatter(x, y, label='原始数据')
plt.plot(x, y_fit, label='拟合曲线', color='red')
plt.legend()
plt.show()
