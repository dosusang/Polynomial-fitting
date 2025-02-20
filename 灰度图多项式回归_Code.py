from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 设置中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 读取图片并转换为灰度图像
image = Image.open('TestImage.png')  # 替换为你的图片路径
gray_image = image.convert('L')  # 将图像转换为灰度图

# 转换为 numpy 数组
gray_image_np = np.array(gray_image)

# 获取图片的大小
height, width = gray_image_np.shape

# 生成图像的像素坐标
x, y = np.meshgrid(np.arange(width), np.arange(height))
x = x.ravel()
y = y.ravel()
z = gray_image_np.ravel()  # 灰度值作为输出

# 准备数据（坐标作为输入，灰度值作为目标）
X = np.vstack((x, y)).T  # 将 x 和 y 合并为特征
y = z  # 灰度值作为目标

# 将输入特征转换为多项式特征
poly = PolynomialFeatures(degree=10)  # 你可以调整多项式的阶数
X_poly = poly.fit_transform(X)

# 使用线性回归模型来拟合多项式特征
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# 用训练好的模型进行预测
z_fit_poly = poly_model.predict(X_poly)

# 将拟合结果恢复成图像的形状
z_fit_image_poly = z_fit_poly.reshape(height, width)

# 输出多项式回归系数
print("多项式回归系数：", poly_model.coef_)

# 绘制原始图像和拟合后的图像
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 原始图像
ax[0].imshow(gray_image_np, cmap='gray')
ax[0].set_title('原始图像')
ax[0].axis('off')  # 关闭坐标轴

# 多项式拟合后的图像
ax[1].imshow(z_fit_image_poly, cmap='gray')
ax[1].set_title('多项式拟合后的图像')
ax[1].axis('off')  # 关闭坐标轴

plt.show()
