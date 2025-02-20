import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox
import pyperclip  # 用于复制到剪贴板

# 设置中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 创建Tkinter窗口
window = tk.Tk()
window.title("交互式数据点生成与拟合")

# 设置数据容器
x_points = []
y_points = []

# 当前选择的多项式阶数
degree = 4

# 画图函数
def plot_points():
    # 清空之前的图形
    ax.clear()
    
    # 设置坐标轴范围
    try:
        x_max = float(entry_x_max.get())
        y_max = float(entry_y_max.get())
    except ValueError:
        messagebox.showerror("输入错误", "最大值输入无效，请输入数字。")
        return
    
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    # 绘制数据点
    ax.scatter(x_points, y_points, label="数据点", color='blue')
    ax.set_title('点击生成数据点')
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.legend()

    # 绘制画布
    canvas.draw()

# 拟合并绘制拟合曲线
def fit_and_plot():
    # 确保有足够的点进行拟合
    if len(x_points) < 2:
        messagebox.showerror("数据不足", "请至少点击两个数据点进行拟合。")
        return
    
    # 获取当前选择的多项式阶数
    global degree
    degree = int(scale_degree.get())  # 获取滑块的值
    
    try:
        coeffs = np.polyfit(x_points, y_points, degree)
        p = np.poly1d(coeffs)
        
        # 获取X轴最大值
        try:
            x_max = float(entry_x_max.get())
            y_max = float(entry_y_max.get())
        except ValueError:
            messagebox.showerror("输入错误", "最大值输入无效，请输入数字。")
            return
        
        # 为了使拟合曲线更平滑，生成更密集的x值，范围从0到x_max
        x_fine = np.linspace(0, x_max, 500)  # 生成500个点，确保从0到x_max
        y_fit = p(x_fine)
        
        # 清空之前的图形
        ax.clear()
        
        # 设置坐标轴范围
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        
        # 绘制数据点和拟合曲线
        ax.scatter(x_points, y_points, label="数据点", color='blue')
        ax.plot(x_fine, y_fit, label=f"拟合曲线 (阶数 {degree})", color='red')
        ax.set_title(f"多项式拟合 (阶数 {degree})")
        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.legend()
        
        # 绘制画布
        canvas.draw()
        
        # 格式化拟合的多项式表达式
        poly_expr = "y = " + " + ".join([f"{coeff:.4f}x^{degree-i}" for i, coeff in enumerate(coeffs)])
        
        # 将多项式表达式复制到剪贴板
        pyperclip.copy(poly_expr)
        
        # 显示拟合的多项式系数
        messagebox.showinfo("拟合结果", f"拟合的多项式系数: {coeffs}\n\n拟合表达式已复制到剪贴板：\n{poly_expr}")
    
    except Exception as e:
        messagebox.showerror("拟合错误", f"拟合过程中出现错误: {e}")

# 清除当前的数据点和拟合结果
def clear_data():
    global x_points, y_points
    x_points = []
    y_points = []
    plot_points()

# 删除上一个点
def undo_last_point(event=None):
    if x_points and y_points:
        x_points.pop()
        y_points.pop()
        plot_points()

# 鼠标点击事件处理函数
def on_click(event):
    # 获取鼠标点击的位置
    if event.inaxes:
        x = event.xdata
        y = event.ydata
        if x is not None and y is not None:
            # 将点击的点添加到数据列表中
            x_points.append(x)
            y_points.append(y)
            plot_points()

# 创建画布显示图形
frame_canvas = tk.Frame(window)
frame_canvas.grid(row=0, column=0, padx=10, pady=10, rowspan=4)

# 设置最大值输入框
frame_controls = tk.Frame(window)
frame_controls.grid(row=0, column=1, padx=10, pady=10, sticky="n")

label_x_max = tk.Label(frame_controls, text="X 最大值：")
label_x_max.grid(row=0, column=0, pady=5)
entry_x_max = tk.Entry(frame_controls)
entry_x_max.insert(0, "1")  # 默认最大值为10
entry_x_max.grid(row=0, column=1, pady=5)

label_y_max = tk.Label(frame_controls, text="Y 最大值：")
label_y_max.grid(row=1, column=0, pady=5)
entry_y_max = tk.Entry(frame_controls)
entry_y_max.insert(0, "1")  # 默认最大值为10
entry_y_max.grid(row=1, column=1, pady=5)

# 添加多项式阶数控制滑块
label_degree = tk.Label(frame_controls, text="选择多项式阶数：")
label_degree.grid(row=2, column=0, pady=5)

scale_degree = tk.Scale(frame_controls, from_=1, to=10, orient="horizontal")
scale_degree.set(degree)  # 默认阶数为4
scale_degree.grid(row=2, column=1, pady=5)

# 按钮区
button_frame = tk.Frame(window)
button_frame.grid(row=1, column=1, padx=10, pady=10)

button_fit = tk.Button(button_frame, text="拟合数据", command=fit_and_plot)
button_fit.grid(row=0, column=0, padx=10, pady=10)

button_clear = tk.Button(button_frame, text="清除数据", command=clear_data)
button_clear.grid(row=1, column=0, padx=10, pady=10)

# 创建matplotlib图形
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title('点击生成数据点')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')

# 绘制初始图形并将其嵌入到Tkinter窗口
canvas = FigureCanvasTkAgg(fig, master=frame_canvas)
canvas.draw()
canvas.get_tk_widget().pack()

# 绑定鼠标点击事件
canvas.mpl_connect('button_press_event', on_click)

# 监听Ctrl+Z组合键用于撤销
window.bind('<Control-z>', undo_last_point)

# 启动Tkinter主循环
window.mainloop()
