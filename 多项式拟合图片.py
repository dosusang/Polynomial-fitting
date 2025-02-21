import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import BytesIO

class PolynomialFitterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像通道多项式拟合器")
        self.create_widgets()
        self.img = None
        self.photo_image = None

    def create_widgets(self):
        # 控制面板
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        # 图片选择按钮
        self.btn_select = tk.Button(control_frame, text="选择图片", command=self.load_image)
        self.btn_select.pack(side=tk.LEFT, padx=5)

        # 多项式阶数输入
        self.degree_label = tk.Label(control_frame, text="多项式阶数:")
        self.degree_label.pack(side=tk.LEFT, padx=5)
        self.degree_entry = tk.Entry(control_frame, width=5)
        self.degree_entry.pack(side=tk.LEFT, padx=5)
        self.degree_entry.insert(0, "3")

        # 拟合按钮
        self.btn_fit = tk.Button(control_frame, text="开始拟合", command=self.process_image)
        self.btn_fit.pack(side=tk.LEFT, padx=5)

        # 图片显示区域
        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=10)

        self.original_label = tk.Label(image_frame, text="原始图像")
        self.original_label.pack(side=tk.LEFT, padx=20)
        self.original_panel = tk.Label(image_frame)
        self.original_panel.pack(side=tk.LEFT, padx=20)

        self.fitted_label = tk.Label(image_frame, text="拟合结果")
        self.fitted_label.pack(side=tk.LEFT, padx=20)
        self.fitted_panel = tk.Label(image_frame)
        self.fitted_panel.pack(side=tk.LEFT, padx=20)

        # 多项式表达式显示
        self.text_box = tk.Text(self.root, width=80, height=15)
        self.text_box.pack(pady=10, padx=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            try:
                self.img = Image.open(file_path).convert("RGB")
                self.show_image(self.img, self.original_panel)
                self.text_box.delete(1.0, tk.END)
            except Exception as e:
                messagebox.showerror("错误", f"无法加载图像: {str(e)}")

    def show_image(self, img, panel):
        img.thumbnail((300, 300))
        self.photo_image = ImageTk.PhotoImage(img)
        panel.config(image=self.photo_image)
        panel.image = self.photo_image

    def process_image(self):
        if self.img is None:
            messagebox.showwarning("警告", "请先选择图像！")
            return

        try:
            degree = int(self.degree_entry.get())
            if degree < 1 or degree > 6:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "请输入1-6之间的整数阶数")
            return

        # 转换图像数据
        img_array = np.array(self.img) / 255.0
        height, width, _ = img_array.shape

        # 创建坐标网格
        u = np.linspace(0, 1, width)
        v = np.linspace(0, 1, height)
        U, V = np.meshgrid(u, v)
        coordinates = np.column_stack((U.ravel(), V.ravel()))

        # 初始化结果
        fitted = np.zeros_like(img_array)
        expressions = []

        # 对每个通道进行拟合
        for ch in range(3):
            # 提取通道数据
            channel_data = img_array[:, :, ch].ravel()

            # 创建多项式特征
            poly = PolynomialFeatures(degree=degree, include_bias=True)
            X_poly = poly.fit_transform(coordinates)

            # 训练模型
            model = LinearRegression(fit_intercept=False)
            model.fit(X_poly, channel_data)

            # 生成预测结果
            predicted = model.predict(X_poly).reshape(height, width)
            fitted[:, :, ch] = np.clip(predicted, 0, 1)

            # 生成多项式表达式
            expr = self.generate_expression(poly, model, ['u', 'v'])
            expressions.append(f"通道 {['红', '绿', '蓝'][ch]}:\n{expr}\n")

        # 显示拟合结果
        fitted_img = Image.fromarray((fitted * 255).astype(np.uint8))
        self.show_image(fitted_img, self.fitted_panel)

        # 显示多项式表达式
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, "\n".join(expressions))

    def generate_expression(self, poly, model, variables):
        terms = []
        for i, name in enumerate(poly.get_feature_names_out(input_features=variables)):
            coef = model.coef_[i]
            if abs(coef) > 1e-5:  # 过滤小系数项
                term = f"{coef:.4f}*{name}"
                terms.append(term)
        return " + ".join(terms) if terms else "0"

if __name__ == "__main__":
    root = tk.Tk()
    app = PolynomialFitterApp(root)
    root.mainloop()
