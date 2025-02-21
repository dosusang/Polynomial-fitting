import numpy as np
from math import sqrt, pi
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# 球谐基函数生成器
def real_spherical_harmonics(l, m, theta, phi):
    """
    计算实数形式的球谐函数
    l: 阶数 (0,1,2,...)
    m: 阶内编号 (-l <= m <= l)
    theta: 极角 [0, pi]
    phi: 方位角 [0, 2pi]
    """
    from scipy.special import sph_harm
    complex_Y = sph_harm(abs(m), l, phi, theta)
    if m < 0:
        Y = sqrt(2) * (-1)**m * complex_Y.imag
    elif m > 0:
        Y = sqrt(2) * (-1)**m * complex_Y.real
    else:
        Y = complex_Y.real
    return Y.real

class SHFitterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("球谐函数图像拟合器")
        self.max_degree = 5  # 最大允许阶数
        self.create_widgets()
        self.img = None

    def create_widgets(self):
        # 控制面板
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        # 图片选择按钮
        self.btn_load = tk.Button(control_frame, text="加载图像", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        # 阶数选择
        self.degree_label = tk.Label(control_frame, text="球谐阶数 (0-5):")
        self.degree_label.pack(side=tk.LEFT, padx=5)
        self.degree_spin = tk.Spinbox(control_frame, from_=0, to=5, width=3)
        self.degree_spin.pack(side=tk.LEFT, padx=5)
        self.degree_spin.delete(0, "end")
        self.degree_spin.insert(0, "3")

        # 拟合按钮
        self.btn_fit = tk.Button(control_frame, text="执行拟合", command=self.process_image)
        self.btn_fit.pack(side=tk.LEFT, padx=5)

        # 图像显示区域
        img_frame = tk.Frame(self.root)
        img_frame.pack(pady=10)
        
        self.original_label = tk.Label(img_frame, text="原始图像")
        self.original_label.pack(side=tk.LEFT, padx=20)
        self.original_panel = tk.Label(img_frame)
        self.original_panel.pack(side=tk.LEFT, padx=20)
        
        self.fitted_label = tk.Label(img_frame, text="拟合结果")
        self.fitted_label.pack(side=tk.LEFT, padx=20)
        self.fitted_panel = tk.Label(img_frame)
        self.fitted_panel.pack(side=tk.LEFT, padx=20)

        # 结果显示
        self.result_text = tk.Text(self.root, width=80, height=15)
        self.result_text.pack(pady=10, padx=20)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if path:
            self.img = Image.open(path).convert("RGB")
            self.show_image(self.img, self.original_panel)
            self.result_text.delete(1.0, tk.END)

    def show_image(self, img, panel):
        img.thumbnail((300, 300))
        tk_img = ImageTk.PhotoImage(img)
        panel.config(image=tk_img)
        panel.image = tk_img

    def process_image(self):
        if self.img is None:
            messagebox.showwarning("警告", "请先加载图像")
            return

        try:
            degree = int(self.degree_spin.get())
            if degree < 0 or degree > self.max_degree:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", f"请输入0-{self.max_degree}之间的整数")
            return

        # 准备数据
        img_array = np.array(self.img) / 255.0
        h, w, _ = img_array.shape
        
        # 生成球面坐标参数
        u = np.linspace(0, 1, w)
        v = np.linspace(0, 1, h)
        U, V = np.meshgrid(u, v)
        
        # 将UV映射到球面坐标
        theta = V * np.pi    # 极角 [0, π]
        phi = U * 2 * np.pi  # 方位角 [0, 2π]

        # 生成球谐基函数
        features = []
        for l in range(degree+1):
            for m in range(-l, l+1):
                basis = real_spherical_harmonics(l, m, theta, phi)
                features.append(basis.ravel())
        X = np.column_stack(features)

        # 执行拟合
        fitted = np.zeros_like(img_array)
        coeffs = []
        for ch in range(3):
            model = LinearRegression(fit_intercept=False)
            model.fit(X, img_array[..., ch].ravel())
            fitted[..., ch] = model.predict(X).reshape(h, w)
            coeffs.append(model.coef_)

        # 显示结果
        fitted_img = Image.fromarray((np.clip(fitted, 0, 1)*255).astype(np.uint8))
        self.show_image(fitted_img, self.fitted_panel)
        
        # 生成表达式
        self.show_coefficients(degree, coeffs)

    def show_coefficients(self, degree, coeffs):
        text = ""
        for ch_idx, channel in enumerate(["Red", "Green", "Blue"]):
            text += f"=== {channel} Channel ===\n"
            coeff_idx = 0
            for l in range(degree+1):
                for m in range(-l, l+1):
                    c = coeffs[ch_idx][coeff_idx]
                    if abs(c) > 1e-4:
                        text += f"Y_{l}^{m}: {c:.6f}\n"
                    coeff_idx += 1
            text += "\n"
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)

if __name__ == "__main__":
    root = tk.Tk()
    app = SHFitterApp(root)
    root.mainloop()
