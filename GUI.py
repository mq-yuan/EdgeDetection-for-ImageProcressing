import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.Dog import DoG
import os
import cv2
from utils.FDoG import run
import numpy as np
from utils.Canny import NMS

plt.rcParams['font.sans-serif'] = ['SimHei']


def select_file():
    # 单个文件选择
    selected_file_path = filedialog.askopenfilename()  # 使用askopenfilename函数选择单个文件
    select_path.set(selected_file_path)


def EdgeDetection(path, canvas, sigma_c, sigma_m, rho, sobel_size, etf_size):
    input_img = path
    img = cv2.imread(input_img)
    if img.shape[:2] != (1024, 1024):
        img = cv2.resize(img, (1024, 1024))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Prewitt
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # Sobel
    x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # Canny
    gaussian = cv2.GaussianBlur(grayImage, (3, 3), 0)
    Canny = cv2.Canny(gaussian, 50, 150)

    # Dog
    differenced = DoG(grayImage, 25, 4, 1.1, None, 0.1)

    # FDoG
    edge = run(
        img=grayImage, sobel_size=sobel_size,
        etf_iter=6, etf_size=etf_size,
        fdog_iter=5, sigma_c=sigma_c, rho=rho, sigma_m=sigma_m,
        tau=0.907
    )

    # NMS + FDoG
    NMS_ = NMS(edge)

    # 结果列表存储
    titles = ['Orignal', 'Prewitt', 'Sobel', 'Canny', 'FDoG', 'NMS+FDoG']
    images = [rgb_img, Prewitt, Sobel, Canny, edge, NMS_]

    # 为便于比较进行反色处理
    for i in [1, 2, 3]:
        images[i] = cv2.bitwise_not(images[i])

    # 展示图片&保存图片
    filename = os.path.splitext(os.path.split(path)[1])[0]
    dirsname = os.path.join('./ans/', filename)
    if not os.path.exists(dirsname):
        os.makedirs(dirsname)

    for i in range(len(images)):
        a = f.add_subplot(2, 3, i+1)
        a.clear()
        a.imshow(images[i], 'gray')
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(titles[i])
        if titles[i] == 'Orignal':
            cv2.imwrite(os.path.join(dirsname, f'{titles[i]}.jpg'), cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(os.path.join(dirsname, f'{titles[i]}.jpg'), images[i])
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
    f.savefig(os.path.join(dirsname, f'ALL.jpg'), dpi = 600)
    print(f'ans saved to {dirsname}')


if __name__ == "__main__":
    root = tk.Tk()
    select_path = tk.StringVar()
    sigma_c = tk.DoubleVar()
    sigma_c.set(1.0)
    etf_size = tk.IntVar()
    etf_size.set(7)
    sobel_size = tk.IntVar()
    sobel_size.set(5)
    rho = tk.DoubleVar()
    rho.set(0.997)
    sigma_m = tk.DoubleVar()
    sigma_m.set(3.0)
    f = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasTkAgg(f, master=root)

    tk.Label(root, text="文件路径：").pack(side='top', anchor='nw')
    tk.Entry(root, textvariable=select_path).pack(side='top', anchor='n', fill='x')

    tk.Button(root, text="选择图片", command=select_file).pack(side='top', anchor='ne')

    tk.Label(root, text="sigma_c(another's standard variance will be set to1.6 * sigma_c)").pack(side='top', anchor='nw')
    tk.Entry(root, textvariable=sigma_c).pack(side='top', anchor='n', fill='x')

    tk.Label(root, text="sigma_m").pack(side='top', anchor='nw')
    tk.Entry(root, textvariable=sigma_m).pack(side='top', anchor='n', fill='x')

    tk.Label(root, text="rho").pack(side='top', anchor='nw')
    tk.Entry(root, textvariable=rho).pack(side='top', anchor='n', fill='x')

    tk.Label(root, text="etf_size").pack(side='top', anchor='nw')
    tk.Entry(root, textvariable=etf_size).pack(side='top', anchor='n', fill='x')

    tk.Label(root, text="sobel_size").pack(side='top', anchor='nw')
    tk.Entry(root, textvariable=sobel_size).pack(side='top', anchor='n', fill='x')

    tk.Button(root, text="边缘提取", command=lambda: EdgeDetection(select_path.get(), canvas, sigma_c.get(), sigma_m.get(),
                                                               rho.get(), sobel_size.get(), etf_size.get())).pack()
    root.mainloop()
