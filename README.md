![](https://typora-ilgzh.oss-cn-beijing.aliyuncs.com/202303241427863.jpg)

# 边缘检测

## 介绍
本次实验使用python实现了Prewitt、Sobel、Canny、FDoG和使用非极大值约束得FDoG这五种边缘检测方式，并制作了GUI便于使用。为使效果对比更加明显，对Prewitt、Sobel、Canny的输出结果进行反色处理。

## 环境依赖
- `pip install -r requirements.txt`.

## 实验测试
在windows系统且确保已安装python3的情况下，点击`run_me.bat`文件可直接运行。

或者可以使用 `python ./GUI.py` .命令

## 文件架构

文件中已附带`Image`文件夹如下：

```py
Image
├─RGB # 存放图像处理常用RGB类型测试图
|  ├─4.1.01.tiff
|  ├─4.1.02.tiff
|  ......
|  ├─LenaRGB.bmp
|  └PeppersRGB.bmp
├─Personal # 个人图像
|    ├─1_1.jpg
|    ├─1_2.png
|    ├─1_3.jpg
|    └1_4.jpg
├─Gray # 存放图像处理常用灰度测试图
|  ├─5.1.09.tiff
|  ├─5.1.10.tiff
|  ......
|  ├─Lena.bmp
|  └Peppers.bmp
```

点击“选择图片”选择测试图片后点击“边缘提取”进行边缘提取，程序将自动上述提到的五种方法和原图展示到屏幕上，并保存至`./ans/xxx`文件夹中，关闭窗口程序结束。

## Citations
```
@inproceedings{Kang2007CoherentLD,
  title={Coherent line drawing},
  author={Henry Kang and Seungyong Lee and Charles K. Chui},
  booktitle={International Symposium on Non-Photorealistic Animation and Rendering},
  year={2007}
}
```