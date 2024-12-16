# Insect_Identification_System

## 项目介绍
该项目是一个基于SwinTransformer和flask框架的昆虫识别系统，可以实现对各种各样的昆虫的识别。SwinTransformerModel.py可以对项目根目录的dataset文件夹中已分类的图像数据集进行训练，会将训练好的模型权重在保存在model文件夹中。p redict.py文件可以预测指定路径中的所有昆虫图像，默认路径为./test_images。还使用flask框架搭建了一个网页端，可以在网页中对比查看昆虫图像和预测结果。

## 功能特点

- 能够严格规范数据集结构。`./dataset`中只能出现文件夹和图像文件，并且图像文件只能出现在叶子文件夹，非图像文件、空文件夹、非叶子节点图像文件均会被删除，保证数据集的结构规范，便于进行标签分类。
- 能够制定文件夹及文件进行预测。运行predict.py文件进行预测时，默认预测./test_images文件夹及其子文件夹中的所有图像文件，同时也可以制定其他路径的文件或文件夹进行预测，如果是单个文件则直接预测，如果是文件夹，则会预测该文件夹及其子文件夹中的所有图像文件。
- 模型保存自动编号。模型训练好之后，模型权重文件的名称能够按照{base_name}_{next_index}.pth的格式自动编号，其中 next_index 由程序自动获取。
- 具有web端。采用flask框架写了一个后端，还写了一个极简风格的前端，前端支持同时显示图片和对应预测结果，可以在前端选择上传文件提交给后端预测，然后后端能够返回预测结果给前端。

## 使用

### 1. 环境要求

- MacOS/Linux/Windows
- Torch, Transformer

### 2. 项目拉取

克隆本项目：

```sh
git clone https://github.com/ccomoe/Insects-Classification-System.git
cd Insects-Classification-System
```

