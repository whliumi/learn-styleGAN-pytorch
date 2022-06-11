# 深度生成模型大项目——styleGAN的复现

[TOC]

## 项目介绍

​	本项目首先是将styleGAN的原本tensorflow代码转成pytorch，然后在此基础上做一些实验，比如styleGANv1版本比原来的proGAN的效果提升，以及如果不加噪声，不加像素归一化或者不加STYLE或ADAIN的影响。

## 代码结构

1. 文件夹FFHQ

   将原本FFHQ数据集解压之后的图片放入FFHQ文件夹，其中图像是3×1024×1024，所以很占用空间（数据太大，发给您的是空文件夹）。

2. 文件夹FFHQ_data

   将FFHQ中的图像按4×4，8×8，...,1024×1024储存，在计算G_synthesis的loss时会用到（数据太大，发给您的是空文件夹）。

3. 文件夹train_checkpoint
   训练G_mapping，G_synthesis，D_basic之后的checkpoint。

4. 文件夹training

   networks_stylegan.py主要是G_mapping，G_synthesis，D_basic的代码部分。其中是否加噪声，是否加像素归一化或者是否加STYLE或ADAIN都在G_synthesis类的forward函数的参数里。

5. dataset.py

   主要用于生成4×4，8×8，...,1024×1024像素的图片，并将其储存在文件夹FFHQ_data。

6. training.py

   train函数用于训练，generate_image函数用于生成图像。
