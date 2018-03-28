## 项目介绍

本项目通过 Kinect v2 和 G1080Ti 显卡实现实时的三维重建，目前支持一个Kinect。

很快将支持：

    * 多Kinect数据的融合

    * 3D Telepresence 功能

## 环境设置


1. 安装visual studio 2017，版本最好是v15.4。（v15.5与cuda9.1不兼容，v15.6不清楚)

2. 安装Cuda9.1，注意此步骤需在安装vs以后进行，这样vs中就会被装上cuda的插件。

3. 确保你的visual studio可以将一个普通工程改成CUDA工程，这个流程可以网上搜索一下。

4. 安装point cloud library (pcl) 1.8.1 with GPU，需要手动编译，会遇到很多的坑，建议按照网上精细教程进行。

5. 通过cmake生成本项目的vs工程，编译器选择visual studio 2017 v15 Win64

6. 在vs中打开本项目以后，还需要进行一些cuda方面的配置：

    * 可以删除Object Files文件夹
    
    * 右键项目 > 生成依赖项 > 生成自定义 > CUDA 9.1 打勾

    * 右键项目 > 属性 > CUDA C/C++ > Common > 64-bit (--machine 64)

    * 右键项目 > 属性 > CUDA C/C++ > Device > compute_60,sm_60 （这是显卡的架构代号，机器不同需要搜一下）

    * 右键*.cu > 属性 > 常规 > 项类型 > CUDAS C/C++

    * 右键*.cu > 属性 > CUDA C/C++ > Common > 64-bit (--machine 64)

    * 右键*.cu > 属性 > CUDA C/C++ > Device > compute_60,sm_60
