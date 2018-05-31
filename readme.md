## 项目介绍

本项目通过 Kinect v2， G1080Ti显卡和网线直连，实现实时的3D Telepresence。

## 环境设置

1. 安装visual studio 2017，版本最好是v15.4。（v15.5与cuda9.1不兼容，v15.6不清楚)

2. 安装Cuda9.1，注意此步骤需在安装vs以后进行，这样vs中就会被装上cuda的插件。

3. 确保你的visual studio可以将一个普通工程改成CUDA工程，这个流程可以网上搜索一下。

4. 安装point cloud library (pcl) 1.8.1 with GPU，需要手动编译，会遇到很多的坑，建议按照网上精细教程进行。

5. 通过cmake生成本项目的vs工程，编译器选择visual studio 2017 v15 Win64

6. 在vs中打开本项目以后，还需要进行一些配置：

	* 需选择x64编译模式；最好选择Release模式（Debug模式太慢）

	* 可以删除Object Files文件夹
    
	* 右键项目 > 生成依赖项 > 生成自定义 > CUDA 9.1 打勾

	* 右键项目 > 属性 > CUDA C/C++ > Common > 64-bit (--machine 64)

	* 右键项目 > 属性 > CUDA C/C++ > Device > compute_60,sm_60 （这是显卡的架构代号，机器不同需要搜一下）

	* 右键*.cu > 属性 > 常规 > 项类型 > CUDAS C/C++

7. Realsense配置：下载并安装realsense SDK2, 目前realsense在VS中只能通过props文件来配置：
	
	* 在属性管理器，将intel.realsense.props配置文件加入工程中
	
	* 双击intel.realsense，将其中的变化部分（粗体字）手动添加到工程属性里

8. 最后在Visual Studio中删除CmakeList.txt，以免Cmake重新编译了。（这一系列麻烦操作的根源在于各种库不一定都能被Cmake支持）
