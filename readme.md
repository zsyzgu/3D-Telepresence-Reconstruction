1. 安装最新版本的visual studio

2. 安装point cloud library (pcl): http://pointclouds.org/downloads/windows.html

3. 通过cmake生成vs工程

4. vs工程没有通过cmakelist维护，所以需要手动添加各源文件才能运行

由于pcl1.8.1和vs 15.4 / vs15.5之间的兼容bug，我在pcl源码中改动了一些，才能编译通过：
1. 使用StandaloneMarchingCubes时引出的编译bug: https://github.com/Microsoft/vcpkg/issues/1968
2. 有一个部分是说最新的cuda不支持怎么怎么直接访问显存的，先下载到主存即可
3. 还有很多很多的bug来自pcl、微软和英伟达，我已经忘了，祝您编译愉快！
