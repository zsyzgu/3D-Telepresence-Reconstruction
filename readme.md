## Introduction
    
This project is the kernel part of TeleCP: a 3D telepresence software framework that supports high-level co-presence.
The rendering part of TeleCP is at https://github.com/zsyzgu/3D-Telepresence-Rendering
	
TeleCP has two advantages:

* easy-to-deploy

    * commercial hardware
	
	* open-source
	
	* Unity3D plugin for App development

* supporting high-level co-presence

	* temporal synchronicity: low end-to-end delay; sync assistance
	
	* spatial synchronicity: shared space; shared props

## Hardware Requirements

A 3D telepresence system consists of two capture sites. At each site, we have:

* a computer with

	* Intel i7-7700k CPU
	
	* GTX 1080Ti GPU
	
	* Intel X520-SR2 network cards

* HTC Vive

* 3 * Intel Realsense D415

We used optical fiber to connect the two sites.

## Software and Environment

* Install **visual studio 2017 (VS)**. Install neccessary components.

* Install **Cuda 10.0**.

* Install Point Cloud Library (**PCL 1.8.1**).

* Install **Realsense SDK2**.

* Install **Opencv 3.4.0**. Set "OpenCV_DIR" as the path with OpenCVConfig.cmake (e.g., opencv\build\x64\vc15\lib).

* Use **CMake** to generate this project. The complier is "visual studio 2017 v15 win64".

* In the VS project, choose "x64 + Release" complie mode.

* Set the path of *.dll of opencv and realsense in the environment variable.

## Build the Project

* **calibration.exe** is the EXE for background removal, cameras calibration and origin point setting.

* **TeleCP.exe** is the final EXE that connects two ends for the 3D telepresence.

The building steps are as follows:

1. Release **calibration.exe** and **TeleCP.dll** from this project. Before generating **TeleCP.dll**, we should define its version (server/client) in **Parameters.h**.

2. Move **TeleCP.dll** to "Assets/Plugins/" of the rendering part of TeleCP, which is written in Unity3D (https://github.com/zsyzgu/3D-Telepresence-Rendering).

3. Generate **TeleCP.exe** in Unity3D.
