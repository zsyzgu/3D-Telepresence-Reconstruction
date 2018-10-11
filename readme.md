## Introduction:
    
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

## Hardware requirements:

A 3D telepresence system consists of two capture sites. At each site, we have:

* a computer with

	* Intel i7-7700k CPU
	
	* GTX 1080Ti GPU
	
	* Intel X520-SR2 network cards

* HTC Vive

* 3 * Intel Realsense D415

We used optical fiber to connect the two sites.

## Environment

* Install **visual studio 2017 (VS)**. Install neccessary components.

* Install **Cuda 10.0**.

* Install Point Cloud Library (**PCL 1.8.1**).

* Install **Realsense SDK2**.

* Install **Opencv 3.4.0**. Set "OpenCV_DIR" as the path with OpenCVConfig.cmake (e.g., opencv\build\x64\vc15\lib).

* Use **CMake** to generate this project. The complier is "visual studio 2017 v15 win64".

* In the VS project, choose "x64 + Release" complie mode.

* Set the path of *.dll of opencv and realsense in the environment variable.
