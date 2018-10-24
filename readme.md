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

Combined with the rendering part of TeleCP **3D-Telepresence-Rendering** (https://github.com/zsyzgu/3D-Telepresence-Rendering), we can release the following executable files:

* **calibration.exe**: the EXE for background removal, cameras calibration and original point setting.

* **TeleCP.exe**: the EXE that connects two ends for the 3D telepresence.

The building steps are as follows:

1. Release **calibration.exe** from this project.

2. Define the version (server/client) in **Parameters.h** and release **TeleCP.dll** from this project.

3. Move **TeleCP.dll** to "Assets/Plugins/" in the project **3D-Telepresence-Rendering** (made by Unity3D).

4. Generate **TeleCP.exe** in Unity3D.

In particular, **TeleCP.dll** together with the project **3D-Telepresence-Rendering** make up the Unity3D plugin of TeleCP.

## Run the Project

* Run **calibration.exe** to calibrate each end.

    * **Background removal**: move away everything that are not included in the virtual scene. Press key **"b"** to remove them from the 3D reconstruction.
    
    * **Cameras calibration**: press key **"r"** to start the calibration process. We use a checkerboard for the calibration. The checkerboard is a A4 paper printed with **checkerboard.png** pasted on a flat board.
    
    * **Original point setting**: press key **"o"** to set the original point as the center of checkerboard.
    
    This calibration information is stored in **Background.cfg** and **Extrinsics.cfg**.

* Run **TeleCP.exe** to connect the two ends for the 3D telepresence.

    * We use an optical fibre to connect the two ends.
    
    * Set the IP of **server** to **192.168.1.1**; set the IP of **client** to **192.168.1.2**.
    
    * First, run **TeleCP.exe** in server; then, run **TeleCP.exe** in client.

## API

* **TeleCP.h** summerizes APIs of TeleCP. Please see its annotation.

* **calibration_exe.cpp** and **telepresence_dll.cpp** give examples of how to use these APIs.
