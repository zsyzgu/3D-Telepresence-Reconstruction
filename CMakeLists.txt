 cmake_minimum_required( VERSION 2.8 )
set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH} )

project( pc-recog )
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "pc-recog" )

# Find Packages
find_package( KinectSDK2 REQUIRED )
find_package( PCL 1.8 REQUIRED )
find_package( CUDA REQUIRED )

# Additional Include Directories
include_directories( ${PCL_INCLUDE_DIRS} )
include_directories( ${KinectSDK2_INCLUDE_DIRS} )

# Preprocessor Definitions
add_definitions( ${PCL_DEFINITIONS} )

# Additional Library Directories
link_directories( ${PCL_LIBRARY_DIRS} )
link_directories( ${KinectSDK2_LIBRARY_DIRS} )

cuda_add_executable( pc-recog main.cpp
	Kinect2Grabber.h Kinect2Grabber.cpp
	PointCloudProcess.cpp PointCloudProcess.h
	Recognition.cpp Recognition.h
	Timer.cpp Timer.h
	BilateralFilter.cu )

# Additional Dependencies
target_link_libraries( pc-recog ${PCL_LIBRARIES} )
target_link_libraries( pc-recog ${KinectSDK2_LIBRARIES} )
