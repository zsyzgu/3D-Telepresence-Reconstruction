 cmake_minimum_required( VERSION 2.8 )
set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH} )

project( 3D-Telepresence )
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "3D-Telepresence" )

# OpenMP
FIND_PACKAGE( OpenMP REQUIRED )  
if(OPENMP_FOUND)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()  

# Cuda
find_package( CUDA REQUIRED )
set(CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60;-G;-g)
file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)
source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

# Find Packages
find_package( PCL 1.8 REQUIRED )
set( KinectSDK2_FUSION TRUE )
find_package( KinectSDK2 REQUIRED )
# Find OpenCV, set OpenCV_DIR variable to the absolute path to
# the directory containing OpenCVConfig.cmake file (build\x64\vc14\lib)
find_package(OpenCV REQUIRED)

# Additional Include Directories
include_directories( ${PCL_INCLUDE_DIRS} )
include_directories( ${KinectSDK2_INCLUDE_DIRS} )
include_directories( "C:/Program Files (x86)/Intel RealSense SDK 2.0/include" )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Preprocessor Definitions
add_definitions( ${PCL_DEFINITIONS} )

# Additional Library Directories
link_directories( ${PCL_LIBRARY_DIRS} )
link_directories( ${KinectSDK2_LIBRARY_DIRS} )
link_directories( "C:/Program Files (x86)/Intel RealSense SDK 2.0/lib/x64" )

cuda_add_executable( 3D-Telepresence main.cpp
	CudaHandleError.h
	Parameters.h
	Vertex.h
	RealsenseGrabber.h
	RealsenseGrabber.cpp
	PointCloudProcess.h
	PointCloudProcess.cpp
	SceneRegistration.h
	SceneRegistration.cpp
	Transmission.h
	Transmission.cpp
	Timer.h
	Timer.cpp
	TsdfVolume.h
	TsdfVolume.cpp
	BilateralFilter.cu
	TsdfVolume.cu)

# Additional Dependencies
target_link_libraries( 3D-Telepresence ${PCL_LIBRARIES} )
target_link_libraries( 3D-Telepresence ${KinectSDK2_LIBRARIES} )
target_link_libraries( 3D-Telepresence realsense2.lib )
target_link_libraries( 3D-Telepresence ${OpenCV_LIBS} )
