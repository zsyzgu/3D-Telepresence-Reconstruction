 cmake_minimum_required( VERSION 2.8 )
set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH} )

project( 3D-Telepresence )
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "test" )

# Cuda
enable_language(CUDA) 

# OpenMP
FIND_PACKAGE( OpenMP REQUIRED )
if(OPENMP_FOUND)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()

# Find Packages
find_package( PCL 1.8 REQUIRED )
find_package( OpenCV REQUIRED ) # Set OpenCV_DIR as the path of OpenCVConfig.cmake (build\x64\vc14\lib)

# Additional Include Directories
include_directories( ${PCL_INCLUDE_DIRS} )
include_directories( "C:/Program Files (x86)/Intel RealSense SDK 2.0/include" )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "./include" )

# Preprocessor Definitions
add_definitions( ${PCL_DEFINITIONS} )

# Additional Library Directories
link_directories( ${PCL_LIBRARY_DIRS} )
link_directories( "C:/Program Files (x86)/Intel RealSense SDK 2.0/lib/x64" )

file(GLOB SRCS include/*.h include/*.cuh src/*.cpp src/cuda/*.cu)

# Test/Development
add_executable( test ${SRCS} tests/main.cpp )
target_link_libraries( test ${PCL_LIBRARIES} )
target_link_libraries( test realsense2.lib )
target_link_libraries( test ${OpenCV_LIBS} )

# Generating calibration.exe
add_executable( calibration ${SRCS} tests/calibration_exe.cpp )
target_link_libraries( calibration ${PCL_LIBRARIES} )
target_link_libraries( calibration realsense2.lib )
target_link_libraries( calibration ${OpenCV_LIBS} )

# Generating TeleCP.dll
add_library( TeleCP SHARED ${SRCS} tests/telepresence_dll.cpp )
target_link_libraries( TeleCP ${PCL_LIBRARIES} )
target_link_libraries( TeleCP realsense2.lib )
target_link_libraries( TeleCP ${OpenCV_LIBS} )
