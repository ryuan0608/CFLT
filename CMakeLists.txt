cmake_minimum_required (VERSION 3.2)
project (CLMFace)

# c++14 is required
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# opencv is required
find_package(OpenCV 3.3 REQUIRED PATHS ${OpenCV_DIR} COMPONENTS core imgproc calib3d highgui objdetect tracking)

if(${OpenCV_FOUND})
	MESSAGE("OpenCV information:") 
	MESSAGE("  OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}") 
	MESSAGE("  OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}") 
	MESSAGE("  OpenCV_LIBRARY_DIRS: ${OpenCV_LINK_DIRECTORIES}") 
else()
    MESSAGE(FATAL_ERROR "OpenCV not found in the system.")
endif()

include_directories("${CMAKE_SOURCE_DIR}/HyperLandmarks/src/include")
include_directories("${CMAKE_SOURCE_DIR}/CLM/include")

# two submodules are involved
add_subdirectory(HyperLandmarks/)
add_subdirectory(CLM/)

# add executable
add_executable(CLMFaceTest CLMFaceTest.cpp)
target_link_libraries(CLMFaceTest HyperLandmarks)
target_link_libraries(CLMFaceTest CLM)
target_link_libraries(CLMFaceTest ${OpenCV_LIBS})
