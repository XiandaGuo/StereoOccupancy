cmake_minimum_required(VERSION 3.12)
project(depth-anything-tensorrt)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# CUDA PATH
set(CUDA_TOOLKIT_ROOT_DIR /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp)
# CUDNN PATH
set(CUDNN_INSTALL_DIR /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/cudnn-linux-x86_64-8.9.7.29_cuda11-archive)
# Add source files
set(SOURCES
	main.cpp
	utils.cpp
	depth_anything.cpp
    # ... add more source files here if needed
)

# Add headers
set(HEADERS
	utils.h
	depth_anything.h
    # ... add more header files here if needed
)

# Find and include CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find and include OpenCV
set(OpenCV_DIR "/mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set TensorRT path if not set in environment variables
set(TENSORRT_DIR "/mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/TensorRT-8.5.3.1")

# cudnn libraries
set(CUDNN_LIB_DIR ${CUDNN_INSTALL_DIR}/lib)

# cuda libraries
set(CUDA_LIB_DIR ${CUDA_TOOLKIT_ROOT_DIR/lib})
# Include TensorRT
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib ${CUDNN_LIB_DIR} ${CUDA_LIB_DIR})
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)



# Add executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)