# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/build

# Include any dependencies generated for this target.
include CMakeFiles/depth-anything-tensorrt.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/depth-anything-tensorrt.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/depth-anything-tensorrt.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/depth-anything-tensorrt.dir/flags.make

CMakeFiles/depth-anything-tensorrt.dir/main.cpp.o: CMakeFiles/depth-anything-tensorrt.dir/flags.make
CMakeFiles/depth-anything-tensorrt.dir/main.cpp.o: /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/main.cpp
CMakeFiles/depth-anything-tensorrt.dir/main.cpp.o: CMakeFiles/depth-anything-tensorrt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/depth-anything-tensorrt.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/depth-anything-tensorrt.dir/main.cpp.o -MF CMakeFiles/depth-anything-tensorrt.dir/main.cpp.o.d -o CMakeFiles/depth-anything-tensorrt.dir/main.cpp.o -c /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/main.cpp

CMakeFiles/depth-anything-tensorrt.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/depth-anything-tensorrt.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/main.cpp > CMakeFiles/depth-anything-tensorrt.dir/main.cpp.i

CMakeFiles/depth-anything-tensorrt.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/depth-anything-tensorrt.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/main.cpp -o CMakeFiles/depth-anything-tensorrt.dir/main.cpp.s

CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.o: CMakeFiles/depth-anything-tensorrt.dir/flags.make
CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.o: /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/utils.cpp
CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.o: CMakeFiles/depth-anything-tensorrt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.o -MF CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.o.d -o CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.o -c /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/utils.cpp

CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/utils.cpp > CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.i

CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/utils.cpp -o CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.s

CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.o: CMakeFiles/depth-anything-tensorrt.dir/flags.make
CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.o: /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/depth_anything.cpp
CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.o: CMakeFiles/depth-anything-tensorrt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.o -MF CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.o.d -o CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.o -c /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/depth_anything.cpp

CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/depth_anything.cpp > CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.i

CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/depth_anything.cpp -o CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.s

# Object files for target depth-anything-tensorrt
depth__anything__tensorrt_OBJECTS = \
"CMakeFiles/depth-anything-tensorrt.dir/main.cpp.o" \
"CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.o" \
"CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.o"

# External object files for target depth-anything-tensorrt
depth__anything__tensorrt_EXTERNAL_OBJECTS =

depth-anything-tensorrt: CMakeFiles/depth-anything-tensorrt.dir/main.cpp.o
depth-anything-tensorrt: CMakeFiles/depth-anything-tensorrt.dir/utils.cpp.o
depth-anything-tensorrt: CMakeFiles/depth-anything-tensorrt.dir/depth_anything.cpp.o
depth-anything-tensorrt: CMakeFiles/depth-anything-tensorrt.dir/build.make
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_gapi.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_stitching.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_alphamat.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_aruco.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_barcode.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_bgsegm.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_bioinspired.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_ccalib.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_cvv.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_dnn_objdetect.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_dnn_superres.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_dpm.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_face.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_freetype.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_fuzzy.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_hdf.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_hfs.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_img_hash.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_intensity_transform.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_line_descriptor.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_mcc.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_quality.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_rapid.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_reg.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_rgbd.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_saliency.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_stereo.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_structured_light.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_superres.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_surface_matching.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_tracking.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_videostab.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_wechat_qrcode.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_xfeatures2d.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_xobjdetect.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_xphoto.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libcudart_static.a
depth-anything-tensorrt: /usr/lib/x86_64-linux-gnu/librt.so
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_shape.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_highgui.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_datasets.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_plot.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_text.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_ml.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_phase_unwrapping.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_optflow.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_ximgproc.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_video.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_videoio.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_imgcodecs.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_objdetect.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_calib3d.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_dnn.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_features2d.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_flann.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_photo.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_imgproc.so.4.5.5
depth-anything-tensorrt: /mnt/nas/algorithm/dujun.nie/miniconda3/envs/dpth_cpp/lib/libopencv_core.so.4.5.5
depth-anything-tensorrt: CMakeFiles/depth-anything-tensorrt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable depth-anything-tensorrt"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/depth-anything-tensorrt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/depth-anything-tensorrt.dir/build: depth-anything-tensorrt
.PHONY : CMakeFiles/depth-anything-tensorrt.dir/build

CMakeFiles/depth-anything-tensorrt.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/depth-anything-tensorrt.dir/cmake_clean.cmake
.PHONY : CMakeFiles/depth-anything-tensorrt.dir/clean

CMakeFiles/depth-anything-tensorrt.dir/depend:
	cd /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/build /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/build /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/depth-anything-tensorrt-main/build/CMakeFiles/depth-anything-tensorrt.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/depth-anything-tensorrt.dir/depend

