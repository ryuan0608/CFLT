# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/orangechicken/Documents/Developer/CFLT

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/orangechicken/Documents/Developer/CFLT/build

# Include any dependencies generated for this target.
include CMakeFiles/CLMFaceTest.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CLMFaceTest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CLMFaceTest.dir/flags.make

CMakeFiles/CLMFaceTest.dir/CLMFaceTest.cpp.o: CMakeFiles/CLMFaceTest.dir/flags.make
CMakeFiles/CLMFaceTest.dir/CLMFaceTest.cpp.o: ../CLMFaceTest.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/orangechicken/Documents/Developer/CFLT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CLMFaceTest.dir/CLMFaceTest.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CLMFaceTest.dir/CLMFaceTest.cpp.o -c /Users/orangechicken/Documents/Developer/CFLT/CLMFaceTest.cpp

CMakeFiles/CLMFaceTest.dir/CLMFaceTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CLMFaceTest.dir/CLMFaceTest.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/orangechicken/Documents/Developer/CFLT/CLMFaceTest.cpp > CMakeFiles/CLMFaceTest.dir/CLMFaceTest.cpp.i

CMakeFiles/CLMFaceTest.dir/CLMFaceTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CLMFaceTest.dir/CLMFaceTest.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/orangechicken/Documents/Developer/CFLT/CLMFaceTest.cpp -o CMakeFiles/CLMFaceTest.dir/CLMFaceTest.cpp.s

# Object files for target CLMFaceTest
CLMFaceTest_OBJECTS = \
"CMakeFiles/CLMFaceTest.dir/CLMFaceTest.cpp.o"

# External object files for target CLMFaceTest
CLMFaceTest_EXTERNAL_OBJECTS =

CLMFaceTest: CMakeFiles/CLMFaceTest.dir/CLMFaceTest.cpp.o
CLMFaceTest: CMakeFiles/CLMFaceTest.dir/build.make
CLMFaceTest: HyperLandmarks/libHyperLandmarks.a
CLMFaceTest: CLM/libCLM.a
CLMFaceTest: /usr/local/lib/libopencv_calib3d.3.4.2.dylib
CLMFaceTest: /usr/local/lib/libopencv_objdetect.3.4.2.dylib
CLMFaceTest: /usr/local/lib/libopencv_tracking.3.4.2.dylib
CLMFaceTest: /usr/local/lib/libopencv_features2d.3.4.2.dylib
CLMFaceTest: /usr/local/lib/libopencv_highgui.3.4.2.dylib
CLMFaceTest: /usr/local/lib/libopencv_videoio.3.4.2.dylib
CLMFaceTest: /usr/local/lib/libopencv_plot.3.4.2.dylib
CLMFaceTest: /usr/local/lib/libopencv_video.3.4.2.dylib
CLMFaceTest: /usr/local/lib/libopencv_dnn.3.4.2.dylib
CLMFaceTest: /usr/local/lib/libopencv_datasets.3.4.2.dylib
CLMFaceTest: /usr/local/lib/libopencv_flann.3.4.2.dylib
CLMFaceTest: /usr/local/lib/libopencv_imgcodecs.3.4.2.dylib
CLMFaceTest: /usr/local/lib/libopencv_imgproc.3.4.2.dylib
CLMFaceTest: /usr/local/lib/libopencv_ml.3.4.2.dylib
CLMFaceTest: /usr/local/lib/libopencv_core.3.4.2.dylib
CLMFaceTest: CMakeFiles/CLMFaceTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/orangechicken/Documents/Developer/CFLT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable CLMFaceTest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CLMFaceTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CLMFaceTest.dir/build: CLMFaceTest

.PHONY : CMakeFiles/CLMFaceTest.dir/build

CMakeFiles/CLMFaceTest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CLMFaceTest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CLMFaceTest.dir/clean

CMakeFiles/CLMFaceTest.dir/depend:
	cd /Users/orangechicken/Documents/Developer/CFLT/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/orangechicken/Documents/Developer/CFLT /Users/orangechicken/Documents/Developer/CFLT /Users/orangechicken/Documents/Developer/CFLT/build /Users/orangechicken/Documents/Developer/CFLT/build /Users/orangechicken/Documents/Developer/CFLT/build/CMakeFiles/CLMFaceTest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CLMFaceTest.dir/depend
