# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yikuangyang/dartExample

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yikuangyang/dartExample/build

# Include any dependencies generated for this target.
include CMakeFiles/track_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/track_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/track_test.dir/flags.make

CMakeFiles/track_test.dir/track_copy.cpp.o: CMakeFiles/track_test.dir/flags.make
CMakeFiles/track_test.dir/track_copy.cpp.o: ../track_copy.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yikuangyang/dartExample/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/track_test.dir/track_copy.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/track_test.dir/track_copy.cpp.o -c /home/yikuangyang/dartExample/track_copy.cpp

CMakeFiles/track_test.dir/track_copy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/track_test.dir/track_copy.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yikuangyang/dartExample/track_copy.cpp > CMakeFiles/track_test.dir/track_copy.cpp.i

CMakeFiles/track_test.dir/track_copy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/track_test.dir/track_copy.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yikuangyang/dartExample/track_copy.cpp -o CMakeFiles/track_test.dir/track_copy.cpp.s

CMakeFiles/track_test.dir/track_copy.cpp.o.requires:

.PHONY : CMakeFiles/track_test.dir/track_copy.cpp.o.requires

CMakeFiles/track_test.dir/track_copy.cpp.o.provides: CMakeFiles/track_test.dir/track_copy.cpp.o.requires
	$(MAKE) -f CMakeFiles/track_test.dir/build.make CMakeFiles/track_test.dir/track_copy.cpp.o.provides.build
.PHONY : CMakeFiles/track_test.dir/track_copy.cpp.o.provides

CMakeFiles/track_test.dir/track_copy.cpp.o.provides.build: CMakeFiles/track_test.dir/track_copy.cpp.o


# Object files for target track_test
track_test_OBJECTS = \
"CMakeFiles/track_test.dir/track_copy.cpp.o"

# External object files for target track_test
track_test_EXTERNAL_OBJECTS =

track_test: CMakeFiles/track_test.dir/track_copy.cpp.o
track_test: CMakeFiles/track_test.dir/build.make
track_test: /home/yikuangyang/Pangolin/build/src/libpangolin.so
track_test: /usr/local/cuda/lib64/libcudart_static.a
track_test: /usr/lib/x86_64-linux-gnu/librt.so
track_test: /usr/lib/x86_64-linux-gnu/libglut.so
track_test: /usr/lib/x86_64-linux-gnu/libXmu.so
track_test: /usr/lib/x86_64-linux-gnu/libXi.so
track_test: /usr/lib/x86_64-linux-gnu/libGLU.so
track_test: /usr/lib/x86_64-linux-gnu/libGL.so
track_test: /usr/lib/x86_64-linux-gnu/libGLEW.so
track_test: /usr/lib/x86_64-linux-gnu/libSM.so
track_test: /usr/lib/x86_64-linux-gnu/libICE.so
track_test: /usr/lib/x86_64-linux-gnu/libX11.so
track_test: /usr/lib/x86_64-linux-gnu/libXext.so
track_test: /usr/lib/x86_64-linux-gnu/libdc1394.so
track_test: /opt/ros/kinetic/lib/librealsense.so
track_test: /usr/lib/libOpenNI.so
track_test: /usr/lib/libOpenNI2.so
track_test: /usr/lib/x86_64-linux-gnu/libpng.so
track_test: /usr/lib/x86_64-linux-gnu/libz.so
track_test: /usr/lib/x86_64-linux-gnu/libjpeg.so
track_test: /usr/lib/x86_64-linux-gnu/libtiff.so
track_test: /usr/lib/x86_64-linux-gnu/libIlmImf.so
track_test: CMakeFiles/track_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yikuangyang/dartExample/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable track_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/track_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/track_test.dir/build: track_test

.PHONY : CMakeFiles/track_test.dir/build

CMakeFiles/track_test.dir/requires: CMakeFiles/track_test.dir/track_copy.cpp.o.requires

.PHONY : CMakeFiles/track_test.dir/requires

CMakeFiles/track_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/track_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/track_test.dir/clean

CMakeFiles/track_test.dir/depend:
	cd /home/yikuangyang/dartExample/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yikuangyang/dartExample /home/yikuangyang/dartExample /home/yikuangyang/dartExample/build /home/yikuangyang/dartExample/build /home/yikuangyang/dartExample/build/CMakeFiles/track_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/track_test.dir/depend
