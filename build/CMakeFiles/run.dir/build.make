# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/nvidia/Desktop/2023-first-energy

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/Desktop/2023-first-energy/build

# Include any dependencies generated for this target.
include CMakeFiles/run.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/run.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/run.dir/flags.make

CMakeFiles/run.dir/main.cpp.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/2023-first-energy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/run.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run.dir/main.cpp.o -c /home/nvidia/Desktop/2023-first-energy/main.cpp

CMakeFiles/run.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/2023-first-energy/main.cpp > CMakeFiles/run.dir/main.cpp.i

CMakeFiles/run.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/2023-first-energy/main.cpp -o CMakeFiles/run.dir/main.cpp.s

CMakeFiles/run.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/run.dir/main.cpp.o.requires

CMakeFiles/run.dir/main.cpp.o.provides: CMakeFiles/run.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/run.dir/main.cpp.o.provides

CMakeFiles/run.dir/main.cpp.o.provides.build: CMakeFiles/run.dir/main.cpp.o


CMakeFiles/run.dir/src/ArmorDetector.cpp.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/src/ArmorDetector.cpp.o: ../src/ArmorDetector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/2023-first-energy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/run.dir/src/ArmorDetector.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run.dir/src/ArmorDetector.cpp.o -c /home/nvidia/Desktop/2023-first-energy/src/ArmorDetector.cpp

CMakeFiles/run.dir/src/ArmorDetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run.dir/src/ArmorDetector.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/2023-first-energy/src/ArmorDetector.cpp > CMakeFiles/run.dir/src/ArmorDetector.cpp.i

CMakeFiles/run.dir/src/ArmorDetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run.dir/src/ArmorDetector.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/2023-first-energy/src/ArmorDetector.cpp -o CMakeFiles/run.dir/src/ArmorDetector.cpp.s

CMakeFiles/run.dir/src/ArmorDetector.cpp.o.requires:

.PHONY : CMakeFiles/run.dir/src/ArmorDetector.cpp.o.requires

CMakeFiles/run.dir/src/ArmorDetector.cpp.o.provides: CMakeFiles/run.dir/src/ArmorDetector.cpp.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/src/ArmorDetector.cpp.o.provides.build
.PHONY : CMakeFiles/run.dir/src/ArmorDetector.cpp.o.provides

CMakeFiles/run.dir/src/ArmorDetector.cpp.o.provides.build: CMakeFiles/run.dir/src/ArmorDetector.cpp.o


CMakeFiles/run.dir/src/CRC_Check.cpp.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/src/CRC_Check.cpp.o: ../src/CRC_Check.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/2023-first-energy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/run.dir/src/CRC_Check.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run.dir/src/CRC_Check.cpp.o -c /home/nvidia/Desktop/2023-first-energy/src/CRC_Check.cpp

CMakeFiles/run.dir/src/CRC_Check.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run.dir/src/CRC_Check.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/2023-first-energy/src/CRC_Check.cpp > CMakeFiles/run.dir/src/CRC_Check.cpp.i

CMakeFiles/run.dir/src/CRC_Check.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run.dir/src/CRC_Check.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/2023-first-energy/src/CRC_Check.cpp -o CMakeFiles/run.dir/src/CRC_Check.cpp.s

CMakeFiles/run.dir/src/CRC_Check.cpp.o.requires:

.PHONY : CMakeFiles/run.dir/src/CRC_Check.cpp.o.requires

CMakeFiles/run.dir/src/CRC_Check.cpp.o.provides: CMakeFiles/run.dir/src/CRC_Check.cpp.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/src/CRC_Check.cpp.o.provides.build
.PHONY : CMakeFiles/run.dir/src/CRC_Check.cpp.o.provides

CMakeFiles/run.dir/src/CRC_Check.cpp.o.provides.build: CMakeFiles/run.dir/src/CRC_Check.cpp.o


CMakeFiles/run.dir/src/KAL.cpp.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/src/KAL.cpp.o: ../src/KAL.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/2023-first-energy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/run.dir/src/KAL.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run.dir/src/KAL.cpp.o -c /home/nvidia/Desktop/2023-first-energy/src/KAL.cpp

CMakeFiles/run.dir/src/KAL.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run.dir/src/KAL.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/2023-first-energy/src/KAL.cpp > CMakeFiles/run.dir/src/KAL.cpp.i

CMakeFiles/run.dir/src/KAL.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run.dir/src/KAL.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/2023-first-energy/src/KAL.cpp -o CMakeFiles/run.dir/src/KAL.cpp.s

CMakeFiles/run.dir/src/KAL.cpp.o.requires:

.PHONY : CMakeFiles/run.dir/src/KAL.cpp.o.requires

CMakeFiles/run.dir/src/KAL.cpp.o.provides: CMakeFiles/run.dir/src/KAL.cpp.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/src/KAL.cpp.o.provides.build
.PHONY : CMakeFiles/run.dir/src/KAL.cpp.o.provides

CMakeFiles/run.dir/src/KAL.cpp.o.provides.build: CMakeFiles/run.dir/src/KAL.cpp.o


CMakeFiles/run.dir/src/Thread.cpp.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/src/Thread.cpp.o: ../src/Thread.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/2023-first-energy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/run.dir/src/Thread.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run.dir/src/Thread.cpp.o -c /home/nvidia/Desktop/2023-first-energy/src/Thread.cpp

CMakeFiles/run.dir/src/Thread.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run.dir/src/Thread.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/2023-first-energy/src/Thread.cpp > CMakeFiles/run.dir/src/Thread.cpp.i

CMakeFiles/run.dir/src/Thread.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run.dir/src/Thread.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/2023-first-energy/src/Thread.cpp -o CMakeFiles/run.dir/src/Thread.cpp.s

CMakeFiles/run.dir/src/Thread.cpp.o.requires:

.PHONY : CMakeFiles/run.dir/src/Thread.cpp.o.requires

CMakeFiles/run.dir/src/Thread.cpp.o.provides: CMakeFiles/run.dir/src/Thread.cpp.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/src/Thread.cpp.o.provides.build
.PHONY : CMakeFiles/run.dir/src/Thread.cpp.o.provides

CMakeFiles/run.dir/src/Thread.cpp.o.provides.build: CMakeFiles/run.dir/src/Thread.cpp.o


CMakeFiles/run.dir/src/camera.cpp.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/src/camera.cpp.o: ../src/camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/2023-first-energy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/run.dir/src/camera.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run.dir/src/camera.cpp.o -c /home/nvidia/Desktop/2023-first-energy/src/camera.cpp

CMakeFiles/run.dir/src/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run.dir/src/camera.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/2023-first-energy/src/camera.cpp > CMakeFiles/run.dir/src/camera.cpp.i

CMakeFiles/run.dir/src/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run.dir/src/camera.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/2023-first-energy/src/camera.cpp -o CMakeFiles/run.dir/src/camera.cpp.s

CMakeFiles/run.dir/src/camera.cpp.o.requires:

.PHONY : CMakeFiles/run.dir/src/camera.cpp.o.requires

CMakeFiles/run.dir/src/camera.cpp.o.provides: CMakeFiles/run.dir/src/camera.cpp.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/src/camera.cpp.o.provides.build
.PHONY : CMakeFiles/run.dir/src/camera.cpp.o.provides

CMakeFiles/run.dir/src/camera.cpp.o.provides.build: CMakeFiles/run.dir/src/camera.cpp.o


CMakeFiles/run.dir/src/energy_get.cpp.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/src/energy_get.cpp.o: ../src/energy_get.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/2023-first-energy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/run.dir/src/energy_get.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run.dir/src/energy_get.cpp.o -c /home/nvidia/Desktop/2023-first-energy/src/energy_get.cpp

CMakeFiles/run.dir/src/energy_get.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run.dir/src/energy_get.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/2023-first-energy/src/energy_get.cpp > CMakeFiles/run.dir/src/energy_get.cpp.i

CMakeFiles/run.dir/src/energy_get.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run.dir/src/energy_get.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/2023-first-energy/src/energy_get.cpp -o CMakeFiles/run.dir/src/energy_get.cpp.s

CMakeFiles/run.dir/src/energy_get.cpp.o.requires:

.PHONY : CMakeFiles/run.dir/src/energy_get.cpp.o.requires

CMakeFiles/run.dir/src/energy_get.cpp.o.provides: CMakeFiles/run.dir/src/energy_get.cpp.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/src/energy_get.cpp.o.provides.build
.PHONY : CMakeFiles/run.dir/src/energy_get.cpp.o.provides

CMakeFiles/run.dir/src/energy_get.cpp.o.provides.build: CMakeFiles/run.dir/src/energy_get.cpp.o


CMakeFiles/run.dir/src/energy_predict.cpp.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/src/energy_predict.cpp.o: ../src/energy_predict.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/2023-first-energy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/run.dir/src/energy_predict.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run.dir/src/energy_predict.cpp.o -c /home/nvidia/Desktop/2023-first-energy/src/energy_predict.cpp

CMakeFiles/run.dir/src/energy_predict.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run.dir/src/energy_predict.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/2023-first-energy/src/energy_predict.cpp > CMakeFiles/run.dir/src/energy_predict.cpp.i

CMakeFiles/run.dir/src/energy_predict.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run.dir/src/energy_predict.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/2023-first-energy/src/energy_predict.cpp -o CMakeFiles/run.dir/src/energy_predict.cpp.s

CMakeFiles/run.dir/src/energy_predict.cpp.o.requires:

.PHONY : CMakeFiles/run.dir/src/energy_predict.cpp.o.requires

CMakeFiles/run.dir/src/energy_predict.cpp.o.provides: CMakeFiles/run.dir/src/energy_predict.cpp.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/src/energy_predict.cpp.o.provides.build
.PHONY : CMakeFiles/run.dir/src/energy_predict.cpp.o.provides

CMakeFiles/run.dir/src/energy_predict.cpp.o.provides.build: CMakeFiles/run.dir/src/energy_predict.cpp.o


CMakeFiles/run.dir/src/energy_state.cpp.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/src/energy_state.cpp.o: ../src/energy_state.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/2023-first-energy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/run.dir/src/energy_state.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run.dir/src/energy_state.cpp.o -c /home/nvidia/Desktop/2023-first-energy/src/energy_state.cpp

CMakeFiles/run.dir/src/energy_state.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run.dir/src/energy_state.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/2023-first-energy/src/energy_state.cpp > CMakeFiles/run.dir/src/energy_state.cpp.i

CMakeFiles/run.dir/src/energy_state.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run.dir/src/energy_state.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/2023-first-energy/src/energy_state.cpp -o CMakeFiles/run.dir/src/energy_state.cpp.s

CMakeFiles/run.dir/src/energy_state.cpp.o.requires:

.PHONY : CMakeFiles/run.dir/src/energy_state.cpp.o.requires

CMakeFiles/run.dir/src/energy_state.cpp.o.provides: CMakeFiles/run.dir/src/energy_state.cpp.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/src/energy_state.cpp.o.provides.build
.PHONY : CMakeFiles/run.dir/src/energy_state.cpp.o.provides

CMakeFiles/run.dir/src/energy_state.cpp.o.provides.build: CMakeFiles/run.dir/src/energy_state.cpp.o


CMakeFiles/run.dir/src/serialport.cpp.o: CMakeFiles/run.dir/flags.make
CMakeFiles/run.dir/src/serialport.cpp.o: ../src/serialport.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/2023-first-energy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/run.dir/src/serialport.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run.dir/src/serialport.cpp.o -c /home/nvidia/Desktop/2023-first-energy/src/serialport.cpp

CMakeFiles/run.dir/src/serialport.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run.dir/src/serialport.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/2023-first-energy/src/serialport.cpp > CMakeFiles/run.dir/src/serialport.cpp.i

CMakeFiles/run.dir/src/serialport.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run.dir/src/serialport.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/2023-first-energy/src/serialport.cpp -o CMakeFiles/run.dir/src/serialport.cpp.s

CMakeFiles/run.dir/src/serialport.cpp.o.requires:

.PHONY : CMakeFiles/run.dir/src/serialport.cpp.o.requires

CMakeFiles/run.dir/src/serialport.cpp.o.provides: CMakeFiles/run.dir/src/serialport.cpp.o.requires
	$(MAKE) -f CMakeFiles/run.dir/build.make CMakeFiles/run.dir/src/serialport.cpp.o.provides.build
.PHONY : CMakeFiles/run.dir/src/serialport.cpp.o.provides

CMakeFiles/run.dir/src/serialport.cpp.o.provides.build: CMakeFiles/run.dir/src/serialport.cpp.o


# Object files for target run
run_OBJECTS = \
"CMakeFiles/run.dir/main.cpp.o" \
"CMakeFiles/run.dir/src/ArmorDetector.cpp.o" \
"CMakeFiles/run.dir/src/CRC_Check.cpp.o" \
"CMakeFiles/run.dir/src/KAL.cpp.o" \
"CMakeFiles/run.dir/src/Thread.cpp.o" \
"CMakeFiles/run.dir/src/camera.cpp.o" \
"CMakeFiles/run.dir/src/energy_get.cpp.o" \
"CMakeFiles/run.dir/src/energy_predict.cpp.o" \
"CMakeFiles/run.dir/src/energy_state.cpp.o" \
"CMakeFiles/run.dir/src/serialport.cpp.o"

# External object files for target run
run_EXTERNAL_OBJECTS =

run: CMakeFiles/run.dir/main.cpp.o
run: CMakeFiles/run.dir/src/ArmorDetector.cpp.o
run: CMakeFiles/run.dir/src/CRC_Check.cpp.o
run: CMakeFiles/run.dir/src/KAL.cpp.o
run: CMakeFiles/run.dir/src/Thread.cpp.o
run: CMakeFiles/run.dir/src/camera.cpp.o
run: CMakeFiles/run.dir/src/energy_get.cpp.o
run: CMakeFiles/run.dir/src/energy_predict.cpp.o
run: CMakeFiles/run.dir/src/energy_state.cpp.o
run: CMakeFiles/run.dir/src/serialport.cpp.o
run: CMakeFiles/run.dir/build.make
run: /usr/lib/aarch64-linux-gnu/libSM.so
run: /usr/lib/aarch64-linux-gnu/libICE.so
run: /usr/lib/aarch64-linux-gnu/libX11.so
run: /usr/lib/aarch64-linux-gnu/libXext.so
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_img_hash.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /usr/local/lib/libopencv_world.so.4.5.5
run: /lib/libMVSDK.so
run: /usr/local/lib/libopencv_world.so.4.5.5
run: CMakeFiles/run.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/Desktop/2023-first-energy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX executable run"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/run.dir/link.txt --verbose=$(VERBOSE)
	../tools/create-startup.sh /home/nvidia/Desktop/2023-first-energy /home/nvidia/Desktop/2023-first-energy/tools

# Rule to build all files generated by this target.
CMakeFiles/run.dir/build: run

.PHONY : CMakeFiles/run.dir/build

CMakeFiles/run.dir/requires: CMakeFiles/run.dir/main.cpp.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/src/ArmorDetector.cpp.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/src/CRC_Check.cpp.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/src/KAL.cpp.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/src/Thread.cpp.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/src/camera.cpp.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/src/energy_get.cpp.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/src/energy_predict.cpp.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/src/energy_state.cpp.o.requires
CMakeFiles/run.dir/requires: CMakeFiles/run.dir/src/serialport.cpp.o.requires

.PHONY : CMakeFiles/run.dir/requires

CMakeFiles/run.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/run.dir/cmake_clean.cmake
.PHONY : CMakeFiles/run.dir/clean

CMakeFiles/run.dir/depend:
	cd /home/nvidia/Desktop/2023-first-energy/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/Desktop/2023-first-energy /home/nvidia/Desktop/2023-first-energy /home/nvidia/Desktop/2023-first-energy/build /home/nvidia/Desktop/2023-first-energy/build /home/nvidia/Desktop/2023-first-energy/build/CMakeFiles/run.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/run.dir/depend
