# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.19

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2021.1\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2021.1\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = S:\Programacao\CParalelaUfrgs2020_2\Trabalho2_2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = S:\Programacao\CParalelaUfrgs2020_2\Trabalho2_2\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Trabalho2_2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Trabalho2_2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Trabalho2_2.dir/flags.make

CMakeFiles/Trabalho2_2.dir/main.cpp.obj: CMakeFiles/Trabalho2_2.dir/flags.make
CMakeFiles/Trabalho2_2.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=S:\Programacao\CParalelaUfrgs2020_2\Trabalho2_2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Trabalho2_2.dir/main.cpp.obj"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\Trabalho2_2.dir\main.cpp.obj -c S:\Programacao\CParalelaUfrgs2020_2\Trabalho2_2\main.cpp

CMakeFiles/Trabalho2_2.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Trabalho2_2.dir/main.cpp.i"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E S:\Programacao\CParalelaUfrgs2020_2\Trabalho2_2\main.cpp > CMakeFiles\Trabalho2_2.dir\main.cpp.i

CMakeFiles/Trabalho2_2.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Trabalho2_2.dir/main.cpp.s"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S S:\Programacao\CParalelaUfrgs2020_2\Trabalho2_2\main.cpp -o CMakeFiles\Trabalho2_2.dir\main.cpp.s

# Object files for target Trabalho2_2
Trabalho2_2_OBJECTS = \
"CMakeFiles/Trabalho2_2.dir/main.cpp.obj"

# External object files for target Trabalho2_2
Trabalho2_2_EXTERNAL_OBJECTS =

Trabalho2_2.exe: CMakeFiles/Trabalho2_2.dir/main.cpp.obj
Trabalho2_2.exe: CMakeFiles/Trabalho2_2.dir/build.make
Trabalho2_2.exe: CMakeFiles/Trabalho2_2.dir/linklibs.rsp
Trabalho2_2.exe: CMakeFiles/Trabalho2_2.dir/objects1.rsp
Trabalho2_2.exe: CMakeFiles/Trabalho2_2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=S:\Programacao\CParalelaUfrgs2020_2\Trabalho2_2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Trabalho2_2.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\Trabalho2_2.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Trabalho2_2.dir/build: Trabalho2_2.exe

.PHONY : CMakeFiles/Trabalho2_2.dir/build

CMakeFiles/Trabalho2_2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\Trabalho2_2.dir\cmake_clean.cmake
.PHONY : CMakeFiles/Trabalho2_2.dir/clean

CMakeFiles/Trabalho2_2.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" S:\Programacao\CParalelaUfrgs2020_2\Trabalho2_2 S:\Programacao\CParalelaUfrgs2020_2\Trabalho2_2 S:\Programacao\CParalelaUfrgs2020_2\Trabalho2_2\cmake-build-debug S:\Programacao\CParalelaUfrgs2020_2\Trabalho2_2\cmake-build-debug S:\Programacao\CParalelaUfrgs2020_2\Trabalho2_2\cmake-build-debug\CMakeFiles\Trabalho2_2.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Trabalho2_2.dir/depend

