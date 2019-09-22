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
CMAKE_SOURCE_DIR = /home/artfin/Desktop/repos/generalHamDerivatives/v1.0/trajectory-python-integration

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/artfin/HDD-WD/repos/generalHamDerivatives/v1.0/trajectory-python-integration

# Include any dependencies generated for this target.
include CMakeFiles/compute-hamiltonian.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/compute-hamiltonian.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/compute-hamiltonian.dir/flags.make

CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o: CMakeFiles/compute-hamiltonian.dir/flags.make
CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o: /home/artfin/Desktop/repos/generalHamDerivatives/v1.0/trajectory-python-integration/compute-hamiltonian.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/artfin/HDD-WD/repos/generalHamDerivatives/v1.0/trajectory-python-integration/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o -c /home/artfin/Desktop/repos/generalHamDerivatives/v1.0/trajectory-python-integration/compute-hamiltonian.cpp

CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/artfin/Desktop/repos/generalHamDerivatives/v1.0/trajectory-python-integration/compute-hamiltonian.cpp > CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.i

CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/artfin/Desktop/repos/generalHamDerivatives/v1.0/trajectory-python-integration/compute-hamiltonian.cpp -o CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.s

CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o.requires:

.PHONY : CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o.requires

CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o.provides: CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o.requires
	$(MAKE) -f CMakeFiles/compute-hamiltonian.dir/build.make CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o.provides.build
.PHONY : CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o.provides

CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o.provides.build: CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o


# Object files for target compute-hamiltonian
compute__hamiltonian_OBJECTS = \
"CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o"

# External object files for target compute-hamiltonian
compute__hamiltonian_EXTERNAL_OBJECTS =

compute-hamiltonian: CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o
compute-hamiltonian: CMakeFiles/compute-hamiltonian.dir/build.make
compute-hamiltonian: /home/artfin/Desktop/libs/sundials-4.1.0/instdir/lib/libsundials_nvecserial.a
compute-hamiltonian: /home/artfin/Desktop/libs/sundials-4.1.0/instdir/lib/libsundials_cvode.a
compute-hamiltonian: /home/artfin/Desktop/repos/common_h/co2_ar/ai_pes_co2ar.o
compute-hamiltonian: CMakeFiles/compute-hamiltonian.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/artfin/HDD-WD/repos/generalHamDerivatives/v1.0/trajectory-python-integration/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable compute-hamiltonian"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compute-hamiltonian.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/compute-hamiltonian.dir/build: compute-hamiltonian

.PHONY : CMakeFiles/compute-hamiltonian.dir/build

CMakeFiles/compute-hamiltonian.dir/requires: CMakeFiles/compute-hamiltonian.dir/compute-hamiltonian.cpp.o.requires

.PHONY : CMakeFiles/compute-hamiltonian.dir/requires

CMakeFiles/compute-hamiltonian.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/compute-hamiltonian.dir/cmake_clean.cmake
.PHONY : CMakeFiles/compute-hamiltonian.dir/clean

CMakeFiles/compute-hamiltonian.dir/depend:
	cd /media/artfin/HDD-WD/repos/generalHamDerivatives/v1.0/trajectory-python-integration && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/artfin/Desktop/repos/generalHamDerivatives/v1.0/trajectory-python-integration /home/artfin/Desktop/repos/generalHamDerivatives/v1.0/trajectory-python-integration /media/artfin/HDD-WD/repos/generalHamDerivatives/v1.0/trajectory-python-integration /media/artfin/HDD-WD/repos/generalHamDerivatives/v1.0/trajectory-python-integration /media/artfin/HDD-WD/repos/generalHamDerivatives/v1.0/trajectory-python-integration/CMakeFiles/compute-hamiltonian.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/compute-hamiltonian.dir/depend
