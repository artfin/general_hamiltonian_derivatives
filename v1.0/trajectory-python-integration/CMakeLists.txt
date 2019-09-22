cmake_minimum_required(VERSION 3.10)
project(trajectory-wrapper)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED  11)
set(CMAKE_CXX_FLAGS "-g -O3 -Wall -Wextra")

set(EIGEN_DIR "/usr/local/include/eigen3")
include_directories(${EIGEN_DIR})

set(CVODE_DIR "/home/artfin/Desktop/libs/sundials-4.1.0/instdir/include")
set(COMMON_DIR "/home/artfin/Desktop/repos/common_h")

include_directories(${CVODE_DIR})
include_directories(${COMMON_DIR})

set(GENERAL_DIR "/media/artfin/HDD-WD/repos/generalHamDerivatives/v1.0/")
include_directories(${GENERAL_DIR})

add_executable(
    compute-derivatives
    compute-derivatives.cpp 
    generated.hpp
)

add_executable(
    compute-hamiltonian
    compute-hamiltonian.cpp
    generated.hpp
)

add_executable(
    compute-trajectory
    compute-trajectory.cpp
    generated.hpp
    trajectory.cpp
    trajectory.hpp
    phase_point.hpp
    phase_point.cpp
)

set(CVODE_LINK1 "/home/artfin/Desktop/libs/sundials-4.1.0/instdir/lib/libsundials_nvecserial.a")
set(CVODE_LINK2 "/home/artfin/Desktop/libs/sundials-4.1.0/instdir/lib/libsundials_cvode.a")
set(POT "/home/artfin/Desktop/repos/common_h/co2_ar/ai_pes_co2ar.o")

target_link_libraries(compute-derivatives ${CVODE_LINK1} ${CVODE_LINK2} ${POT})
target_link_libraries(compute-hamiltonian ${CVODE_LINK1} ${CVODE_LINK2} ${POT})
target_link_libraries(compute-trajectory ${CVODE_LINK1} ${CVODE_LINK2} ${POT})
