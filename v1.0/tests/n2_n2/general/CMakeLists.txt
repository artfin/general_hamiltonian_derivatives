cmake_minimum_required(VERSION 3.10)
project(general-hamiltonian-derivatives)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED  11)
set(CMAKE_CXX_FLAGS "-g -O3 -Wall -Wextra")

set(EIGEN_DIR "/usr/local/include/eigen3")
include_directories(${EIGEN_DIR})

set(COMMON_DIR "/home/artfin/Desktop/repos/common_h")

include_directories(${COMMON_DIR})

set(GENERAL_DIR "../../../")
include_directories(${GENERAL_DIR})

add_executable(
    general
    main.cpp
    generated.hpp
)

set(POT "/home/artfin/Desktop/repos/common_h/n2_n2/ai_pes_n2_n2.o")
target_link_libraries(general ${POT})
