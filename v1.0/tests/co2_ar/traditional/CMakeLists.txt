cmake_minimum_required(VERSION 3.10)
project(co2-ar-old-derivatives)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED  11)
set(CMAKE_CXX_FLAGS "-g -O3 -Wall -Wextra")

set(EIGEN_DIR "/usr/local/include/eigen3")
include_directories(${EIGEN_DIR})

set(HOME ON)
set(ODYSSEY OFF)
if (${HOME})
    message([STATUS] " SELECTED HOME COMPILATION OPTION") 
endif()
if (${ODYSSEY})
    message([STATUS] " SELECTED ODYSSEY COMPILATION OPTION")
endif()

if (${HOME})
    set(CVODE_DIR "/home/artfin/Desktop/libs/sundials-4.1.0/instdir/include")
    set(COMMON_DIR "/home/artfin/Desktop/repos/common_h")
endif()

if (${ODYSSEY})
    set(CVODE_DIR "/n/home11/artfin/libs/sundials-4.1.0/instdir/include")
    set(COMMON_DIR "/n/home11/artfin/common_h/")
endif()

include_directories(${CVODE_DIR})
include_directories(${COMMON_DIR})

add_executable(
        main
        main.cpp
        matrix_euler.hpp
        matrix_euler.cpp
)


if (${HOME})
    set(CVODE_LINK1 "/home/artfin/Desktop/libs/sundials-4.1.0/instdir/lib/libsundials_nvecserial.a")
    set(CVODE_LINK2 "/home/artfin/Desktop/libs/sundials-4.1.0/instdir/lib/libsundials_cvode.a")
endif()

if (${ODYSSEY})
    set(CVODE_LINK1 "/n/home11/artfin/libs/sundials-4.1.0/instdir/lib/libsundials_nvecserial.a")
    set(CVODE_LINK2 "/n/home11/artfin/libs/sundials-4.1.0/instdir/lib/libsundials_cvode.a")
endif()

target_link_libraries(main ${CVODE_LINK1} ${CVODE_LINK2})

#find_package(GSL REQUIRED)
#target_link_libraries(angles-exchange-spectra GSL::gsl GSL::gslcblas)

if (${ODYSSEY})
    set(FORTRAN_LINK "/usr/lib/gcc/x86_64-redhat-linux/4.8.2/libgfortran.so")
endif()
if (${HOME})
    set(FORTRAN_LINK "/usr/lib/gcc/x86_64-linux-gnu/7/libgfortran.so")
endif()
#target_link_libraries(angles-exchange-spectra ${FORTRAN_LINK})

