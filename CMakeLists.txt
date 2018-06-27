project("dartExample")
cmake_minimum_required(VERSION 2.8)

if( NOT CMAKE_BUILD_TYPE )
   set( CMAKE_BUILD_TYPE Release )
endif()

add_definitions(-std=c++11)

find_path(DART_ROOT NAMES lib/libdart.so src/tracker.h HINTS ${PROJECT_SOURCE_DIR}/../dart ${PROJECT_SOUCE_DIR}/dart)
message(STATUS "Found dart is at ${DART_ROOT}")
include_directories(${DART_ROOT}/src)
link_directories(${DART_ROOT}/lib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${DART_ROOT}/CMakeModules/")

# find packages with modules
find_package(Pangolin REQUIRED)
find_package(DEPTHSENSE)
find_package(CUDA REQUIRED)
find_package(GLUT REQUIRED)
find_package(Boost REQUIRED thread system)


# find packages with pkg-config
find_package(PkgConfig)
pkg_check_modules(eigen3 REQUIRED eigen3)
pkg_check_modules(matheval REQUIRED libmatheval)

if(DEPTHSENSE_FOUND)
    add_definitions(-DHAS_INTEL)
    include_directories(${DEPTHSENSE_INCLUDE_DIR})
    link_libraries(${DEPTHSENSE_LIBRARIES})
else()
    message(STATUS "no DS")
endif()

include_directories(  ${Pangolin_INCLUDE_DIRS} )
link_directories(     ${Pangolin_LIBRARY_DIRS} )
link_libraries(       ${Pangolin_LIBRARIES}    )

include_directories(
    ${eigen3_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

link_libraries(
    ${Pangolin_LIBRARIES}
    ${matheval_LIBRARIES}
)

##########################################################################
#   Build GPU or CPU-only executable depending on cuda                   #
##########################################################################

include_directories( ${CUDA_INCLUDE_DIRS} )
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CUDA_NVCC_FLAGS "-arch=sm_35" "--use_fast_math"  "-O3" "--ptxas-options=--verbose") # "-fmad=false" "-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP"
add_definitions(-DCUDA_BUILD)

set(all_sources ${sources} ${cpu_sources} ${gpu_sources})
link_directories( ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)

cuda_add_executable(
    #executable
    track_space_justin_demo
    #sources
    ${all_sources}
    #main
    track_space_justin_demo.cpp
)
target_link_libraries(track_space_justin_demo dart)

cuda_add_executable(
    #executable
    my_track
    #sources
    ${all_sources}
    #main
    mytrack.cpp
)
target_link_libraries(my_track dart)
target_link_libraries(my_track ${GLUT_LIBRARY})

cuda_add_executable(
    #executable
    track_test
    #sources
    ${all_sources}
    #main
    track_copy.cpp
)
target_link_libraries(track_test dart)
target_link_libraries(track_test ${GLUT_LIBRARY})

cuda_add_executable(
    #executable
    track_mani
    #sources
    ${all_sources}
    #main
    track_manipulation.cpp
)
target_link_libraries(track_mani dart)
target_link_libraries(track_mani ${GLUT_LIBRARY})