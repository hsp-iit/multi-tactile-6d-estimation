cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#### Build PhysX library ####
# PHYSX_PATH - path to the `{cloned repository}/physx` repo directory git://github.com/NVIDIAGameWorks/PhysX.git

project(example)
set(PHYSX_PATH /path/to/physx/directory)
set(PHYSX_ROOT_DIR ${PHYSX_PATH})
set(PHYSX_INCLUDE_DIRS ${PHYSX_PATH}/include/ ${PHYSX_PATH}/../pxshared/include/ )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (OpenMP)
if(OpenMP_CXX_FOUND)
  message("Found")
endif()
set(PHYSX_LIBRARIES
	PhysXExtensions
	PhysX
	PhysXPvdSDK
	PhysXVehicle
	PhysXCharacterKinematic
	PhysXCooking
	PhysXCommon
	PhysXFoundation
)

ADD_EXECUTABLE(script

    src/main_general_general.cpp
    src/wavefront.h
    src/wavefront.cpp
    src/FrameworkFoundation.h
    src/PxTkFile.h
)

set(TARGET_BUILD_PLATFORM "linux") # has to match the TARGET_BUILD_PLATFORM in $ENV{PHYSX_PATH}/physix/buildtools/cmake_generate_projects.py
set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu" CACHE INTERNAL "")
set(PLATFORM "Linux")
set(PX_BUILDSNIPPETS OFF CACHE BOOL "Generate the snippets")
set(PX_BUILDPUBLICSAMPLES OFF CACHE BOOL "Generate the samples projects")
set(PX_GENERATE_STATIC_LIBRARIES ON CACHE BOOL "Generate static libraries")
set(PX_FLOAT_POINT_PRECISE_MATH OFF CACHE BOOL "Float point precise math")
set(NV_USE_STATIC_WINCRT ON CACHE BOOL "Use the statically linked windows CRT")
set(NV_USE_DEBUG_WINCRT ON CACHE BOOL "Use the debug version of the CRT")
set(PXSHARED_PATH ${PHYSX_PATH}/../pxshared)
set(PXSHARED_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CMAKEMODULES_VERSION "1.27")
set(CMAKEMODULES_PATH ${PHYSX_PATH}/../externals/cmakemodules)
set(PX_OUTPUT_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/externals/physx)
set(PX_OUTPUT_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/externals/physx)
add_subdirectory(${PHYSX_PATH}/compiler/public externals/physx)
find_package(Threads REQUIRED)
include_directories(PRIVATE ${PHYSX_ROOT_DIR}/include)


TARGET_LINK_LIBRARIES(
    general_general
    ${PHYSX_LIBRARIES}
    Threads::Threads
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)
