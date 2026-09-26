# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
set(PKG_NAME cann_bench)
project(${PKG_NAME} VERSION 1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/ascend.cmake)
include(cmake/python.cmake)
include(cmake/torch.cmake)
include(cmake/torch_npu.cmake)
include(cmake/func.cmake)

# NPU architecture
set(NPU_ARCH "ascend910b" CACHE STRING "NPU architecture")
set(EXTENSION_MODULE_NAME "cann_bench" CACHE STRING "Extension module name")

# 输出目录
set(DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist)

# include directories
set(INCLUDE_DIRECTORIES
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
    ${ASCEND_INCLUDE_DIRS}
)

# link directories
set(LINK_DIRECTORIES
    ${TORCH_NPU_LIB_PATH}
    ${ASCEND_DIR}/lib64
)

# link libraries
set(LINK_LIBRARIES
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    runtime
)

# compile options
set(COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
    -O3
    -fdiagnostics-color=always
    -w
    -DEXTENSION_MODULE_NAME=${EXTENSION_MODULE_NAME}
)

# 初始化全局变量
set(ALL_KERNEL_SRCS "" CACHE INTERNAL "All kernel source files")
set(ALL_KERNEL_INCLUDE_DIRS "" CACHE INTERNAL "All kernel include directories")
set(ALL_PLUGIN_SRCS "" CACHE INTERNAL "All plugin source files")
set(ALL_PLUGIN_INCLUDE_DIRS "" CACHE INTERNAL "All plugin include directories")
set(ALL_KERNEL_ARGS_LIST "" CACHE INTERNAL "All kernel args")

# 添加算子目录(自注册)
add_subdirectory(csrc)

message(STATUS "Registered kernel sources: ${ALL_KERNEL_SRCS}")
message(STATUS "Registered kernel include dirs: ${ALL_KERNEL_INCLUDE_DIRS}")
message(STATUS "Registered plugin sources: ${ALL_PLUGIN_SRCS}")
message(STATUS "Registered plugin include dirs: ${ALL_PLUGIN_INCLUDE_DIRS}")

# ========== Kernel编译(bisheng) ==========
set(_SAVED_CMAKE_C_COMPILER ${CMAKE_C_COMPILER})
set(_SAVED_CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER})
set(_SAVED_CMAKE_LINKER ${CMAKE_LINKER})
set(_SAVED_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})
unset(CMAKE_CXX_FLAGS)

set(KERNEL_COMPILE_FLAGS "--npu-arch=dav-2201 -xasc")
set_source_files_properties(${ALL_KERNEL_SRCS} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "${KERNEL_COMPILE_FLAGS}")

set(KERNEL_TARGET all_kernels_obj)
add_library(${KERNEL_TARGET} OBJECT ${ALL_KERNEL_SRCS})
target_include_directories(${KERNEL_TARGET} PRIVATE ${ALL_KERNEL_INCLUDE_DIRS} ${INCLUDE_DIRECTORIES})

set(CMAKE_C_COMPILER ${_SAVED_CMAKE_C_COMPILER})
set(CMAKE_CXX_COMPILER ${_SAVED_CMAKE_CXX_COMPILER})
set(CMAKE_LINKER ${_SAVED_CMAKE_LINKER})
set(CMAKE_CXX_FLAGS ${_SAVED_CMAKE_CXX_FLAGS})

# ========== Plugin编译(g++) ==========
set(PLUGIN_TARGET all_plugins_obj)
add_library(${PLUGIN_TARGET} OBJECT ${ALL_PLUGIN_SRCS})
target_compile_options(${PLUGIN_TARGET} PRIVATE ${COMPILE_OPTIONS})
target_include_directories(${PLUGIN_TARGET} PRIVATE ${ALL_PLUGIN_INCLUDE_DIRS} ${INCLUDE_DIRECTORIES})

# ========== 创建共享库 ==========
set(EXTENSION_CPP ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extension.cpp)
add_library(_C SHARED
    ${EXTENSION_CPP}
    $<TARGET_OBJECTS:${KERNEL_TARGET}>
    $<TARGET_OBJECTS:${PLUGIN_TARGET}>
)
set_target_properties(_C PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    SUFFIX ".abi3.so"
    OUTPUT_NAME "_C"
)
target_compile_definitions(_C PRIVATE Py_LIMITED_API=0x03080000)
target_compile_options(_C PRIVATE ${COMPILE_OPTIONS})
target_include_directories(_C PRIVATE ${INCLUDE_DIRECTORIES})
target_link_directories(_C PRIVATE ${LINK_DIRECTORIES})
target_link_libraries(_C PRIVATE ${LINK_LIBRARIES})

add_custom_command(TARGET _C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:_C>
    ${CMAKE_CURRENT_SOURCE_DIR}/${EXTENSION_MODULE_NAME}/$<TARGET_FILE_NAME:_C>
    COMMENT "Copying compiled extension to ${EXTENSION_MODULE_NAME}/"
)