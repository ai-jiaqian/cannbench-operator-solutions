# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------

# Direct launch算子注册宏
# 算子目录通过调用register_direct_launch_op()将自己注册到全局列表

# 全局变量定义
set(ALL_KERNEL_SRCS "" CACHE INTERNAL "All kernel source files")
set(ALL_KERNEL_INCLUDE_DIRS "" CACHE INTERNAL "All kernel include directories")
set(ALL_PLUGIN_SRCS "" CACHE INTERNAL "All plugin source files")
set(ALL_PLUGIN_INCLUDE_DIRS "" CACHE INTERNAL "All plugin include directories")

# 注册Direct Launch算子
# 参数:
#   KERNEL_SRCS: kernel源文件列表(bisheng编译)
#   KERNEL_INCLUDE_DIR: kernel需要的include目录
#   PLUGIN_SRCS: plugin源文件列表(g++编译)
#   PLUGIN_INCLUDE_DIR: plugin需要的include目录
#   KERNEL_ARGS: bisheng编译参数(如--npu-arch=dav-2201)
macro(register_direct_launch_op KERNEL_SRCS KERNEL_INCLUDE_DIR PLUGIN_SRCS PLUGIN_INCLUDE_DIR KERNEL_ARGS)
    get_filename_component(OP_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    message(STATUS "Registering direct launch op: ${OP_NAME}")

    # 添加kernel源文件到全局列表
    set(_TEMP_KERNEL ${ALL_KERNEL_SRCS})
    list(APPEND _TEMP_KERNEL ${KERNEL_SRCS})
    set(ALL_KERNEL_SRCS ${_TEMP_KERNEL} CACHE INTERNAL "All kernel source files")

    # 添加kernel include目录
    set(_TEMP_KERNEL_INC ${ALL_KERNEL_INCLUDE_DIRS})
    list(APPEND _TEMP_KERNEL_INC ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_INCLUDE_DIR})
    set(ALL_KERNEL_INCLUDE_DIRS ${_TEMP_KERNEL_INC} CACHE INTERNAL "All kernel include directories")

    # 添加plugin源文件到全局列表
    set(_TEMP_PLUGIN ${ALL_PLUGIN_SRCS})
    list(APPEND _TEMP_PLUGIN ${PLUGIN_SRCS})
    set(ALL_PLUGIN_SRCS ${_TEMP_PLUGIN} CACHE INTERNAL "All plugin source files")

    # 添加plugin include目录
    set(_TEMP_PLUGIN_INC ${ALL_PLUGIN_INCLUDE_DIRS})
    list(APPEND _TEMP_PLUGIN_INC ${CMAKE_CURRENT_SOURCE_DIR}/${PLUGIN_INCLUDE_DIR})
    set(ALL_PLUGIN_INCLUDE_DIRS ${_TEMP_PLUGIN_INC} CACHE INTERNAL "All plugin include directories")

    # 存储编译参数
    set(_KERNEL_ARGS_LIST ${ALL_KERNEL_ARGS_LIST})
    list(APPEND _KERNEL_ARGS_LIST "${OP_NAME}|${KERNEL_ARGS}")
    set(ALL_KERNEL_ARGS_LIST ${_KERNEL_ARGS_LIST} CACHE INTERNAL "All kernel args")

    message(STATUS "Registered ${OP_NAME}: kernel=${KERNEL_SRCS}, plugin=${PLUGIN_SRCS}")
endmacro()