# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# EngramGateFusion算子自注册

# Kernel源文件(bisheng编译)
set(ENGRAM_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/engram_kernel.cpp
)

# Plugin源文件(g++编译)
set(ENGRAM_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/engram_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${ENGRAM_KERNEL_SRCS}"
    op_kernel
    "${ENGRAM_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
