# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Scatter算子自注册

# Kernel源文件(bisheng编译)
set(SCATTER_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/scatter_kernel.cpp
)

# Plugin源文件(g++编译)
set(SCATTER_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/scatter_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${SCATTER_KERNEL_SRCS}"
    op_kernel
    "${SCATTER_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
