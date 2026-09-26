# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Dilation2D算子自注册

# Kernel源文件(bisheng编译)
set(DILATION_2D_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/dilation_2d_kernel.cpp
)

# Plugin源文件(g++编译)
set(DILATION_2D_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/dilation_2d_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${DILATION_2D_KERNEL_SRCS}"
    op_kernel
    "${DILATION_2D_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
