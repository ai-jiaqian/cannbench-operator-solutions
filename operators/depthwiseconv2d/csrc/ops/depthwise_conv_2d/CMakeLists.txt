# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# DepthwiseConv2D算子自注册

# Kernel源文件(bisheng编译)
set(DEPTHWISE_CONV_2D_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/depthwise_conv_2d_kernel.cpp
)

# Plugin源文件(g++编译)
set(DEPTHWISE_CONV_2D_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/depthwise_conv_2d_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${DEPTHWISE_CONV_2D_KERNEL_SRCS}"
    op_kernel
    "${DEPTHWISE_CONV_2D_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
