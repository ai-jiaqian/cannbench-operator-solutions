# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ResizeBilinear算子自注册

# Kernel源文件(bisheng编译)
set(RESIZE_BILINEAR_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/resize_bilinear_kernel.cpp
)

# Plugin源文件(g++编译)
set(RESIZE_BILINEAR_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/resize_bilinear_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${RESIZE_BILINEAR_KERNEL_SRCS}"
    op_kernel
    "${RESIZE_BILINEAR_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
