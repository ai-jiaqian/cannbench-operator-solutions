# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# StridedSlice算子自注册

# Kernel源文件(bisheng编译)
set(STRIDED_SLICE_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/strided_slice_kernel.cpp
)

# Plugin源文件(g++编译)
set(STRIDED_SLICE_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/strided_slice_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${STRIDED_SLICE_KERNEL_SRCS}"
    op_kernel
    "${STRIDED_SLICE_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
