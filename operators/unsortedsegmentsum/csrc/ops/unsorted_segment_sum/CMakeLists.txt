# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# UnsortedSegmentSum算子自注册

# Kernel源文件(bisheng编译)
set(USS_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/unsorted_segment_sum_kernel.cpp
)

# Plugin源文件(g++编译)
set(USS_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/unsorted_segment_sum_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${USS_KERNEL_SRCS}"
    op_kernel
    "${USS_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
