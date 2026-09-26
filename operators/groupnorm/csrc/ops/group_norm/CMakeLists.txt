# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# GroupNorm算子自注册

# Kernel源文件(bisheng编译)
set(GROUP_NORM_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/group_norm_kernel.cpp
)

# Plugin源文件(g++编译)
set(GROUP_NORM_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/group_norm_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${GROUP_NORM_KERNEL_SRCS}"
    op_kernel
    "${GROUP_NORM_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
