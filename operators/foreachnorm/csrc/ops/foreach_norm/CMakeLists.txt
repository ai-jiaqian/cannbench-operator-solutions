# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ForeachNorm算子自注册

# Kernel源文件(bisheng编译)
set(FOREACH_NORM_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/foreach_norm_kernel.cpp
)

# Plugin源文件(g++编译)
set(FOREACH_NORM_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/foreach_norm_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${FOREACH_NORM_KERNEL_SRCS}"
    op_kernel
    "${FOREACH_NORM_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
