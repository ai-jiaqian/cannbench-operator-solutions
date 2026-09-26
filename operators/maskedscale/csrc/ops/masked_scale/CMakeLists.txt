# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# MaskedScale算子自注册

# Kernel源文件(bisheng编译)
set(MASKED_SCALE_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/masked_scale_kernel.cpp
)

# Plugin源文件(g++编译)
set(MASKED_SCALE_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/masked_scale_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${MASKED_SCALE_KERNEL_SRCS}"
    op_kernel
    "${MASKED_SCALE_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
