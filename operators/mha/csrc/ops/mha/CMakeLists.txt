# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# MHA算子自注册

# Kernel源文件(bisheng编译)
set(MHA_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mha_kernel.cpp
)

# Plugin源文件(g++编译)
set(MHA_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/mha_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${MHA_KERNEL_SRCS}"
    op_kernel
    "${MHA_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
