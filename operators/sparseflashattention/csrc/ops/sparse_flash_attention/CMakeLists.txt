# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# SparseFlashAttention算子自注册

# Kernel源文件(bisheng编译)
set(SFA_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/sparse_flash_attention_kernel.cpp
)

# Plugin源文件(g++编译)
set(SFA_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/sparse_flash_attention_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${SFA_KERNEL_SRCS}"
    op_kernel
    "${SFA_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
