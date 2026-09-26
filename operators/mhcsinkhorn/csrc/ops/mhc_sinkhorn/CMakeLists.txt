# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# MhcSinkhorn算子自注册

# Kernel源文件(bisheng编译)
set(MHC_SINKHORN_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mhc_sinkhorn_kernel.cpp
)

# Plugin源文件(g++编译)
set(MHC_SINKHORN_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/mhc_sinkhorn_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${MHC_SINKHORN_KERNEL_SRCS}"
    op_kernel
    "${MHC_SINKHORN_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
