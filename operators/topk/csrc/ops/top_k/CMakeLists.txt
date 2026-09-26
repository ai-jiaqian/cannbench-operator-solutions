# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# TopK算子自注册

# Kernel源文件(bisheng编译)
set(TOPK_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/topk_kernel.cpp
)

# Plugin源文件(g++编译)
set(TOPK_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/topk_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${TOPK_KERNEL_SRCS}"
    op_kernel
    "${TOPK_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
