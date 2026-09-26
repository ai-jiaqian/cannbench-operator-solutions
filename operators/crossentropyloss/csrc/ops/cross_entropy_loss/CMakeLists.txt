# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# CrossEntropyLoss算子自注册

# Kernel源文件(bisheng编译)
set(CE_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/ce_kernel.cpp
)

# Plugin源文件(g++编译)
set(CE_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/ce_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${CE_KERNEL_SRCS}"
    op_kernel
    "${CE_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
