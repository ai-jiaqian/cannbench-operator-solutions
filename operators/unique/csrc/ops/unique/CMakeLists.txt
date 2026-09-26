# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Unique算子自注册

# Kernel源文件(bisheng编译)
set(UNIQUE_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/unique_kernel.cpp
)

# Plugin源文件(g++编译)
set(UNIQUE_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/unique_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${UNIQUE_KERNEL_SRCS}"
    op_kernel
    "${UNIQUE_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
