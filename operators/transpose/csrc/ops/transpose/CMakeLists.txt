# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Transpose算子自注册

# Kernel源文件(bisheng编译)
set(TRANSPOSE_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/transpose_kernel.cpp
)

# Plugin源文件(g++编译)
set(TRANSPOSE_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/transpose_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${TRANSPOSE_KERNEL_SRCS}"
    op_kernel
    "${TRANSPOSE_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
