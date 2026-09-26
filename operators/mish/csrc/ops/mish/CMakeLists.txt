# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Mish算子自注册

# Kernel源文件(bisheng编译)
set(MISH_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mish_kernel.cpp
)

# Plugin源文件(g++编译)
set(MISH_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/mish_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${MISH_KERNEL_SRCS}"
    op_kernel
    "${MISH_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
