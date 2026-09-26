# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# GroupedMatmul算子自注册

# Kernel源文件(bisheng编译)
set(GROUPED_MATMUL_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/grouped_matmul_kernel.cpp
)

# Plugin源文件(g++编译)
set(GROUPED_MATMUL_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/grouped_matmul_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${GROUPED_MATMUL_KERNEL_SRCS}"
    op_kernel
    "${GROUPED_MATMUL_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
