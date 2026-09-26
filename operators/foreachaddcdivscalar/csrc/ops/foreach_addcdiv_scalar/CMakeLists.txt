# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ForeachAddcdivScalar 算子自注册

# Kernel源文件(bisheng编译)
set(FOREACH_ADDCDIV_SCALAR_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/foreach_addcdiv_scalar_kernel.cpp
)

# Plugin源文件(g++编译)
set(FOREACH_ADDCDIV_SCALAR_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/foreach_addcdiv_scalar_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${FOREACH_ADDCDIV_SCALAR_KERNEL_SRCS}"
    op_kernel
    "${FOREACH_ADDCDIV_SCALAR_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
