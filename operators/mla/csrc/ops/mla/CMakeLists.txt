# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# MLA算子自注册

# Kernel源文件(bisheng编译)
set(MLA_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mla_cube.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mla_tiling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mla_softmax.cpp
)

# Plugin源文件(g++编译)
set(MLA_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/mla_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${MLA_KERNEL_SRCS}"
    op_kernel
    "${MLA_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
