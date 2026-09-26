# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# MlaProlog 算子自注册

# Kernel源文件(bisheng编译)
set(MLA_PROLOG_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mla_prolog_tiling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mla_prolog_mm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/mla_prolog_vec.cpp
)

# Plugin源文件(g++编译)
set(MLA_PROLOG_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/mla_prolog_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${MLA_PROLOG_KERNEL_SRCS}"
    op_kernel
    "${MLA_PROLOG_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
