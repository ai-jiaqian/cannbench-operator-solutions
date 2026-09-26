# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# SwiGlu算子自注册

# Kernel源文件(bisheng编译)
set(SWI_GLU_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/swi_glu_kernel.cpp
)

# Plugin源文件(g++编译)
set(SWI_GLU_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/swi_glu_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${SWI_GLU_KERNEL_SRCS}"
    op_kernel
    "${SWI_GLU_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
