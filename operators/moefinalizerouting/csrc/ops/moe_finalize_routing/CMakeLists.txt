# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# MoeFinalizeRouting算子自注册

# Kernel源文件(bisheng编译)
set(MOE_FINALIZE_ROUTING_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/moe_finalize_routing_kernel.cpp
)

# Plugin源文件(g++编译)
set(MOE_FINALIZE_ROUTING_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/moe_finalize_routing_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${MOE_FINALIZE_ROUTING_KERNEL_SRCS}"
    op_kernel
    "${MOE_FINALIZE_ROUTING_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
