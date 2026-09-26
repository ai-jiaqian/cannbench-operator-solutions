# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# AdaptiveAvgPool3D算子自注册

# Kernel源文件(bisheng编译)
set(AAP3D_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/adaptive_avg_pool_3d_kernel.cpp
)

# Plugin源文件(g++编译)
set(AAP3D_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/adaptive_avg_pool_3d_plugin.cpp
)

register_direct_launch_op(
    "${AAP3D_KERNEL_SRCS}"
    op_kernel
    "${AAP3D_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
