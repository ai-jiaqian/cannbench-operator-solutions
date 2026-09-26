# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# GRU operator self-registration

# Kernel source files (compiled with bisheng)
set(GRU_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gru_kernel.cpp
)

# Plugin source files (compiled with g++)
set(GRU_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/gru_plugin.cpp
)

register_direct_launch_op(
    "${GRU_KERNEL_SRCS}"
    op_kernel
    "${GRU_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
