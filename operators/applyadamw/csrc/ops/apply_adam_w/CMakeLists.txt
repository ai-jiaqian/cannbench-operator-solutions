# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ApplyAdamW算子自注册

# Kernel源文件(bisheng编译)
set(APPLY_ADAM_W_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/apply_adam_w_kernel.cpp
)

# Plugin源文件(g++编译)
set(APPLY_ADAM_W_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/apply_adam_w_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${APPLY_ADAM_W_KERNEL_SRCS}"
    op_kernel
    "${APPLY_ADAM_W_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
