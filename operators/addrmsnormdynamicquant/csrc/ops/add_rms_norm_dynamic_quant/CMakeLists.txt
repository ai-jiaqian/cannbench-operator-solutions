# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# AddRmsNormDynamicQuant 算子自注册

# Kernel源文件(bisheng编译)
set(ARDQ_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/add_rms_norm_dynamic_quant_kernel.cpp
)

# Plugin源文件(g++编译)
set(ARDQ_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/add_rms_norm_dynamic_quant_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${ARDQ_KERNEL_SRCS}"
    op_kernel
    "${ARDQ_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
