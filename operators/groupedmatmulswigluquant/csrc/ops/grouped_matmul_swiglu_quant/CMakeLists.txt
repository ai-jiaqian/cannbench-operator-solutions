# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# GroupedMatmulSwigluQuant 算子自注册

# Kernel源文件(bisheng编译)：分组矩阵乘 + (反量化/SwiGLU/再量化)
set(GROUPED_MATMUL_SWIGLU_QUANT_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gmsq_cube_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gmsq_vec_kernel.cpp
)

# Plugin源文件(g++编译)
set(GROUPED_MATMUL_SWIGLU_QUANT_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/gmsq_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${GROUPED_MATMUL_SWIGLU_QUANT_KERNEL_SRCS}"
    op_kernel
    "${GROUPED_MATMUL_SWIGLU_QUANT_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
