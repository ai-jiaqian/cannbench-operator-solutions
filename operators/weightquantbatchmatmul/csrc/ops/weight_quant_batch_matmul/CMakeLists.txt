# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# WeightQuantBatchMatmul算子自注册

# Kernel源文件(bisheng编译)
set(WQMM_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/wqmm_host.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/wqmm_dequant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/wqmm_mm.cpp
)

# Plugin源文件(g++编译)
set(WQMM_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/wqmm_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${WQMM_KERNEL_SRCS}"
    op_kernel
    "${WQMM_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
