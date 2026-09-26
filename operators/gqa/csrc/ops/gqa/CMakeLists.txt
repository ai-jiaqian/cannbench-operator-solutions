# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# GQA算子自注册

# Kernel源文件(bisheng编译)
set(GQA_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gqa_pack_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gqa_qk_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gqa_softmax_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gqa_softmax_flat_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gqa_pv_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gqa_unpack_kernel.cpp
)

# Plugin源文件(g++编译)
set(GQA_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/gqa_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${GQA_KERNEL_SRCS}"
    op_kernel
    "${GQA_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
