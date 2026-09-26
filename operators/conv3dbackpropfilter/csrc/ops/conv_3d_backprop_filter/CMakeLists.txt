# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Conv3DBackpropFilter算子自注册

# Kernel源文件(bisheng编译)
set(CONV3D_BPF_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/conv_3d_bpf_kernel.cpp
)

# Plugin源文件(g++编译)
set(CONV3D_BPF_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/conv_3d_bpf_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${CONV3D_BPF_KERNEL_SRCS}"
    op_kernel
    "${CONV3D_BPF_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
