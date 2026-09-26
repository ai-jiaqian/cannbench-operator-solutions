# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ROIAlign算子自注册

# Kernel源文件(bisheng编译)
set(ROI_ALIGN_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/roi_align_kernel.cpp
)

# Plugin源文件(g++编译)
set(ROI_ALIGN_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/roi_align_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${ROI_ALIGN_KERNEL_SRCS}"
    op_kernel
    "${ROI_ALIGN_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
