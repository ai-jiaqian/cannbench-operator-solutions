# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# NMS算子自注册

# Kernel源文件(bisheng编译)
set(NMS_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/nms_kernel.cpp
)

# Plugin源文件(g++编译)
set(NMS_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/nms_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${NMS_KERNEL_SRCS}"
    op_kernel
    "${NMS_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
