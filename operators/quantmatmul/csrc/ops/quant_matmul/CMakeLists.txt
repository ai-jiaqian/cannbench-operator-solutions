# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# QuantMatmul算子自注册

# Kernel源文件(bisheng编译)
# 必须按 kernel task type 拆分成两个翻译单元：构建工具按文件取 kernel 类型标记，
# AIC 纯 Cube kernel 与 AIV epilogue kernel 放在同一个 .cpp 里会导致
# 除第一个之外的 kernel 被静默注册成错误的任务类型（无处写出结果）。
set(QUANT_MATMUL_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/quant_matmul_cube_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/quant_matmul_epilogue_kernel.cpp
)

# Plugin源文件(g++编译)
set(QUANT_MATMUL_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/quant_matmul_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${QUANT_MATMUL_KERNEL_SRCS}"
    op_kernel
    "${QUANT_MATMUL_PLUGIN_SRCS}"
    op_kernel
    "--npu-arch=dav-2201"
)
