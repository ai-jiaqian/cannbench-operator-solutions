#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# Build wheel package and optionally install

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

NPU_ARCH="ascend910b"
INSTALL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --soc=*)
            NPU_ARCH="${1#*=}"
            shift
            ;;
        --install)
            INSTALL=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "=== Building cann_bench wheel package ==="
echo "NPU_ARCH: ${NPU_ARCH}"

# Clean dist directory
DIST_DIR="${SCRIPT_DIR}/dist"
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

# Build wheel
bash "${SCRIPT_DIR}/scripts/build_wheel.sh"

if [[ "${INSTALL}" == "true" ]]; then
    echo "=== Installing wheel package ==="
    pip install ${DIST_DIR}/cann_bench*.whl --force-reinstall --no-deps
fi

echo ""
echo "=== Build complete ==="
ls -la "${DIST_DIR}"