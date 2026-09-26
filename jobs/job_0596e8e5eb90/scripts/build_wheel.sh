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
# Build Python wheel package for direct_launch_example

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="${PROJECT_DIR}/dist"

echo "=== Building wheel package ==="

# Clean previous builds
rm -rf "${PROJECT_DIR}/build" "${PROJECT_DIR}/*.egg-info"

# Ensure dist directory exists
mkdir -p "${DIST_DIR}"

# Build wheel
cd "${PROJECT_DIR}"
python3 setup.py clean
python3 -m build --wheel --no-isolation --outdir "${DIST_DIR}"

# Find and display wheel
WHEEL_FILE=$(find "${DIST_DIR}" -name "*.whl" -type f | head -1)
if [[ -z "${WHEEL_FILE}" ]]; then
    echo "ERROR: No wheel package found"
    exit 1
fi

echo "=== Wheel package built successfully ==="
echo "Output: ${WHEEL_FILE}"