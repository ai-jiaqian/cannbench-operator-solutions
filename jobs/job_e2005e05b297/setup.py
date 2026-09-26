#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import shutil
import subprocess
import logging
from setuptools import setup, find_packages, Distribution, Command
from wheel.bdist_wheel import bdist_wheel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
PACKAGE_NAME = "cann_bench"
VERSION = "1.0.0"
DESCRIPTION = "Direct launch Ascend operators with Python bindings"


class CleanCommand(Command):
    """usage: python setup.py clean"""
    description = "Clean build artifacts"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        folders = ['build', f'{PACKAGE_NAME}.egg-info']
        for f in folders:
            if os.path.exists(f):
                shutil.rmtree(f)
                logging.info(f"Removed: {f}")
        # Clean wheel in dist
        if os.path.exists('dist'):
            for f in os.listdir('dist'):
                if f.endswith('.whl'):
                    os.remove(os.path.join('dist', f))
        logging.info("Cleaned.")


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False

    def has_ext_modules(self):
        return True


class ABI3Wheel(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()
        python = "cp38"
        abi = "abi3"
        plat = "linux_aarch64"
        return python, abi, plat

    def run(self):
        self.run_command('cmake_build')
        super().run()


class CMakeBuildCommand(Command):
    description = "Build CMake extensions"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import torch
        import torch_npu

        cpu_count = os.cpu_count() or 2
        Torch_DIR = os.path.join(torch.utils.cmake_prefix_path, "Torch")
        TORCH_NPU_PATH = os.path.dirname(torch_npu.__file__)
        NPU_ARCH = os.environ.get('NPU_ARCH', 'ascend910b')

        logging.info(f"Torch: {Torch_DIR}")
        logging.info(f"Torch NPU: {TORCH_NPU_PATH}")
        logging.info(f"NPU_ARCH: {NPU_ARCH}")

        build_dir = os.path.join(os.getcwd(), 'build')
        subprocess.check_call([
            'cmake', '-S', os.getcwd(), '-B', build_dir,
            '-DCMAKE_BUILD_TYPE=Release',
            f'-DTorch_DIR={Torch_DIR}',
            f'-DTORCH_NPU_PATH={TORCH_NPU_PATH}',
            f'-DNPU_ARCH={NPU_ARCH}'
        ])
        subprocess.check_call([
            'cmake', '--build', build_dir,
            '--config', 'Release', '--parallel', str(cpu_count)
        ])
        logging.info("CMake build complete.")


cmdclass = {
    'clean': CleanCommand,
    'bdist_wheel': ABI3Wheel,
    'cmake_build': CMakeBuildCommand,
}

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    package_data={PACKAGE_NAME: ['*.abi3.so']},
    distclass=BinaryDistribution,
    cmdclass=cmdclass,
    zip_safe=False,
    install_requires=["torch", "torch_npu"],
    python_requires='>=3.8',
)