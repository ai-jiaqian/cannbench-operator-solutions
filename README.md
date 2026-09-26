# CANNBench 官网最优算子源码

简体中文 | [English](README.en.md)

本仓库公开 `jiaqian` 在 CANNBench 方案中**当前选中的公开评测组件源码**。每个算子对应一份源码快照和一个官网 Job，可在 [算子目录](CATALOG.md)查看当前 53 个算子的来源与成绩。这里的“最优”指**我们这套方案当前选用的组件**，不是全站所有参赛者的最高成绩。

## 仓库结构

- `operators/<operator_key>/csrc/ops/<source_dir>/`：从官网提交 ZIP 中提取的、当前选中算子的原始源码文件。旁边的 `provenance.json` 记录文件 SHA-256、来源 Job、官网分数、通过用例数和评测集版本。
- `jobs/<job_id>/`：对应 Job 提交包中的公共构建文件和 Python 入口。`provenance.json` 记录原始 ZIP 的 SHA-256 及公开文件的哈希。
- `results/current.json`：当前公开组件清单及获取时间。
- `scripts/sync_official.py`：读取官网当前组件；只有组件发生变化时才更新仓库。账号令牌从本机 macOS 钥匙串读取，不写入仓库。
- `scripts/package_operator.py`：将一个算子与其来源 Job 的构建文件组装为单算子源码目录，供本地检查或构建。过滤后的目录与原始多算子提交 ZIP 并非字节一致。

## 校验与使用

检出仓库后，运行 `python3 scripts/verify.py` 核对公开源码的哈希。组装单个算子的示例：

```bash
python3 scripts/package_operator.py exp /tmp/cannbench-exp-source
```

构建时需要与来源 Job 兼容的昇腾 CANN、PyTorch 和 `torch_npu` 环境。[CANNBench 评测指南](https://gitcode.com/cann/cann-bench/blob/master/docs/guide/quick_start.md)说明了源码构建和单算子评测方法。若将过滤后的源码包提交到官网，应明确选择目标算子；保留的 Python 入口可能还定义了原始多算子 Job 中其他算子的包装函数。

目录中的官网分数属于 `provenance.json` 指向的**原始 Job**。重新组装或修改源码会形成新的候选，必须重新评测，才能声称取得相同成绩。仓库的校验脚本只检查文件和元数据，不运行 NPU 评测。

## 许可与来源

CANNBench 构建文件及许多提交源码保留了华为版权和 CANN Open Software License Agreement Version 2.0 声明。[LICENSE](LICENSE) 是该许可的全文：它将用途限制在华为 AI 处理器相关软件，并要求再分发时保留声明和许可。本仓库按这些条款公开源码，未将华为来源文件改授 Apache 许可，也不宣称该许可经过 OSI 认证。算子选择元数据和本仓脚本采用相同的仓库许可。源码来源见 [NOTICE](NOTICE)。

本仓库不包含账号令牌、私有评测用例、模型交互、实验日志或编译产物。
