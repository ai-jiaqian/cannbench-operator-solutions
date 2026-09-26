# CANNBench 官网自进化Agent最优算子源码

简体中文 | [English](README.en.md)

本仓库将持续更新并公开 `jiaqian` 的自进化Agent在 CANNBench Leaderboard中 **当前提交的公开评测组件源码**。每个算子对应一份源码快照和一个官网 Job，这些算子的实现、测试、提交与维护均由Agent自主完成，本仓库为其提交记录的定时同步，可在 [算子目录](CATALOG.md)查看最新版 53 个算子的来源与成绩。这里的“最优”指**我们这套方案当前选用的组件**，不是全站所有参赛者的最高成绩。

## 仓库结构

- `operators/<operator_key>/csrc/ops/<source_dir>/`：从官网提交 ZIP 中提取的、当前选中算子的原始源码文件。旁边的 `provenance.json` 记录文件 SHA-256、来源 Job、官网分数、通过用例数和评测集版本。
- `results/current.json`：当前公开组件清单及获取时间。
- `scripts/sync_official.py`：读取官网当前组件；只有组件发生变化时才更新仓库。账号令牌从本机 macOS 钥匙串读取，不写入仓库。

## 校验与使用

检出仓库后，运行 `python3 scripts/verify.py` 核对算子源码的哈希。构建或评测时，可按 [CANNBench 评测指南](https://gitcode.com/cann/cann-bench/blob/master/docs/guide/quick_start.md)将源码接入兼容的提交工程。目录中的官网成绩对应 `provenance.json` 指向的原始 Job。

## 许可与来源

许多提交源码保留了华为版权和 CANN Open Software License Agreement Version 2.0 声明。[LICENSE](LICENSE) 是该许可的全文：它将用途限制在华为 AI 处理器相关软件，并要求再分发时保留声明和许可。本仓库按这些条款公开源码，未将华为来源文件改授 Apache 许可，也不宣称该许可经过 OSI 认证。算子选择元数据和本仓脚本采用相同的仓库许可。源码来源见 [NOTICE](NOTICE)。

本仓库不包含账号令牌、私有评测用例、模型交互、实验日志或编译产物。
