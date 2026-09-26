# CANNBench official best operator sources

[简体中文](README.md) | English

This repository publishes the source of the **currently selected public-case components** in the `jiaqian` CANNBench solution. It is an operator-source catalog, with one exact source snapshot and official Job reference per operator. See [CATALOG.md](CATALOG.md) for the current 53 entries.

## Layout

- `operators/<operator_key>/csrc/ops/<source_dir>/`: exact selected operator source files from the official submission ZIP. `provenance.json` records their SHA-256 hashes, source Job, official score, case count, and benchmark version.
- `jobs/<job_id>/`: the common build scaffold and Python entrypoint from that Job's submitted ZIP. `provenance.json` records the original ZIP SHA-256 and each published file hash.
- `results/current.json`: current public solution selection and retrieval timestamp.
- `scripts/sync_official.py`: fetches the current component list and updates only when the official selection changes. It reads the account token from the local macOS Keychain and never saves it in this repository.
- `scripts/package_operator.py`: assembles one operator and its original Job scaffold into a source directory for local inspection/build. This filtered package is not byte-identical to the full original multi-operator submission ZIP.

## Build and verification

Run `python3 scripts/verify.py` after checkout to verify the published source hashes. To assemble one operator, run `python3 scripts/package_operator.py exp /tmp/cannbench-exp-source`. Build in an Ascend CANN/PyTorch/torch_npu environment compatible with the selected Job. CANNBench's [evaluation guide](https://gitcode.com/cann/cann-bench/blob/master/docs/guide/quick_start.md) documents the source build and per-operator evaluation workflow. When using a filtered package on the website, explicitly select the one intended operator; the preserved Python entrypoint may also define wrappers for other operators from the original multi-operator Job.

The official score belongs to the original Job, identified in `provenance.json`. Reassembling or editing a package creates a new candidate and requires a new evaluation before claiming the same result. The repository's verification script checks bytes and metadata; it does not run an NPU benchmark.

## License and origin

The CANNBench scaffold and many submitted source files retain Huawei copyright and CANN Open Software License Agreement Version 2.0 notices. [LICENSE](LICENSE) contains that agreement; it restricts use to software for Huawei AI Processors and requires retaining notices and the agreement when redistributing. This repository makes the source public under those terms. It does not relicense Huawei-origin files under Apache or claim OSI-approved licensing. The operator selection metadata and repository scripts are contributed under the same repository license. See [NOTICE](NOTICE) for source provenance.

No tokens, private benchmark cases, model traffic, experiment logs, or compiled artifacts are part of this repository.
