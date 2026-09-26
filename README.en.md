# CANNBench official best operator sources

[简体中文](README.md) | English

This repository publishes the source of the **currently selected public-case components** in the `jiaqian` CANNBench solution. It is an operator-source catalog, with one exact source snapshot and official Job reference per operator. See [CATALOG.md](CATALOG.md) for the current 53 entries.

## Layout

- `operators/<operator_key>/csrc/ops/<source_dir>/`: exact selected operator source files from the official submission ZIP. `provenance.json` records their SHA-256 hashes, source Job, official score, case count, and benchmark version.
- `results/current.json`: current public solution selection and retrieval timestamp.
- `scripts/sync_official.py`: fetches the current component list and updates only when the official selection changes. It reads the account token from the local macOS Keychain and never saves it in this repository.

## Build and verification

Run `python3 scripts/verify.py` after checkout to verify the operator source hashes. For building or evaluation, use the source in a compatible submission project following the [CANNBench evaluation guide](https://gitcode.com/cann/cann-bench/blob/master/docs/guide/quick_start.md). The catalog's official result belongs to the original Job identified in `provenance.json`.

## License and origin

Many submitted source files retain Huawei copyright and CANN Open Software License Agreement Version 2.0 notices. [LICENSE](LICENSE) contains that agreement; it restricts use to software for Huawei AI Processors and requires retaining notices and the agreement when redistributing. This repository makes the source public under those terms. It does not relicense Huawei-origin files under Apache or claim OSI-approved licensing. The operator selection metadata and repository scripts are contributed under the same repository license. See [NOTICE](NOTICE) for source provenance.

No tokens, private benchmark cases, model traffic, experiment logs, or compiled artifacts are part of this repository.
