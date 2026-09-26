#!/usr/bin/env python3
"""Refresh the public source catalog from the selected CANNBench solution."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from shutil import rmtree
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://cannbench.com"
SOLUTION_ID = "sol_57a81237c790d4a9"
ACCOUNT = "jiaqian"
EXPECTED_OPERATORS = 53
COMMON_FILES = {
    "CMakeLists.txt",
    "build.sh",
    "requirements.txt",
    "setup.py",
    "cann_bench/__init__.py",
    "csrc/CMakeLists.txt",
    "csrc/extension.cpp",
    "csrc/ops/CMakeLists.txt",
    "scripts/build_wheel.sh",
}


def request(path: str, token: str | None = None) -> bytes:
    headers = {"Cache-Control": "no-cache"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(
        urllib.request.Request(BASE_URL + path, headers=headers), timeout=40
    ) as response:
        return bytes(response.read())


def keychain_token() -> str:
    return subprocess.check_output(
        [
            "security",
            "find-generic-password",
            "-a",
            ACCOUNT,
            "-s",
            "codex.cannbench.token",
            "-w",
        ],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def source_files(
    archive: bytes, operators: list[dict[str, Any]]
) -> tuple[dict[str, bytes], dict[str, dict[str, bytes]]]:
    with zipfile.ZipFile(BytesIO(archive)) as source:
        if source.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
        members = {}
        for info in source.infolist():
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or info.file_size > 1_000_000:
                raise ValueError(f"unsafe source path: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"symlink in source ZIP: {info.filename}")
            members[info.filename] = source.read(info)

    common = {
        name: data
        for name, data in members.items()
        if name in COMMON_FILES
        or (name.startswith("cmake/") and name.endswith(".cmake"))
    }
    if not COMMON_FILES.issubset(common):
        raise ValueError(
            f"missing common files: {sorted(COMMON_FILES - common.keys())}"
        )

    selected = {}
    all_dirs = {
        name.split("/")[2]
        for name in members
        if name.startswith("csrc/ops/") and name.count("/") >= 3
    }
    for op in operators:
        matches = [
            name
            for name in all_dirs
            if normalize(name) == normalize(op["operator_key"])
        ]
        if len(matches) != 1:
            raise ValueError(
                f"operator directory mismatch: {op['operator_key']}: {matches}"
            )
        prefix = f"csrc/ops/{matches[0]}/"
        files = {
            name: data
            for name, data in members.items()
            if name.startswith(prefix)
            and (name.endswith((".cpp", ".h")) or name.endswith("/CMakeLists.txt"))
        }
        if (
            not any(name.endswith(".cpp") for name in files)
            or prefix + "CMakeLists.txt" not in files
        ):
            raise ValueError(f"incomplete operator source: {op['operator_key']}")
        selected[op["operator_key"]] = files
    return common, selected


def selection(components: dict[str, Any]) -> list[dict[str, Any]]:
    if components.get("entry_id") != SOLUTION_ID:
        raise ValueError("unexpected solution identity")
    rows = [
        item for item in components["components"] if item.get("case_set") == "standard"
    ]
    if (
        len(rows) != EXPECTED_OPERATORS
        or len({item["operator_key"] for item in rows}) != EXPECTED_OPERATORS
    ):
        raise ValueError("incomplete or duplicated public operator catalog")
    if any(item.get("display_user") != ACCOUNT for item in rows):
        raise ValueError("component belongs to another account")
    keys = (
        "operator_key",
        "operator_name",
        "benchmark_version",
        "job_id",
        "submission_id",
        "score",
        "avg_speedup",
        "passed_cases",
        "total_cases",
        "updated_at",
    )
    return sorted(
        ({key: item.get(key) for key in keys} for item in rows),
        key=lambda row: row["operator_key"],
    )


def write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def render_catalog(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Current official public components",
        "",
        "Source: [CANNBench solution](https://cannbench.com/leaderboard). Scores and pass counts are official public-case results for the named Job. Benchmark versions are shown per operator.",
        "",
        "| Operator | Version | Public cases | Score | Source Job |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        op = row["operator_key"]
        lines.append(
            f"| [{row['operator_name']}](operators/{op}/) | {row['benchmark_version']} | "
            f"{row['passed_cases']}/{row['total_cases']} | {row['score']:.6f} | "
            f"[{row['job_id']}](https://cannbench.com/jobs/{row['job_id']}) |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--components-json", type=Path, help="use a saved public API response"
    )
    args = parser.parse_args()
    if args.components_json:
        components = json.loads(args.components_json.read_text())
    else:
        components = json.loads(
            request(
                f"/api/public/leaderboard/solution/{SOLUTION_ID}/components?result_scope=default"
            )
        )
    rows = selection(components)
    current_path = ROOT / "results/current.json"
    old = (
        json.loads(current_path.read_text())["operators"]
        if current_path.exists()
        else []
    )
    if rows == old:
        print("No official component change.")
        return

    by_job: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_job.setdefault(row["job_id"], []).append(row)
    token = keychain_token()
    downloaded = {}
    source_sets = {}
    for job, ops in sorted(by_job.items()):
        detail = json.loads(request(f"/api/jobs/{job}", token))["job"]
        if (
            detail.get("id") != job
            or detail.get("display_user") != ACCOUNT
            or detail.get("case_set") != "standard"
            or not detail.get("job_completed")
            or any(
                detail.get("submission_id") != op["submission_id"]
                or detail.get("benchmark_version") != op["benchmark_version"]
                for op in ops
            )
        ):
            raise ValueError(f"official Job provenance mismatch: {job}")
        archive = request(f"/api/jobs/{job}/submission/download", token)
        common, selected = source_files(archive, ops)
        downloaded[job] = (archive, common)
        source_sets.update(selected)

    jobs_dir = ROOT / "jobs"
    ops_dir = ROOT / "operators"
    jobs_dir.mkdir(exist_ok=True)
    ops_dir.mkdir(exist_ok=True)
    for old_dir in jobs_dir.iterdir():
        if old_dir.is_dir() and old_dir.name not in by_job:
            rmtree(old_dir)
    for old_dir in ops_dir.iterdir():
        if old_dir.is_dir() and old_dir.name not in {
            row["operator_key"] for row in rows
        }:
            rmtree(old_dir)

    for job, (archive, common) in downloaded.items():
        directory = jobs_dir / job
        if directory.exists():
            rmtree(directory)
        for name, data in common.items():
            write_file(directory / name, data)
        write_file(
            directory / "provenance.json",
            (
                json.dumps(
                    {
                        "job_id": job,
                        "official_submission_zip_sha256": digest(archive),
                        "files": {
                            name: digest(data) for name, data in sorted(common.items())
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            ).encode(),
        )

    for row in rows:
        key = row["operator_key"]
        directory = ops_dir / key
        if directory.exists():
            rmtree(directory)
        files = source_sets[key]
        for name, data in files.items():
            write_file(directory / name, data)
        write_file(
            directory / "provenance.json",
            (
                json.dumps(
                    {
                        **row,
                        "source_job_scaffold": f"jobs/{row['job_id']}",
                        "files": {
                            name: digest(data) for name, data in sorted(files.items())
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            ).encode(),
        )

    current_path.parent.mkdir(exist_ok=True)
    write_file(
        current_path,
        (
            json.dumps(
                {
                    "solution_id": SOLUTION_ID,
                    "source_url": BASE_URL
                    + f"/api/public/leaderboard/solution/{SOLUTION_ID}/components?result_scope=default",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "operators": rows,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        ).encode(),
    )
    write_file(ROOT / "CATALOG.md", render_catalog(rows).encode())
    print(f"Updated {len(rows)} operators from {len(by_job)} official Jobs.")


if __name__ == "__main__":
    main()
