#!/usr/bin/env bash
# Download the manifest-selected UniWM source trajectories and build eval_data/.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="$repo_root/cfg/eval_dataset_manifest.json"
source_data_dir="$repo_root/data"
output_data_dir="$repo_root/eval_data"
download_dir="$repo_root/output/dataset_downloads"
dataset_revision="6585488b03fb5b60be1aba222999fa2a10c4e5b5"

if [[ -e "$output_data_dir" ]]; then
    echo "[ERROR] $output_data_dir already exists." >&2
    echo "Move or remove it before rebuilding the evaluation subset." >&2
    exit 1
fi

mkdir -p "$source_data_dir" "$download_dir"

for source in go_stanford recon sacson scand; do
    archive="$download_dir/${source}.tar"

    echo "[DOWNLOAD] ${source}.tar"
    wget -c \
        -O "$archive" \
        "https://huggingface.co/datasets/fly1113/UniWM_Dataset/resolve/${dataset_revision}/${source}.tar"

    echo "[EXTRACT] ${source}.tar"
    python - "$source" "$manifest" "$archive" "$source_data_dir" <<'PY'
import json
import sys
import tarfile
from pathlib import Path

source, manifest_path, archive, output_dir = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
requested = set()

for logical_dataset, entry in manifest.items():
    if logical_dataset == "action_token_vocabulary":
        continue

    for split in ("train", "validation", "test"):
        for reference in entry[split]:
            if "/" in reference:
                ref_source, trajectory_id = reference.split("/", 1)
            else:
                ref_source, trajectory_id = logical_dataset, reference

            if ref_source == source:
                requested.add(trajectory_id)

target = Path(output_dir).resolve()
with tarfile.open(archive) as tar:
    members = [
        member
        for member in tar.getmembers()
        if any(part in requested for part in member.name.split("/"))
    ]

    for member in members:
        destination = (target / member.name).resolve()
        if not destination.is_relative_to(target):
            raise ValueError(f"Unsafe archive path: {member.name!r}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"Unsupported archive member: {member.name!r}")

    print(f"{source}: extracting {len(requested)} requested trajectories")
    tar.extractall(path=target, members=members)
PY

    rm -f "$archive"
done

python "$repo_root/scripts/create_manifest_dataset.py" \
    --manifest "$manifest" \
    --source-data-dir "$source_data_dir" \
    --output-data-dir "$output_data_dir"

echo "[DONE] Dataset is ready under $output_data_dir"
