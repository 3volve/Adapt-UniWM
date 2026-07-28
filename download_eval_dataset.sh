#!/usr/bin/env bash
# download_data.sh — Download the UniWM eval dataset subset into data/.
# Usage: bash download_data.sh

manifest="./cfg/eval_dataset_manifest.json"

mkdir -p data

for source in go_stanford recon sacson scand; do
    archive="data/${source}.tar"

    echo "[DOWNLOAD] ${source}.tar"
    wget -c \
        -O "$archive" \
        "https://huggingface.co/datasets/fly1113/UniWM_Dataset/resolve/main/${source}.tar"

    echo "[EXTRACT] ${source}.tar"
    python - "$source" "$manifest" "$archive" <<'PY'
import json
import sys
import tarfile

source, manifest_path, archive = sys.argv[1:]
manifest = json.load(open(manifest_path))
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

with tarfile.open(archive) as tar:
    members = [
        member
        for member in tar.getmembers()
        if any(part in requested for part in member.name.split("/"))
    ]
    print(f"{source}: extracting {len(requested)} requested trajectories")
    tar.extractall(path="data", members=members)
PY

    rm -f "$archive"
done

echo "[DONE] Dataset is ready under ./eval_data"