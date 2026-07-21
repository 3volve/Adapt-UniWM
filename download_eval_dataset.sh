#!/usr/bin/env bash
# download_data.sh — Download the UniWM eval dataset subset into data/.
# Usage: bash download_data.sh

set -e

manifest="./eval_dataset_manifest.json"

mkdir -p data && cd data

for split in $(python -c "import json; data=json.load(open('../${manifest}')); print(' '.join(k for k, v in data.items() if 'episodes' in v))"); do
    echo "[DOWNLOAD] ${split}.tar"
    wget -c "https://huggingface.co/datasets/fly1113/UniWM_Dataset/resolve/main/${split}.tar"

    echo "[EXTRACT] ${split}.tar"
    python -c "
import json
import sys
import tarfile

split, manifest, tar_path = sys.argv[1:]
episodes = set(json.load(open(manifest))[split]['episodes'])

with tarfile.open(tar_path) as tar:
    members = [m for m in tar.getmembers() if any(part in episodes for part in m.name.split('/'))]
    tar.extractall(members=members)
" "${split}" "../${manifest}" "${split}.tar"

    echo "[CLEAN] removing ${split}.tar"
    rm -f "${split}.tar"
done

echo "[DONE] Datasets are ready under $(pwd)/"
