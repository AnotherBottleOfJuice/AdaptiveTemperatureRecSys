#!/usr/bin/env bash
# Download the Amazon Reviews 2023 "Beauty_and_Personal_Care" raw reviews (~11 GB)
# for AmazonBeautyDataset, which reads it via polars.scan_ndjson (see recdata/.../amazon/loader.py).
#
# Result: data/amazon_beauty/interactions.jsonl  (symlink into the HF cache, no extra disk)
# matches `path_interactions` in configs/amazon/*/*.yaml.
#
# Run on the remote from the repo root:
#   bash scripts/download_amazon.sh
# On the SLURM cluster, load the env first (uncomment the two lines below).
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

module load Python
source activate adaptivetemperaturerecsys

PYTHON="${PYTHON:-python}"
mkdir -p data/amazon_beauty

"$PYTHON" - <<'PY'
import os, pathlib
from huggingface_hub import hf_hub_download

src = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    filename="raw/review_categories/Beauty_and_Personal_Care.jsonl",
    repo_type="dataset",
)
dst = pathlib.Path("data/amazon_beauty/interactions.jsonl")
dst.parent.mkdir(parents=True, exist_ok=True)
if dst.is_symlink() or dst.exists():
    dst.unlink()
dst.symlink_to(os.path.abspath(src))
print("ready:", dst, "->", src)
PY
