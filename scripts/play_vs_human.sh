#!/bin/bash
# Put a policy checkpoint on the local Showdown server as "OUBot", accepting [Gen 9] OU challenges (CPU only,
# memory-capped). Starts the server if needed. Args go to evaluation/play_vs_human.py (--checkpoint, --name).
#   nohup scripts/play_vs_human.sh > logs/play_vs_human.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate chrono
curl -s localhost:8000 > /dev/null || { scripts/start_showdown.sh > /dev/null 2>&1; sleep 10; }
export JAX_PLATFORMS=cpu
exec systemd-run --user --scope -q -p MemoryMax=2500M -p MemorySwapMax=0 python -u -m evaluation.play_vs_human "$@"
