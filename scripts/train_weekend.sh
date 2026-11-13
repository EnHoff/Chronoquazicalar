#!/bin/bash
# Weekend PPO run: resumes from checkpoints/ou_ppo/latest.pkl if present. Memory-capped so it can't take WSL down.
#   nohup scripts/train_weekend.sh > logs/train.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate chrono
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/oujax_xla}
exec systemd-run --user --scope -q -p MemoryMax=7000M -p MemorySwapMax=0 \
  python -m training.ou_ppo --batch 512 --steps 64 --bank 4096 --minibatches 32 --hours 100 --ckpt-minutes 20 "$@"
