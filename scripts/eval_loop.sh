#!/bin/bash
# Every EVAL_HOURS (default 3), play the newest checkpoints/ou_ppo/policy_*.pkl against SimpleHeuristics on the
# local Showdown server: 50 games, CPU only, niced, in a 2.5 GB cgroup. Results append to
# evaluation/results/progress.jsonl.
#   nohup scripts/eval_loop.sh > logs/eval_loop.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate chrono
export JAX_PLATFORMS=cpu
while true; do
  sleep "$(( ${EVAL_HOURS:-3} * 3600 ))"
  ckpt=$(ls -t checkpoints/ou_ppo/policy_*.pkl | head -1)
  curl -s localhost:8000 > /dev/null || { scripts/start_showdown.sh > /dev/null 2>&1; sleep 10; }
  echo "$(date -Is) evaluating $ckpt"
  nice -n 15 systemd-run --user --scope -q -p MemoryMax=2500M -p MemorySwapMax=0 \
    python -m evaluation.eval_ou_policy --checkpoint "$ckpt" --opponent heuristic --games 50 \
    --out evaluation/results/progress.jsonl | tail -1
done
