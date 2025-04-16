#!/bin/bash

# 学習率とステップ数の候補リスト
lrs=(0.00001 0.0001 0.001 0.01)
steps=(200 400 800 1600 3200 5000)

# 実行コマンド
# bash megnet_optimization_hpo.sh

# 2重ループで条件を全探索
for lr in "${lrs[@]}"; do
  for step in "${steps[@]}"; do
    echo "▶ Running with lr=$lr, num_gradient_steps=$step"
    poetry run python scripts/evaluate.py --tasks opt --label supercon --lr $lr --num_starting_points 128  --model_path /home/fujii/cdvae_comparison/hydra/singlerun/2025-04-15/supercon/ --num_saved_crys 0 --num_gradient_steps $step
  done
done
