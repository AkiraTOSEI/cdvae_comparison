#!/bin/bash

# 学習率とステップ数の候補リスト
lrs=(0.00001 0.0001 0.001 0.01)
steps=(400 800 1600 3200 5000 200)

# 実行コマンド
# bash megnet_optimization_hpo.sh

# 2重ループで条件を全探索
for lr in "${lrs[@]}"; do
  for step in "${steps[@]}"; do
    echo "▶ Running with lr=$lr, num_gradient_steps=$step"
    poetry run python scripts/evaluate.py --tasks opt --label megnet --lr $lr --num_starting_points 128 --model_path /home/fujii/cdvae_comparison/hydra/singlerun/2025-04-15/shin_megnet_lr5e-6/ --target_bg 2.5 --num_saved_crys 0 --megnet_loss_mode True --coef_e_form 0 --num_gradient_steps $step
  done
done
