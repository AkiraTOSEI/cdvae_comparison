#!/bin/bash

# 学習率とステップ数の候補リスト
#lrs=(0.00001 0.0001 0.001 0.01)
#steps=(400 800 1600 3200 5000 200)
bgs=(0.5 1.0 1.5 2.0 3.0 3.5 4.0 4.5 5.0)

# 実行コマンド
# bash megnet_optimization_hpo.sh

# 2重ループで条件を全探索
for bg in "${bgs[@]}"; do
    echo "▶ Running with bg=$bg"
    poetry run python scripts/evaluate.py --tasks opt --label megnet --lr 0.0001 --num_starting_points 128 --model_path /home/fujii/cdvae_comparison/hydra/singlerun/2025-04-15/shin_megnet_lr5e-6/ --target_bg $bg --num_saved_crys 0 --megnet_loss_mode True --coef_e_form 0 --num_gradient_steps 3200
done
