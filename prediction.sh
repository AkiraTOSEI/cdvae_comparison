#!/bin/bash                                                                                                                    

# 実行コマンド: nohup bash run.sh > run.log 2>&1 &

for lr in 0.0001 0.0005 0.001 0.005 0.01
do
  # lr を 100000倍して整数にしてから、ゼロ埋めして5桁のlabelに変換
  label_num=$(echo "$lr * 100000" | bc | cut -d'.' -f1)
  label=$(printf "lr%05d" "$label_num")
  poetry run python scripts/evaluate.py --tasks opt --label "$label" --lr "$lr" --num_starting_points 32  --model_path /home/fujii/cdvae_comparison/hydra/singlerun/2025-04-15/megnet_lr1e-4/ --target_bg 2.5 --num_saved_crys 0 --megnet_loss_mode True
done

