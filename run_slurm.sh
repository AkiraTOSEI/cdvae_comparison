#!/bin/bash
#SBATCH --job-name=HPO_perov      # ジョブ名
#SBATCH --output=output.txt          # 標準出力およびエラー出力ファイル
#SBATCH --error=output.txt           # エラーもoutput.txtに書き込む
#SBATCH --gres=gpu:1                 # GPUを1つ使用
#SBATCH --mail-type=END,FAIL         # メール通知のタイミング（終了とエラー発生時）

OMP_NUM_THREADS=${SLURM_NTASKS}

## Slack Webhook URL (環境によっては管理者に問い合わせて設定が必要)
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/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"

# Slackにメッセージを送る関数
function notify_slack {
    curl -X POST --data-urlencode "payload={\"text\": \"$1\"}" $SLACK_WEBHOOK_URL
}

# デバッグ用: 環境変数の確認
workspace="ws1"

source activate pytorch

# ジョブ開始の通知
notify_slack "mosso ${workspace}で SLURMジョブが開始されました: ${SLURM_JOB_NAME} (ジョブID: ${SLURM_JOB_ID})"

# 実行するコマンド
#sh multi_task.sh
python HPO_crystalformer_promising_candidates.py

# 実行が成功したかどうかを確認
if [ $? -eq 0 ]; then
    notify_slack "DONE!!!!      mosso ${workspace}, ${SLURM_JOB_NAME} (ジョブID: $SLURM_JOB_ID)"
else
    notify_slack "mosso ${workspace}で SLURMジョブでエラーが発生しました: ${SLURM_JOB_NAME} (ジョブID: $SLURM_JOB_ID)"
fi

