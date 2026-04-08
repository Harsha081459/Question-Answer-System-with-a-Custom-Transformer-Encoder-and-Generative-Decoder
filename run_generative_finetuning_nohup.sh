#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/generative_train_$(date +%Y%m%d_%H%M%S).log"

nohup python generative_finetuning.py \
  --tokenizer_path checkpoints_pretrain_base_seq256/step_20000 \
  --pretrain_ckpt checkpoints_pretrain_base_seq256/step_20000/checkpoint.pt \
  --output_dir checkpoints_generative_qa \
  --decoder_variant hybrid \
  --max_input_len 256 \
  --max_target_len 48 \
  --train_batch_size 6 \
  --eval_batch_size 8 \
  --grad_accum 2 \


# #!/usr/bin/env bash
# set -euo pipefail
# 
# LOG_DIR="logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/generative_train_$(date +%Y%m%d_%H%M%S).log"
# 
# nohup python generative_finetuning.py \
#   --tokenizer_path checkpoints_pretrain_base_seq256/step_20000 \
#   --pretrain_ckpt checkpoints_pretrain_base_seq256/step_20000/checkpoint.pt \
#   --output_dir checkpoints_generative_qa \
#   --decoder_variant hybrid \
#   --max_input_len 256 \
#   --max_target_len 48 \
#   --train_batch_size 6 \
#   --eval_batch_size 8 \
#   --grad_accum 2 \
