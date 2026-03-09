This file explains how to run robust MLM pretraining over SSH on Linux.

1) Setup
- conda activate <your_env>
- pip install -r requirements.txt

2) Start pretraining
```bash
python mlm_pretraining.py \
  --tokenizer_name bert-base-uncased \
  --size base \
  --seq_len 256 \
  --out_dir checkpoints_pretrain_base_seq256 \
  --resume_latest
```

3) Monitor
- tail -f logs/pretrain_base_YYYYMMDD_HHMMSS.log
- nvidia-smi

4) Resume behavior
- `--resume_latest` uses the checkpoint pointer file in the output directory.
- If the process restarts, re-run the same command and it resumes.

5) Stop safely
- kill -SIGINT $(cat logs/pretrain_base.pid)
- The script catches interrupt and saves final checkpoint.


# This file explains how to run robust MLM pretraining over SSH on Linux.
# 
# 1) Setup
# - conda activate <your_env>
# - pip install -r requirements.txt
# 
# 2) Start pretraining
# ```bash
# python mlm_pretraining.py \
#   --tokenizer_name bert-base-uncased \
#   --size base \
#   --seq_len 256 \
#   --out_dir checkpoints_pretrain_base_seq256 \
#   --resume_latest
# ```
# 
# 3) Monitor
# - tail -f logs/pretrain_base_YYYYMMDD_HHMMSS.log
# - nvidia-smi
# 
# 4) Resume behavior
# - `--resume_latest` uses the checkpoint pointer file in the output directory.
# - If the process restarts, re-run the same command and it resumes.
# 
# 5) Stop safely
# - kill -SIGINT $(cat logs/pretrain_base.pid)
# - The script catches interrupt and saves final checkpoint.
