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
