#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/generative_train_$(date +%Y%m%d_%H%M%S).log"

nohup python generative_finetuning.py \


# #!/usr/bin/env bash
# set -euo pipefail
# 
# LOG_DIR="logs"
# mkdir -p "${LOG_DIR}"
# LOG_FILE="${LOG_DIR}/generative_train_$(date +%Y%m%d_%H%M%S).log"
# 
# nohup python generative_finetuning.py \
