#!/bin/bash
cd "/home/sem6/main file"
Q1="Who created Python?"
C1="Python was created by Guido van Rossum and first released in 1991. It is a dynamically typed programming language."

Q2="What is the capital of Mars?"
C2="Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System."

models=(
  "checkpoints_generative_qa_stageA_v1"
  "checkpoints_generative_qa_stageB_v1v2_run1"
  "checkpoints_generative_qa_stageC_sentence_run2"
  "checkpoints_generative_qa_stageD_balanced_run2"
)

for m in "${models[@]}"; do
  echo "======================================"
  echo "MODEL: $m"
  
  if [[ "$m" == *"stageC"* ]] || [[ "$m" == *"stageD"* ]]; then
    prefix="--instruction_prefix \"Answer in one concise sentence based only on the context.\""
  else
    prefix=""
  fi


# #!/bin/bash
# cd "/home/sem6/main file"
# Q1="Who created Python?"
# C1="Python was created by Guido van Rossum and first released in 1991. It is a dynamically typed programming language."
# 
# Q2="What is the capital of Mars?"
# C2="Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System."
# 
# models=(
#   "checkpoints_generative_qa_stageA_v1"
#   "checkpoints_generative_qa_stageB_v1v2_run1"
#   "checkpoints_generative_qa_stageC_sentence_run2"
#   "checkpoints_generative_qa_stageD_balanced_run2"
# )
# 
# for m in "${models[@]}"; do
#   echo "======================================"
#   echo "MODEL: $m"
#   
#   if [[ "$m" == *"stageC"* ]] || [[ "$m" == *"stageD"* ]]; then
#     prefix="--instruction_prefix \"Answer in one concise sentence based only on the context.\""
#   else
#     prefix=""
#   fi
