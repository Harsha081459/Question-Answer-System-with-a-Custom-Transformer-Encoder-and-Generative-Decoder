# Generative QA (Phase 3) - From Scratch Decoder + Reused Encoder

This phase adds a trainable Transformer decoder on top of your pretrained encoder checkpoint.

## Files
- `main_hybrid_decoder.py` - main custom hybrid decoder
- `standard_generative_decoder.py` - earlier Transformer decoder baseline
- `generative_data.py` - SQuAD/SQuAD-v2 data pipeline
- `generative_finetuning.py` - staged training (freeze -> partial unfreeze -> full)
- `generative_evaluation.py` - EM/F1/ROUGE-L/BLEU/no-answer accuracy
- `generative_inference.py` - inference CLI
- `run_generative_finetuning_nohup.sh` - nohup launcher

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
chmod +x run_generative_finetuning_nohup.sh
./run_generative_finetuning_nohup.sh
```

## High-impact training sequence (recommended)

### Stage A: Curriculum on SQuAD v1 only (answerable behavior first)
```bash
python generative_finetuning.py \
  --tokenizer_path checkpoints_pretrain_base_seq256/step_20000 \
  --pretrain_ckpt checkpoints_pretrain_base_seq256/step_20000/checkpoint.pt \
  --output_dir checkpoints_generative_qa_stageA_v1 \
  --decoder_variant hybrid \
  --no_squad_v2 \
  --epochs 3 \
  --train_batch_size 16 \
  --eval_batch_size 8 \
  --grad_accum 1 \
  --lr 3e-4 \
  --encoder_lr 8e-5 \
  --freeze_warmup_epochs 1 \
  --unfreeze_top_layers 4 \
  --fp16
```

### Stage B: Continue on SQuAD v1+v2 with answerable rebalancing
```bash
python generative_finetuning.py \
  --tokenizer_path checkpoints_pretrain_base_seq256/step_20000 \


# ```bash
# chmod +x run_generative_finetuning_nohup.sh
# ./run_generative_finetuning_nohup.sh
# ```
# 
# ## High-impact training sequence (recommended)
# 
# ### Stage A: Curriculum on SQuAD v1 only (answerable behavior first)
# ```bash
# python generative_finetuning.py \
#   --tokenizer_path checkpoints_pretrain_base_seq256/step_20000 \
#   --pretrain_ckpt checkpoints_pretrain_base_seq256/step_20000/checkpoint.pt \
#   --output_dir checkpoints_generative_qa_stageA_v1 \
#   --decoder_variant hybrid \
#   --no_squad_v2 \
#   --epochs 3 \
#   --train_batch_size 16 \
#   --eval_batch_size 8 \
#   --grad_accum 1 \
#   --lr 3e-4 \
#   --encoder_lr 8e-5 \
#   --freeze_warmup_epochs 1 \
#   --unfreeze_top_layers 4 \
#   --fp16
# ```
# 
# ### Stage B: Continue on SQuAD v1+v2 with answerable rebalancing
# ```bash
# python generative_finetuning.py \
#   --tokenizer_path checkpoints_pretrain_base_seq256/step_20000 \
