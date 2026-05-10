import os
import torch
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Optimize CPU threads for HuggingFace Spaces free tier
torch.set_num_threads(1)

# Import internal logic
from transformers import AutoTokenizer
from safetensors.torch import load_file
from extractive_finetuning import BertForQuestionAnswering
from main_hybrid_decoder import GenerativeQAModelHybrid
from standard_generative_decoder import DecoderConfig, GenerativeQAModel as StandardGenerativeQAModel
from mlm_pretraining import ModelConfig
from generative_inference import decode_generated_ids, build_target_ids

class PredictRequest(BaseModel):
    model_type: str = Field(..., description="extractive or generative")
    question: str
    context: str
    max_length: int = 256
    # Extractive args
    doc_stride: int = 64
    n_best: int = 20
    max_answer_length: int = 30
    # Generative args
    beam_size: int = 4
    max_new_tokens: int = 32
    length_penalty: float = 1.0
    enable_no_answer_gate: bool = True
    no_answer_threshold: float = 0.0

# Global models
extractive_model = None
extractive_tokenizer = None
extractive_cfg = None

generative_model = None
generative_tokenizer = None
generative_enc_cfg = None

device = "cuda" if torch.cuda.is_available() else "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global extractive_model, extractive_tokenizer, extractive_cfg
    global generative_model, generative_tokenizer, generative_enc_cfg
    
    # Paths 
    ext_dir = Path("checkpoints_qa_squad_v2_lr5e-5_len256_e3")
    gen_path = Path("checkpoints_generative_qa_stageE_tradeoff/best.pt")
    gen_dir = Path("checkpoints_generative_qa_stageE_tradeoff")
    
    # Load Extractive Model
    print("Loading Extractive Model...")
    config_path = ext_dir / "model_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        extractive_cfg = ModelConfig(**json.load(f))
        
    extractive_tokenizer = AutoTokenizer.from_pretrained(str(ext_dir), use_fast=True)


#     no_answer_threshold: float = 0.0
# 
# # Global models
# extractive_model = None
# extractive_tokenizer = None
# extractive_cfg = None
# 
# generative_model = None
# generative_tokenizer = None
# generative_enc_cfg = None
# 
# device = "cuda" if torch.cuda.is_available() else "cpu"
# 
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global extractive_model, extractive_tokenizer, extractive_cfg
#     global generative_model, generative_tokenizer, generative_enc_cfg
#     
#     # Paths 
#     ext_dir = Path("checkpoints_qa_squad_v2_lr5e-5_len256_e3")
#     gen_path = Path("checkpoints_generative_qa_stageE_tradeoff/best.pt")
#     gen_dir = Path("checkpoints_generative_qa_stageE_tradeoff")
#     
#     # Load Extractive Model
#     print("Loading Extractive Model...")
#     config_path = ext_dir / "model_config.json"
#     with open(config_path, "r", encoding="utf-8") as f:
#         extractive_cfg = ModelConfig(**json.load(f))
#         
#     extractive_tokenizer = AutoTokenizer.from_pretrained(str(ext_dir), use_fast=True)
