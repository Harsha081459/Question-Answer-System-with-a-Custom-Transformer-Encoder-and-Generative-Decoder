import os
import torch
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from threading import Lock
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
from generative_data import normalize_text

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("QA_MODEL_DIR", ROOT / "models_fp16"))
_inference_lock = Lock()


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, allow_inf_nan=False)
    model_type: Literal["extractive", "generative"] = Field(..., description="extractive or generative")
    question: str = Field(min_length=1, max_length=512)
    context: str = Field(min_length=1, max_length=12000)
    max_length: int = Field(default=256, ge=64, le=512)
    # Extractive args
    doc_stride: int = Field(default=64, ge=0, le=128)
    n_best: int = Field(default=20, ge=1, le=40)
    max_answer_length: int = Field(default=30, ge=1, le=128)
    # Generative args
    beam_size: int = Field(default=4, ge=1, le=4)
    max_new_tokens: int = Field(default=32, ge=1, le=64)
    length_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
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
    ext_dir = MODEL_ROOT / "extractive"
    gen_dir = MODEL_ROOT / "generative"
    gen_path = gen_dir / "best.pt"
    extractive_model = generative_model = None
    if not (ext_dir / "model.safetensors").is_file() or not gen_path.is_file():
        print("Checkpoints missing: run python download_models.py and restart. /ready will return 503.")
        yield
        return
    
    # Load Extractive Model
    print("Loading Extractive Model...")
    config_path = ext_dir / "model_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        extractive_cfg = ModelConfig(**json.load(f))
        
    extractive_tokenizer = AutoTokenizer.from_pretrained(str(ext_dir), use_fast=True)
    if extractive_tokenizer.pad_token is None:
        extractive_tokenizer.pad_token = extractive_tokenizer.sep_token
        
    extractive_model = BertForQuestionAnswering(extractive_cfg)
    ext_state = load_file(str(ext_dir / "model.safetensors"))
    extractive_model.load_state_dict(ext_state, strict=True)
    extractive_model.to(device)
    extractive_model.eval()
    
    # Load Generative Model
    print("Loading Generative Model...")
    gen_payload = torch.load(gen_path, map_location="cpu", weights_only=True)
    generative_enc_cfg = ModelConfig(**gen_payload["encoder_config"])
    gen_dec_cfg = DecoderConfig(**gen_payload["decoder_config"])
    model_class = GenerativeQAModelHybrid if gen_payload.get("decoder_variant") == "hybrid" else StandardGenerativeQAModel
    generative_model = model_class(generative_enc_cfg, gen_dec_cfg)
    generative_model.load_state_dict(gen_payload["model"], strict=True)
    generative_model.to(device)
    generative_model.eval()
    
    generative_tokenizer = AutoTokenizer.from_pretrained(str(gen_dir), use_fast=True)
    if generative_tokenizer.pad_token is None:
        generative_tokenizer.pad_token = generative_tokenizer.sep_token

    print("Models loaded successfully.")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "models": {"extractive": extractive_model is not None, "generative": generative_model is not None}}


@app.get("/ready")
def ready():
    if extractive_model is None or generative_model is None:
        raise HTTPException(status_code=503, detail="Download the published checkpoints and restart the server")
    return {"status": "ready", "generative_status": "experimental"}


@app.post("/predict")
def predict(req: PredictRequest):
    model = extractive_model if req.model_type == "extractive" else generative_model
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable; run python download_models.py and restart")
    with _inference_lock:
        try:
            return run_extractive(req) if req.model_type == "extractive" else run_generative(req)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Input cannot be tokenized with these length settings; shorten the question or increase max_length") from exc

def run_extractive(req: PredictRequest):
    max_length = min(req.max_length, extractive_cfg.max_position_embeddings)
    doc_stride = min(req.doc_stride, max(8, max_length // 4))

    enc = extractive_tokenizer(
        [req.question],
        [req.context],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
    
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids)).to(device)

    with torch.no_grad():
        out = extractive_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
    start_logits = out["start_logits"].cpu().numpy()
    end_logits = out["end_logits"].cpu().numpy()

    best_score = -1e30
    best_text = ""
    best_null_score = 1e30

    for i in range(input_ids.shape[0]):
        offsets = enc["offset_mapping"][i].tolist()
        seq_ids = enc.sequence_ids(i)
        ids_i = input_ids[i].tolist()
        cls_idx = ids_i.index(extractive_tokenizer.cls_token_id)
        
        null_score = float(start_logits[i][cls_idx] + end_logits[i][cls_idx])
        if null_score < best_null_score:
            best_null_score = null_score

        s_idx = start_logits[i].argsort()[-1 : -req.n_best - 1 : -1].tolist()
        e_idx = end_logits[i].argsort()[-1 : -req.n_best - 1 : -1].tolist()
        for s in s_idx:
            for e in e_idx:
                if s >= len(offsets) or e >= len(offsets):
                    continue
                if seq_ids[s] != 1 or seq_ids[e] != 1:
                    continue
                if e < s or (e - s + 1) > req.max_answer_length:
                    continue
                st, en = offsets[s][0], offsets[e][1]
                if st is None or en is None:
                    continue
                score = float(start_logits[i][s] + end_logits[i][e])
                if score > best_score:
                    best_score = score
                    best_text = req.context[st:en]

    output = {
        "question": req.question,
        "answer": best_text,
        "span_score": best_score,
        "null_score": best_null_score,
        "score_diff_null_minus_span": best_null_score - best_score,
    }
    
    if (best_null_score - best_score) > req.no_answer_threshold:
        output["answer"] = ""
        output["predicted_no_answer"] = True
    else:
        output["predicted_no_answer"] = False

    return output

def run_generative(req: PredictRequest):
    tok = generative_tokenizer
    model = generative_model
    
    inp = f"question: {req.question} context: {req.context}"
    enc = tok(
        [inp],
        truncation=True,
        max_length=min(req.max_length, generative_enc_cfg.max_position_embeddings),
        return_tensors="pt",
    )
    enc_ids = enc["input_ids"].to(device)
    enc_mask = enc["attention_mask"].to(device)
    enc_ttype = enc.get("token_type_ids", torch.zeros_like(enc_ids)).to(device)

    bos = tok.cls_token_id if tok.cls_token_id is not None else tok.pad_token_id
    eos = tok.sep_token_id if tok.sep_token_id is not None else tok.pad_token_id
    pad = tok.pad_token_id
    
    no_answer_text = "The context does not contain the answer."

    with torch.no_grad():
        if req.enable_no_answer_gate:
            out, pred_logprob, _ = model.generate(
                encoder_input_ids=enc_ids,
                encoder_token_type_ids=enc_ttype,
                encoder_attention_mask=enc_mask,
                bos_token_id=bos,
                eos_token_id=eos,
                pad_token_id=pad,
                max_new_tokens=req.max_new_tokens,
                beam_size=req.beam_size,
                length_penalty=req.length_penalty,
                return_logprob=True,
            )
            out_ids = out[0].tolist()
        else:
            out_ids = model.generate(
                encoder_input_ids=enc_ids,
                encoder_token_type_ids=enc_ttype,
                encoder_attention_mask=enc_mask,
                bos_token_id=bos,
                eos_token_id=eos,
                pad_token_id=pad,
                max_new_tokens=req.max_new_tokens,
                beam_size=req.beam_size,
                length_penalty=req.length_penalty,
            )[0].tolist()

    raw_answer = decode_generated_ids(tok, out_ids, bos=bos, eos=eos, pad=pad)

    output = {"question": req.question, "answer": raw_answer, "raw_answer": raw_answer}
    if req.enable_no_answer_gate:
        noans_ids = build_target_ids(
            tokenizer=tok,
            text=no_answer_text,
            bos=bos,
            eos=eos,
            max_new_tokens=req.max_new_tokens,
            device=device,
        )
        with torch.no_grad():
            noans_logprob = model.sequence_logprob(
                encoder_input_ids=enc_ids,
                encoder_token_type_ids=enc_ttype,
                encoder_attention_mask=enc_mask,
                target_ids=noans_ids,
                pad_token_id=pad,
                normalize_by_length=True,
            )[0].item()
            
        score_diff = noans_logprob - pred_logprob
        gated = score_diff > req.no_answer_threshold
        output["answer"] = no_answer_text if gated else raw_answer
        output["gate"] = {
            "score_diff": score_diff,
            "pred_avg_logprob": pred_logprob,
            "no_answer_avg_logprob": noans_logprob,
            "selected_no_answer": gated,
        }

    output["predicted_no_answer"] = normalize_text(output["answer"]) in {"", normalize_text(no_answer_text)}
    return output

# Serve frontend
app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")
