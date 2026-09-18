from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

import app as api
from test_inference_utils import tokenizer


@pytest.mark.parametrize("change", [{"question": " "}, {"beam_size": 100}, {"max_length": -1}, {"max_new_tokens": 10000}, {"model_type": "invalid"}])
def test_invalid_request_rejected_before_model_access(change):
    payload = {"model_type": "extractive", "question": "what?", "context": "the answer", **change}
    assert TestClient(api.app).post("/predict", json=payload).status_code == 422


def test_missing_checkpoints_are_reported_not_a_startup_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(api, "extractive_model", None)
    monkeypatch.setattr(api, "generative_model", None)
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        response = client.post("/predict", json={"model_type": "extractive", "question": "what?", "context": "the answer"})
        assert response.status_code == 503


def test_long_passage_windows_are_padded(tokenizer, monkeypatch):
    batches = []
    class Model:
        def __call__(self, input_ids, **kwargs):
            batches.append(input_ids.shape)
            logits = torch.zeros_like(input_ids, dtype=torch.float32)
            logits[:, 1] = 1
            return {"start_logits": logits, "end_logits": logits}
    monkeypatch.setattr(api, "extractive_model", Model())
    monkeypatch.setattr(api, "extractive_tokenizer", tokenizer)
    monkeypatch.setattr(api, "extractive_cfg", SimpleNamespace(max_position_embeddings=64))
    monkeypatch.setattr(api, "device", "cpu")
    request = api.PredictRequest(model_type="extractive", question="what", context="the answer " * 100, max_length=64)
    result = api.run_extractive(request)
    assert batches[0][0] > 1
    assert batches[0][1] == 64
    assert isinstance(result["answer"], str)
