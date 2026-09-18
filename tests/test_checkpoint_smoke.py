import math
import os

import pytest
from fastapi.testclient import TestClient

from app import app

pytestmark = pytest.mark.skipif(os.getenv("RUN_QA_SMOKE") != "1", reason="requires the published checkpoint download")


def test_published_checkpoints_serve_both_modes():
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        base = {"question": "Who created Python?", "context": "Python was created by Guido van Rossum and released in 1991."}
        for mode in ["extractive", "generative"]:
            response = client.post("/predict", json={**base, "model_type": mode, "beam_size": 1, "max_new_tokens": 12})
            assert response.status_code == 200, response.text
            result = response.json()
            assert isinstance(result["answer"], str)
            print(mode, result)
            if mode == "extractive":
                assert math.isfinite(result["span_score"])
                assert result["answer"] == "Guido van Rossum"
            else:
                assert math.isfinite(result["gate"]["score_diff"])
        response = client.post("/predict", json={**base, "model_type": "extractive", "context": base["context"] * 40})
        assert response.status_code == 200, response.text
