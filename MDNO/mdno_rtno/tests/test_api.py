import pytest

pytest.importorskip("torch", reason="PyTorch is required for API validation tests")

import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from api import ProductionAPI


class ErrorModel(nn.Module):
    def forward(self, inputs):
        raise ValueError("missing temperature field")


class EchoModel(nn.Module):
    def forward(self, inputs):
        return {"output": torch.ones(1)}


@pytest.fixture(scope="module")
def api_client():
    models = {"rtno": ErrorModel(), "mdno": EchoModel()}
    api = ProductionAPI(models=models)
    return TestClient(api.app)


def test_rtno_missing_fields_rejected(api_client):
    response = api_client.post(
        "/predict",
        json={
            "model_type": "rtno",
            "inputs": {"humidity": [0.5]},
            "shape": {"dimensions": [1]},
        },
    )
    assert response.status_code == 422


def test_model_value_error_returns_400(api_client):
    response = api_client.post(
        "/predict",
        json={
            "model_type": "rtno",
            "inputs": {
                "temperature": [288.0],
                "pressure": [101325.0],
            },
            "shape": {"dimensions": [1]},
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["detail"] == "missing temperature field"
