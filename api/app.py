
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.sparse import issparse
from joblib import load

from model import load_model
from schemas import FlowRequest, BatchFlowRequest


BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    BASE_DIR / "checkpoints" /
    "dual_head_federated_transformer.pt"
)

PREPROCESSOR_PATH = (
    BASE_DIR / "processed_data" / "preprocessor.pkl"
)

CONFIG_PATH = (
    BASE_DIR / "checkpoints" / "risk_engine_config.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

N_MC_PASSES = 20
EPS = 1e-10

app = FastAPI(
    title="Federated Transformer IDS API",
    description="Risk-aware intrusion detection and zero-day analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# Load artifacts once during application startup
# --------------------------------------------------

preprocessor = load(PREPROCESSOR_PATH)

with open(CONFIG_PATH) as f:
    risk_config = json.load(f)

model = load_model(str(CHECKPOINT_PATH), DEVICE)

FEATURE_NAMES = list(preprocessor.feature_names_in_)
CLASS_NAMES = risk_config["class_names"]

RISK_WEIGHTS = risk_config["risk_formula"]

OOD_THRESHOLD = risk_config["anomaly_normalization"]["ood_threshold"]

RISK_THRESHOLD = risk_config["risk_threshold"]

NORMAL_INDEX = CLASS_NAMES.index("Normal")

NUM_FEATURES = len(FEATURE_NAMES)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess_flows(flows):
    rows = []

    for flow in flows:
        missing = set(FEATURE_NAMES) - set(flow.keys())
        extra = set(flow.keys()) - set(FEATURE_NAMES)

        if missing:
            raise ValueError(
                f"Missing features: {sorted(missing)}"
            )

        if extra:
            raise ValueError(
                f"Unexpected features: {sorted(extra)}"
            )

        rows.append({
            feature: flow[feature]
            for feature in FEATURE_NAMES
        })

    dataframe = pd.DataFrame(
        rows,
        columns=FEATURE_NAMES
    )

    transformed = preprocessor.transform(dataframe)

    if issparse(transformed):
        transformed = transformed.toarray()

    transformed = np.asarray(
        transformed,
        dtype=np.float32
    )

    if transformed.shape[1] != 122:
        raise ValueError(
            f"Expected 122 transformed features, got "
            f"{transformed.shape[1]}"
        )

    if not np.isfinite(transformed).all():
        raise ValueError(
            "Input contains invalid or non-finite values"
        )

    return torch.tensor(
        transformed,
        dtype=torch.float32,
        device=DEVICE
    )


# --------------------------------------------------
# Inference and uncertainty
# --------------------------------------------------

@torch.no_grad()
def run_inference(x):
    model.eval()

    # Deterministic anomaly score
    logits, anomaly_scores = model.compute_anomaly_score(x)

    # MC Dropout: activate Transformer dropout only
    model.eval()

    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.TransformerEncoderLayer)):
            module.train()

    probability_samples = []

    for _ in range(N_MC_PASSES):
        output = model(x)
        pass_logits = output[0]

        probabilities = torch.softmax(
            pass_logits,
            dim=-1
        )

        probability_samples.append(probabilities)

    # Restore deterministic evaluation mode
    model.eval()

    probability_samples = torch.stack(
        probability_samples,
        dim=0
    )

    mean_probabilities = probability_samples.mean(dim=0)

    predictive_entropy = -(
        mean_probabilities *
        torch.log(mean_probabilities + EPS)
    ).sum(dim=-1)

    individual_entropy = -(
        probability_samples *
        torch.log(probability_samples + EPS)
    ).sum(dim=-1)

    expected_entropy = individual_entropy.mean(dim=0)

    mutual_information = (
        predictive_entropy - expected_entropy
    )

    confidence, predictions = mean_probabilities.max(dim=-1)

    # --------------------------------------------------
    # Risk fusion
    # --------------------------------------------------

    attack_probability = (
        1.0 - mean_probabilities[:, NORMAL_INDEX]
    )

    normalized_anomaly = torch.clamp(
        anomaly_scores / OOD_THRESHOLD,
        min=0,
        max=1
    )

    normalized_uncertainty = torch.clamp(
        predictive_entropy / np.log(len(CLASS_NAMES)),
        min=0,
        max=1
    )

    risk = 100.0 * (
        RISK_WEIGHTS["attack_probability_weight"]
        * attack_probability
        +
        RISK_WEIGHTS["anomaly_score_weight"]
        * normalized_anomaly
        +
        RISK_WEIGHTS["uncertainty_weight"]
        * normalized_uncertainty
    )

    results = []

    for i in range(len(x)):
        risk_value = float(risk[i].item())

        if risk_value < 25:
            level = "LOW"
        elif risk_value < 50:
            level = "MEDIUM"
        elif risk_value < 75:
            level = "HIGH"
        else:
            level = "CRITICAL"

        results.append({
            "predicted_class": CLASS_NAMES[
                int(predictions[i].item())
            ],
            "class_probabilities": {
                CLASS_NAMES[j]: float(
                    mean_probabilities[i, j].item()
                )
                for j in range(len(CLASS_NAMES))
            },
            "confidence": float(confidence[i].item()),
            "anomaly_score": float(
                anomaly_scores[i].item()
            ),
            "anomaly_detected": bool(
                anomaly_scores[i].item() >= OOD_THRESHOLD
            ),
            "predictive_entropy": float(
                predictive_entropy[i].item()
            ),
            "mutual_information": float(
                mutual_information[i].item()
            ),
            "attack_probability": float(
                attack_probability[i].item()
            ),
            "risk_score": risk_value,
            "risk_level": level,
            "alert": bool(
                risk_value >= RISK_THRESHOLD
            )
        })

    return results


# --------------------------------------------------
# API endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Federated Transformer IDS",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.get("/model/info")
def model_info():
    return {
        "model": "DualHeadRiskTransformer",
        "input_features": NUM_FEATURES,
        "transformed_features": 122,
        "classes": CLASS_NAMES,
        "mc_dropout_passes": N_MC_PASSES,
        "risk_threshold": RISK_THRESHOLD,
        "anomaly_threshold": OOD_THRESHOLD,
        "device": str(DEVICE)
    }


@app.post("/predict")
def predict(request: FlowRequest):
    try:
        x = preprocess_flows([request.features])
        result = run_inference(x)

        return {
            "success": True,
            "prediction": result[0]
        }

    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )


@app.post("/predict/batch")
def predict_batch(request: BatchFlowRequest):
    try:
        x = preprocess_flows(request.flows)
        results = run_inference(x)

        return {
            "success": True,
            "total_flows": len(results),
            "predictions": results
        }

    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
