
import pandas as pd
import joblib
import requests

DATASET = "datasets/NSL-KDD/KDDTest+.txt"
PREPROCESSOR = "processed_data/preprocessor.pkl"

# Read tab-separated NSL-KDD data
df = pd.read_csv(DATASET, sep="\t", header=None)

# Load the exact training feature schema
preprocessor = joblib.load(PREPROCESSOR)
feature_names = list(preprocessor.feature_names_in_)

print("Dataset shape:", df.shape)
print("Feature count:", len(feature_names))

# Extract first 41 network-flow features
values = df.iloc[0, :41].tolist()

features = dict(zip(feature_names, values))

# Convert NumPy types to JSON-compatible Python types
features = {
    key: value.item() if hasattr(value, "item") else value
    for key, value in features.items()
}

print("Input features:", len(features))

# Send to local FastAPI
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": features},
    timeout=120
)

print("\nHTTP status:", response.status_code)

if response.status_code == 200:
    prediction = response.json()["prediction"]

    print("\n========== IDS PREDICTION ==========")
    print("Class:", prediction["predicted_class"])
    print("Confidence:", prediction["confidence"])
    print("Anomaly score:", prediction["anomaly_score"])
    print("Anomaly detected:", prediction["anomaly_detected"])
    print("Risk score:", prediction["risk_score"])
    print("Risk level:", prediction["risk_level"])
    print("Alert:", prediction["alert"])
else:
    print(response.text)
