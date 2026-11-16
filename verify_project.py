
from pathlib import Path

ROOT = Path(__file__).resolve().parent

required_files = [
    "api/app.py",
    "api/model.py",
    "api/schemas.py",
    "api/requirements.txt",
    "checkpoints/dual_head_federated_transformer.pt",
    "checkpoints/risk_engine_config.json",
    "processed_data/preprocessor.pkl",
    "results/zero_day_validation_v2.json",
    "results/mc_dropout_uncertainty.json",
    "results/uncertainty_error_analysis.json",
    "results/risk_engine_metrics.json",
    "results/risk_threshold_validation.json",
    "results/federated_metrics.json",
]

print("PROJECT:", ROOT)
print("\nFILE VERIFICATION")

missing = []

for file in required_files:
    path = ROOT / file
    status = "FOUND" if path.is_file() else "MISSING"

    print(f"[{status}] {file}")

    if not path.is_file():
        missing.append(file)

print("\nSUMMARY")
print(f"Found: {len(required_files) - len(missing)}/{len(required_files)}")
print(f"Missing: {len(missing)}")

if missing:
    print("\nMissing files:")
    for file in missing:
        print("-", file)
else:
    print("\nAll essential API files are present.")
