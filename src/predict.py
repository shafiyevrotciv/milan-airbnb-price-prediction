import argparse
from pathlib import Path
import joblib
import pandas as pd

from preprocess import engineer_features, transform_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def predict(input_path: Path, output_path: Path):
    print(f"1. Reading input data: {input_path}")
    df_raw = pd.read_csv(input_path)

    print("2. Creating features and applying preprocessing")
    df_features = engineer_features(df_raw, is_train=False)
    X_processed = transform_data(df_features)

    print("3. Models loading and calculating predictions")
    xgboost_tuned = joblib.load(MODELS_DIR / "xgboost_tuned.joblib")
    lgbm_tuned = joblib.load(MODELS_DIR / "lgbm_tuned.joblib")
    cb = joblib.load(MODELS_DIR / "catboost_base.joblib")
    weights = joblib.load(MODELS_DIR / "weights.joblib")

    preds_xgb = xgboost_tuned.predict(X_processed)
    preds_lgb = lgbm_tuned.predict(X_processed)
    preds_cat = cb.predict(X_processed)

    final_weighted = (
        weights[0] * preds_xgb + weights[1] * preds_lgb + weights[2] * preds_cat
    )

    output_df = pd.DataFrame({"predicted_price": final_weighted.round(2)})
    if "id" in df_raw.columns:
        output_df.insert(0, "id", df_raw["id"])

    output_df.to_csv(output_path, index=False)
    print(f"4. Ready. Predictions are saved in: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Getting Airbnb Milan price predictions")
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "data" / "listings.csv",
        help="Path to CSV file with new data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data" / "predictions.csv",
        help="Where to put final CSV file",
    )
    args = parser.parse_args()
    predict(args.input, args.output)