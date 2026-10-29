from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

LAT_TO_KM = 111.0
LON_TO_KM = 111.0 * np.cos(np.radians(45.46))

NUMERICAL_FEATURES = [
    "hosts_time_as_host_years",
    "host_listings_count",
    "accommodates",
    "bedrooms",
    "beds",
    "minimum_nights",
    "availability_30",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "review_scores_rating",
    "review_scores_location",
    "review_scores_value",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "reviews_per_month",
    "amenities_count",
    "days_since_first_review",
    "has_reviews",
    "bathrooms_count",
    "dist_duomo",
    "dist_porta_nuova",
    "dist_station",
    "coord_rot45_x",
    "coord_rot45_y",
    "lat_to_km",
    "lon_to_km",
]

CATEGORICAL_FEATURES = [
    "host_is_superhost",
    "room_type",
    "term_of_stay",
    "bathrooms_type",
    "geo_cluster",
]


def clean_target(df: pd.DataFrame, filter_outliers: bool = True) -> pd.DataFrame:
    df = df.dropna(subset=["price"]).copy()
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    if filter_outliers:
        limit = df["price"].quantile(0.99)
        df = df[df["price"] <= limit]
    return df


def engineer_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()

    # 1. Category of rent length
    df["term_of_stay"] = pd.cut(
        df["maximum_nights"],
        bins=[0, 14, 30, np.inf],
        labels=["short_term", "mid_term", "long_term"],
    )

    if is_train:
        df = df.dropna(subset=["bathrooms_text", "term_of_stay"]).copy()

    # 2. Number of amenities
    df["amenities_count"] = df["amenities"].fillna("").astype(str).str.count(r"\w+")

    # 3. Days since first review
    first_review_dt = pd.to_datetime(df["first_review"], errors="coerce")
    df["days_since_first_review"] = (pd.Timestamp.now() - first_review_dt).dt.days
    df["has_reviews"] = df["days_since_first_review"].notna().astype(int)

    # 4. Number and type of bathrooms
    extracted = (
        df["bathrooms_text"]
        .fillna("")
        .astype(str)
        .str.extract(r"(?P<bathrooms_count>\d+\.?\d*)\s*(?P<bathrooms_type>.*)")
    )
    df["bathrooms_count"] = pd.to_numeric(extracted["bathrooms_count"], errors="coerce")
    df["bathrooms_type"] = (
        extracted["bathrooms_type"]
        .replace({"baths": "bath", "shared baths": "shared bath", "": np.nan})
        .astype("category")
    )

    # 5. Distance till centers on Milan
    df["dist_duomo"] = (
        abs(df["latitude"] - 45.4641) * LAT_TO_KM
        + abs(df["longitude"] - 9.1919) * LON_TO_KM
    )
    df["dist_porta_nuova"] = (
        abs(df["latitude"] - 45.4839) * LAT_TO_KM
        + abs(df["longitude"] - 9.1896) * LON_TO_KM
    )
    df["dist_station"] = (
        abs(df["latitude"] - 45.4853) * LAT_TO_KM
        + abs(df["longitude"] - 9.2045) * LON_TO_KM
    )
    df["coord_rot45_x"] = df["latitude"] + df["longitude"]
    df["coord_rot45_y"] = df["latitude"] - df["longitude"]
    df["lat_to_km"] = df["latitude"] * LAT_TO_KM
    df["lon_to_km"] = df["longitude"] * LON_TO_KM

    # 6. Is superhost or not
    df["host_is_superhost"] = df["host_is_superhost"].replace({"t": 1, "f": 0})

    return df


def fit_and_save_preprocessor(X_train: pd.DataFrame) -> pd.DataFrame:
    """Fits KMeans, SimpleImputer and OneHotEncoder on X_train and saves them in models/"""
    X_train = X_train.copy()

    # 1. Clasterisation
    kmeans = KMeans(n_clusters=12, random_state=0)
    X_train["geo_cluster"] = kmeans.fit_predict(X_train[["lat_to_km", "lon_to_km"]])

    # Taking features that we need.
    all_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X_train = X_train[all_cols].copy()
    X_train[NUMERICAL_FEATURES] = X_train[NUMERICAL_FEATURES].astype("float")
    X_train[CATEGORICAL_FEATURES] = X_train[CATEGORICAL_FEATURES].astype("category")

    # 2. Imputing empty spaces
    imputer = SimpleImputer(strategy="most_frequent")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )

    # 3. One-Hot Encoding
    ohe = OneHotEncoder(
        drop="if_binary", sparse_output=False, handle_unknown="ignore"
    )
    ohe_Features = ohe.fit_transform(X_train_imputed[CATEGORICAL_FEATURES])
    ohe_cols = ohe.get_feature_names_out(CATEGORICAL_FEATURES)

    X_train_final = pd.concat(
        [
            pd.DataFrame(ohe_Features, columns=ohe_cols, index=X_train.index),
            X_train_imputed.drop(columns=CATEGORICAL_FEATURES),
        ],
        axis=1,
    ).astype("float32")

    # Replacing spaces so LightGBM wouldn't give us warnings
    X_train_final.columns = X_train_final.columns.str.replace(" ", "_")

    # Saving artifacts and preprocessing in one file
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {"kmeans": kmeans, "imputer": imputer, "ohe": ohe}
    joblib.dump(artifacts, MODELS_DIR / "preprocessor.joblib")
    print(f"Preprocessor is saved in {MODELS_DIR / 'preprocessor.joblib'}")

    return X_train_final


def transform_data(X: pd.DataFrame) -> pd.DataFrame:
    """Applies the saved KMeans, SimpleImputer and OneHotEncoder to test/new data"""
    artifacts_path = MODELS_DIR / "preprocessor.joblib"
    if not artifacts_path.exists():
        raise FileNotFoundError("Preprocessor isn't found. First start train.py")

    artifacts = joblib.load(artifacts_path)
    kmeans = artifacts["kmeans"]
    imputer = artifacts["imputer"]
    ohe = artifacts["ohe"]

    X = X.copy()
    coords = X[["lat_to_km", "lon_to_km"]].fillna(0)
    X["geo_cluster"] = kmeans.predict(coords)

    all_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = X[all_cols].copy()
    X[NUMERICAL_FEATURES] = X[NUMERICAL_FEATURES].astype("float")
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].astype("category")

    X_imputed = pd.DataFrame(
        imputer.transform(X), columns=X.columns, index=X.index
    )

    ohe_features = ohe.transform(X_imputed[CATEGORICAL_FEATURES])
    ohe_cols = ohe.get_feature_names_out(CATEGORICAL_FEATURES)

    X_final = pd.concat(
        [
            pd.DataFrame(ohe_features, columns=ohe_cols, index=X.index),
            X_imputed.drop(columns=CATEGORICAL_FEATURES),
        ],
        axis=1,
    ).astype("float32")

    X_final.columns = X_final.columns.str.replace(" ", "_")
    return X_final