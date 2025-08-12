import os
import re
import pickle
import joblib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

DATA_FILE = "pet-records.csv"
MODEL_DIR = "saved_models"
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_FILE = os.path.join(MODEL_DIR, "features.pkl")
BREED_FREQ_FILE = os.path.join(MODEL_DIR, "breed_freq.pkl")
DL_MODEL_FILE = os.path.join(MODEL_DIR, "DeepLearning.h5")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


# ---------- Feature engineering ----------
def feature_engineering(df: pd.DataFrame, breed_freq_map: dict | None = None):
    df = df.copy()
    for col in ['petAge', 'petGender', 'petBreed', 'Note', 'PetCancerSuspect']:
        if col not in df.columns:
            df[col] = np.nan

    df['petAge'] = df['petAge'].fillna("Unknown")
    df['petGender'] = df['petGender'].fillna("Unknown")
    df['petBreed'] = df['petBreed'].fillna("Unknown")
    df['Note'] = df['Note'].fillna("")
    df['PetCancerSuspect'] = df['PetCancerSuspect'].fillna("No")

    def age_to_months(age_str):
        try:
            if not isinstance(age_str, str):
                return np.nan
            s = age_str.lower().strip()

            # match "3 years", "3y", "3 yr", "3yrs", etc.
            years = sum(int(x) for x in re.findall(r'(\d+)\s*(?:y|yr|yrs|year|years)\b', s))
            # match "2 months", "2m", "2 mo", "2mos", etc.
            months = sum(int(x) for x in re.findall(r'(\d+)\s*(?:m|mo|mos|month|months)\b', s))

            if years == 0 and months == 0:
                # try date formats mm/dd/yy, mm/dd/yyyy, yyyy-mm-dd
                for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
                    try:
                        birth_date = datetime.strptime(s, fmt)
                        today = datetime.today()
                        delta = today - birth_date
                        return int(delta.days // 30)
                    except Exception:
                        continue
                # fallback: just numbers = years
                nums = re.findall(r'\d+', s)
                if nums:
                    years = int(nums[0])

            return int(years * 12 + months)
        except Exception:
            return np.nan

    df['petAgeMonths'] = df['petAge'].apply(age_to_months).fillna(0).astype(int)

    if breed_freq_map is None:
        breed_freq_map = df['petBreed'].value_counts().to_dict()
    df['breed_freq'] = df['petBreed'].map(breed_freq_map).fillna(0).astype(int)

    df['is_mix_breed'] = df['petBreed'].str.lower().str.contains(r"mix|cross|mixed").fillna(False).astype(int)

    df['PetCancerSuspect'] = df['PetCancerSuspect'].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    df['note_length'] = df['Note'].apply(lambda x: len(str(x)))
    df['note_word_count'] = df['Note'].apply(lambda x: len(str(x).split()))
    keywords = ['tumor', 'swelling', 'cancer', 'mass', 'lump']
    df['note_keyword_count'] = df['Note'].apply(lambda x: sum(1 for kw in keywords if kw in str(x).lower()))

    gender_ohe = pd.get_dummies(df['petGender'].fillna("Unknown"), prefix='gender')
    df = pd.concat([df, gender_ohe], axis=1)

    features = [
        'petAgeMonths',
        'breed_freq',
        'is_mix_breed',
        'PetCancerSuspect',
        'note_length',
        'note_word_count',
        'note_keyword_count'
    ] + sorted([c for c in gender_ohe.columns])

    return df, features, breed_freq_map


# ---------- Training ----------
def train_models(show_progress=True):
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found.")

    df = pd.read_csv(DATA_FILE)
    if 'PetRiskValue' not in df.columns:
        raise ValueError("Data must contain 'PetRiskValue' column as the target.")

    df_proc, features, breed_freq_map = feature_engineering(df)
    X = df_proc[features].fillna(0)
    y = df_proc['PetRiskValue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "ElasticNet": ElasticNet(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(),
        "BaggingRegressor": BaggingRegressor(),
        "KNN": KNeighborsRegressor(),
        "XGBoost": XGBRegressor(eval_metric="rmse", use_label_encoder=False),
        "LightGBM": LGBMRegressor(),
        "CatBoost": CatBoostRegressor(verbose=0)
    }

    results = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_FILE)
    with open(FEATURES_FILE, "wb") as f:
        pickle.dump(features, f)
    with open(BREED_FREQ_FILE, "wb") as f:
        pickle.dump(breed_freq_map, f)

    for name, model in models.items():
        try:
            if show_progress:
                st.write(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, preds)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            results[name] = {"MAE": mae, "RMSE": rmse}
            joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
        except Exception as e:
            results[name] = {"error": str(e)}
            continue

    try:
        if show_progress:
            st.write("Training Deep Learning model...")
        dl_model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        dl_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
                         metrics=[tf.keras.metrics.MeanAbsoluteError()])
        es = EarlyStopping(patience=5, restore_best_weights=True)
        dl_model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
                     epochs=50, batch_size=16, callbacks=[es], verbose=0)
        dl_preds = dl_model.predict(X_test_scaled).flatten()
        mae = mean_absolute_error(y_test, dl_preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, dl_preds)))
        results["DeepLearning"] = {"MAE": mae, "RMSE": rmse}
        dl_model.save(DL_MODEL_FILE)
    except Exception as e:
        results["DeepLearning"] = {"error": str(e)}

    return results


# ---------- Prediction ----------
def predict_input(petAge, petGender, petBreed, PetCancerSuspect, Note, selected_models):
    if not (os.path.exists(FEATURES_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(BREED_FREQ_FILE)):
        st.error("Models are missing. Train models first.")
        return {}

    with open(FEATURES_FILE, "rb") as f:
        features = pickle.load(f)
    scaler = joblib.load(SCALER_FILE)
    with open(BREED_FREQ_FILE, "rb") as f:
        breed_freq_map = pickle.load(f)

    gender_map = {"M": "Male", "F": "Female", "Unknown": "Unknown"}
    petGender = gender_map.get(petGender, petGender)

    input_df = pd.DataFrame([{
        "petAge": petAge,
        "petGender": petGender,
        "petBreed": petBreed,
        "PetCancerSuspect": PetCancerSuspect,
        "Note": Note
    }])

    input_proc, _, _ = feature_engineering(input_df, breed_freq_map=breed_freq_map)

    X = input_proc.reindex(columns=features, fill_value=0)
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.error(f"Error scaling input: {e}")
        return {}

    predictions = {}
    for model_name in selected_models:
        if model_name == "DeepLearning":
            try:
                dl_model = load_model(DL_MODEL_FILE)
                pred = float(dl_model.predict(X_scaled).flatten()[0])
                predictions[model_name] = pred
            except Exception as e:
                st.warning(f"Failed to load DeepLearning model: {e}")
        else:
            model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
            try:
                model = joblib.load(model_path)
                pred = model.predict(X_scaled)
                if isinstance(pred, (list, np.ndarray)):
                    pred = float(np.array(pred).flatten()[0])
                predictions[model_name] = pred
            except Exception as e:
                predictions[model_name] = f"error: {str(e)}"
    return predictions


# ---------- Streamlit UI ----------
st.title("🐾 Pet Risk Value Prediction")

if os.path.exists(DATA_FILE):
    df_data = pd.read_csv(DATA_FILE)
    pet_gender_options = sorted(df_data['petGender'].dropna().unique()) if 'petGender' in df_data.columns else ["M", "F", "Unknown"]
    pet_breed_options = sorted(df_data['petBreed'].dropna().unique()) if 'petBreed' in df_data.columns else []
    cancer_suspect_options = sorted(df_data['PetCancerSuspect'].dropna().unique()) if 'PetCancerSuspect' in df_data.columns else ["Yes", "No"]
else:
    pet_gender_options = ["M", "F", "Unknown"]
    pet_breed_options = []
    cancer_suspect_options = ["Yes", "No"]

tab1, tab2 = st.tabs(["📊 Train Models", "🔮 Predict Risk"])

with tab1:
    if st.button("Train All Models"):
        with st.spinner("Training all models..."):
            try:
                results = train_models(show_progress=True)
                st.success("Training Completed!")
                st.dataframe(pd.DataFrame(results).T)
            except Exception as e:
                st.error(f"Training failed: {e}")

with tab2:
    petAge = st.text_input("Pet Age (e.g., '3 years 2 months', '3 y 2 m', or '08/06/21')")
    petGender = st.selectbox("Pet Gender", pet_gender_options)
    petBreed = st.selectbox("Pet Breed", pet_breed_options)
    PetCancerSuspect = st.selectbox("Cancer Suspect", cancer_suspect_options)
    Note = st.text_area("Notes", "")

    available_models = ["LinearRegression", "ElasticNet", "DecisionTree", "RandomForest",
                        "GradientBoosting", "BaggingRegressor", "KNN", "XGBoost",
                        "LightGBM", "CatBoost", "DeepLearning"]

    model_choice = st.multiselect("Select Models for Prediction (leave empty for all)", available_models)

    if st.button("Predict Risk Value"):
        if not model_choice:
            model_choice = available_models
        with st.spinner("Predicting..."):
            preds = predict_input(petAge, petGender, petBreed, PetCancerSuspect, Note, model_choice)
            if preds:
                df_out = pd.DataFrame.from_dict(preds, orient='index', columns=['Predicted Risk Value'])
                st.dataframe(df_out)
            else:
                st.info("No predictions returned.")

