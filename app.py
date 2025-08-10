import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import joblib
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime

DATA_FILE = "pet-records.csv"
MODEL_DIR = "saved_models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ===============================
# FEATURE ENGINEERING FUNCTION
# ===============================
def feature_engineering(df):
    for col in ['petAge', 'petGender', 'petBreed', 'Note']:
        df[col] = df[col].fillna("Unknown")

    def age_to_months(age_str):
        if isinstance(age_str, str):
            age_str = age_str.lower()
            years = sum(int(x) for x in re.findall(r'(\d+)\s*year', age_str))
            months = sum(int(x) for x in re.findall(r'(\d+)\s*month', age_str))
            if years == 0 and months == 0:
                try:
                    birth_date = datetime.strptime(age_str, '%m/%d/%y')
                except:
                    try:
                        birth_date = datetime.strptime(age_str, '%m/%d/%Y')
                    except:
                        birth_date = None

                if birth_date:
                    today = datetime.today()
                    delta = today - birth_date
                    months = delta.days // 30
                    return months

                nums = re.findall(r'\d+', age_str)
                if nums:
                    years = int(nums[0])
            return years * 12 + months
        return np.nan

    df['petAgeMonths'] = df['petAge'].apply(age_to_months)

    breed_freq = df['petBreed'].value_counts().to_dict()
    df['breed_freq'] = df['petBreed'].map(breed_freq)
    df['breed_freq'] = df['breed_freq'].fillna(0)

    df['is_mix_breed'] = df['petBreed'].str.lower().str.contains("mix|cross").astype(int)

    df['PetCancerSuspect'] = df['PetCancerSuspect'].map({"Yes": 1, "No": 0}).fillna(0)

    df['note_length'] = df['Note'].apply(lambda x: len(str(x)))
    df['note_word_count'] = df['Note'].apply(lambda x: len(str(x).split()))
    keywords = ['tumor', 'swelling', 'cancer', 'mass', 'lump']
    df['note_keyword_count'] = df['Note'].apply(lambda x: sum(kw in str(x).lower() for kw in keywords))

    gender_ohe = pd.get_dummies(df['petGender'], prefix='gender')
    df = pd.concat([df, gender_ohe], axis=1)

    features = [
        'petAgeMonths',
        'breed_freq',
        'is_mix_breed',
        'PetCancerSuspect',
        'note_length',
        'note_word_count',
        'note_keyword_count'
    ] + list(gender_ohe.columns)

    return df, features

# ===============================
# TRAINING FUNCTION
# ===============================
def train_models():
    df = pd.read_csv(DATA_FILE)
    df, features = feature_engineering(df)
    X = df[features].fillna(0)
    y = df['PetRiskValue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "ElasticNet": ElasticNet(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        "BaggingRegressor": BaggingRegressor(),
        "KNN": KNeighborsRegressor(),
        "XGBoost": XGBRegressor(),
        "LightGBM": LGBMRegressor(),
        "CatBoost": CatBoostRegressor(verbose=0)
    }

    results = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler and features
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "features.pkl"), "wb") as f:
        pickle.dump(features, f)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        results[name] = {"MAE": mae, "RMSE": rmse}
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

    # Deep Learning Model
    dl_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    dl_model.compile(
        optimizer='adam', 
        loss=tf.keras.losses.MeanSquaredError(), 
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    es = EarlyStopping(patience=5, restore_best_weights=True)
    dl_model.fit(
        X_train_scaled, y_train, 
        validation_data=(X_test_scaled, y_test),
        epochs=50, batch_size=16, callbacks=[es], verbose=0
    )
    dl_preds = dl_model.predict(X_test_scaled).flatten()
    mae = mean_absolute_error(y_test, dl_preds)
    rmse = mean_squared_error(y_test, dl_preds, squared=False)
    results["DeepLearning"] = {"MAE": mae, "RMSE": rmse}
    dl_model.save(os.path.join(MODEL_DIR, "DeepLearning.h5"))

    return results

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_input(petAge, petGender, petBreed, PetCancerSuspect, Note, selected_models):
    features_path = os.path.join(MODEL_DIR, "features.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not os.path.exists(features_path) or not os.path.exists(scaler_path):
        st.error("Models are not trained yet. Please go to 'Train Models' tab and train the models first.")
        return {}

    gender_map = {"M": "Male", "F": "Female", "Unknown": "Unknown"}
    petGender = gender_map.get(petGender, "Unknown")

    df = pd.DataFrame([{
        "petAge": petAge,
        "petGender": petGender,
        "petBreed": petBreed,
        "PetCancerSuspect": PetCancerSuspect,
        "Note": Note
    }])
    df, _ = feature_engineering(df)

    with open(features_path, "rb") as f:
        features = pickle.load(f)

    X = df.reindex(columns=features, fill_value=0)

    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    predictions = {}
    for model_name in selected_models:
        if model_name == "DeepLearning":
            from tensorflow.keras.models import load_model
            dl_model = load_model(os.path.join(MODEL_DIR, "DeepLearning.h5"))
            pred = dl_model.predict(X_scaled).flatten()[0]
        else:
            model = joblib.load(os.path.join(MODEL_DIR, f"{model_name}.pkl"))
            pred = model.predict(X_scaled)[0]
        predictions[model_name] = pred
    return predictions

# ===============================
# STREAMLIT UI
# ===============================
st.title("🐾 Pet Risk Value Prediction")

tab1, tab2 = st.tabs(["📊 Train Models", "🔮 Predict Risk"])

with tab1:
    if st.button("Train All Models"):
        st.write("Training models... please wait")
        results = train_models()
        st.success("Training Completed!")
        st.write(pd.DataFrame(results).T)

with tab2:
    petAge = st.text_input("Pet Age (e.g., '3 years 2 months' or '8/6/21')")
    petGender = st.selectbox("Pet Gender", ["M", "F", "Unknown"])
    petBreed = st.text_input("Pet Breed")
    PetCancerSuspect = st.selectbox("Cancer Suspect", ["Yes", "No"])
    Note = st.text_area("Notes")

    available_models = ["LinearRegression", "ElasticNet", "DecisionTree", "RandomForest",
                        "GradientBoosting", "BaggingRegressor", "KNN", "XGBoost",
                        "LightGBM", "CatBoost", "DeepLearning"]

    model_choice = st.multiselect("Select Models for Prediction (leave empty for all)",
                                  available_models)

    if st.button("Predict Risk Value"):
        if not model_choice:
            model_choice = available_models
        predictions = predict_input(petAge, petGender, petBreed, PetCancerSuspect, Note, model_choice)
        if predictions:
            st.write(pd.DataFrame.from_dict(predictions, orient='index', columns=['Predicted Risk Value']))
