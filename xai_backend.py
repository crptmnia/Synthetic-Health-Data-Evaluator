import joblib
import io
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def load_model_from_pkl(pkl_bytes):
    buffer = io.BytesIO(pkl_bytes)
    return joblib.load(buffer)

def generate_shap_explanations(model, X_sample, n_rows=50):

    X_used = X_sample.iloc[:n_rows, :]

    # -------------------------------
    # Handle Pipeline models properly
    # -------------------------------
    if isinstance(model, Pipeline):
        preprocessor = model.named_steps["preprocess"]
        final_model = model.named_steps["model"]

        # Transform data the same way training did
        X_transformed = preprocessor.transform(X_used)

        # Get transformed feature names (important!)
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    else:
        final_model = model
        X_transformed = X_used.values
        feature_names = X_used.columns

    # -------------------------------
    # Select proper SHAP explainer
    # -------------------------------

    # 1. For Random Forest
    if isinstance(final_model, RandomForestClassifier):

        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

    # 2. For XGBoost
    elif isinstance(final_model, xgb.XGBClassifier):
        
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

    # 3. For Logistic Regression
    elif isinstance(final_model, LogisticRegression):
        explainer = shap.LinearExplainer(final_model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    else:
        explainer = shap.Explainer(final_model, X_transformed)
        shap_values = explainer(X_transformed).values

    plots = {}

    # -------------------------------
    # Global SHAP summary
    # -------------------------------
    plt.figure()
    shap.summary_plot(
        shap_values,
        features=X_transformed,
        feature_names=feature_names,
        show=False
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plots["global_importance"] = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # -------------------------------
    # Dependence plots
    # -------------------------------
    for i in range(len(feature_names)):
        plt.figure()
        shap.dependence_plot(
            i,
            shap_values,
            X_transformed,
            feature_names=feature_names,
            interaction_index=None,
            show=False
        )
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plots[f"dependence_{feature_names[i]}"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

    return plots