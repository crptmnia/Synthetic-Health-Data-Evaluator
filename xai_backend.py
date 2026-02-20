import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def load_model_from_pkl(pkl_bytes):
    """Load a model object from uploaded .pkl file bytes."""
    return pickle.loads(pkl_bytes)

def generate_shap_explanations(model, X_sample):
    """
    Generate SHAP explanations for a given model and sample data.
    Returns base64-encoded images for visualization.
    """
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    plots = {}

    # 1. Global importance pattern (summary bar plot)
    fig1 = plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    buf1 = io.BytesIO()
    plt.savefig(buf1, format="png", bbox_inches="tight")
    buf1.seek(0)
    plots["global_importance"] = base64.b64encode(buf1.read()).decode("utf-8")
    plt.close(fig1)

    # 2. Dependence plot (feature interaction pattern)
    fig2 = plt.figure()
    shap.dependence_plot(0, shap_values.values, X_sample, show=False)  # feature index 0 for demo
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png", bbox_inches="tight")
    buf2.seek(0)
    plots["dependence"] = base64.b64encode(buf2.read()).decode("utf-8")
    plt.close(fig2)

    # 3. Force plot (individual prediction explanation)
    fig3 = plt.figure()
    shap.force_plot(explainer.expected_value, shap_values.values[0,:], X_sample.iloc[0,:], matplotlib=True, show=False)
    buf3 = io.BytesIO()
    plt.savefig(buf3, format="png", bbox_inches="tight")
    buf3.seek(0)
    plots["force"] = base64.b64encode(buf3.read()).decode("utf-8")
    plt.close(fig3)

    return plots
