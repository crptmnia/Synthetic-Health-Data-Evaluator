from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
import numpy as np

def preprocess_dataframe(df, target_col="diabetic"):
    df = df.copy()

    # Binary conversion for gender column
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    return df

def run_tstr(synth_df, real_df, target_col="diabetic"):
    # Preprocess both datasets
    synth_df = preprocess_dataframe(synth_df, target_col)
    real_df = preprocess_dataframe(real_df, target_col)

    # Separate features and labels
    X_train, y_train = synth_df.drop(columns=[target_col]), synth_df[target_col]
    X_test, y_test = real_df.drop(columns=[target_col]), real_df[target_col]

    # Keep only numeric columns
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test[X_train.columns]  # align columns

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }

def run_trtr(real_df, target_col="diabetic"):
    # Preprocess dataset
    real_df = preprocess_dataframe(real_df, target_col)

    # Separate features and labels
    X = real_df.drop(columns=[target_col])
    y = real_df[target_col]

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }
