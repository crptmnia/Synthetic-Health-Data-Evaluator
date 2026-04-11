import pandas as pd
import numpy as np

def compute_dpcm2(df1: pd.DataFrame, df2: pd.DataFrame, name1: str = "Dataset1", name2: str = "Dataset2") -> dict:
    """
    Compute the Type 2 Diabetes Prevalence Consistency Measurement (DPCM2).

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataset with required columns.
    df2 : pd.DataFrame
        Second dataset with required columns.
    name1 : str
        Label for the first dataset (e.g., filename).
    name2 : str
        Label for the second dataset (e.g., filename).

    Returns
    -------
    dict
        {
            "dpcm2_final_score": float  # percentage 1–100
        }
    """
# Hardcoded column names as this measurement is specific for this study
    required_cols = ["age", "bmi", "systolic_bp", "diastolic_bp",
                    "family_diabetes", "family_hypertension", "diabetic"]
    for col in required_cols:
        if col not in df1.columns or col not in df2.columns:
            raise ValueError(f"Missing required column: {col}")

    rules = {
        "age": lambda x: (x >= 38) & (x <= 50),
        "bmi": lambda x: (x < 18.5) | (x > 22.9),
        "systolic_bp": lambda x: x > 130,
        "diastolic_bp": lambda x: x > 90,
        "family_diabetes": lambda x: x == 1,
        "family_hypertension": lambda x: x == 1
    }

    weights = {
        "age": 0.25,
        "bmi": 0.25,
        "systolic_bp": 0.10,
        "diastolic_bp": 0.10,
        "family_diabetes": 0.20,
        "family_hypertension": 0.10
    }

    def prevalence(df, col, rule):
        follower_mask = rule(df[col])
        non_mask = ~follower_mask
        p_follower = df.loc[follower_mask, "diabetic"].mean() if follower_mask.any() else 0
        p_non = df.loc[non_mask, "diabetic"].mean() if non_mask.any() else 0
        return p_follower, p_non

    scores = {}
    for col, rule in rules.items():
        p_f1, p_nf1 = prevalence(df1, col, rule)
        p_f2, p_nf2 = prevalence(df2, col, rule)

        delta1 = p_f1 - p_nf1
        delta2 = p_f2 - p_nf2

        score = max(0, 1 - abs(delta1 - delta2))
        scores[col] = score

    weighted_sum = sum(scores[col] * weights[col] for col in scores)
    dpcm2_final_score = weighted_sum * 100

    return {
        "dpcm2_final_score": dpcm2_final_score,
        "attribute_scores": scores
    }
