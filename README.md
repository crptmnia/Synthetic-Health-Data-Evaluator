Synthetic Health Data Evaluation

Overview
This project provides an interactive dashboard for evaluating **synthetic health datasets** against real-world data. It was developed as part of a thesis to assess the quality, utility, and explainability of synthetic data in healthcare research. The tool is built using **Python, Dash, Plotly, and ReportLab**, with backend modules for statistical testing, machine learning evaluation, and explainable AI.

The goal is to give researchers and practitioners a structured way to:
- Compare synthetic datasets with original datasets.
- Evaluate statistical resemblance and prevalence consistency.
- Test downstream utility of synthetic data in predictive modeling.
- Generate explainable AI (SHAP) plots for model interpretation.
- Export results into a professional PDF report.

Functionalities

1. Load Data
- Upload datasets in **CSV or Excel (.xls/.xlsx)** format.
- Preview the first few rows of each dataset.
- Supports multiple dataset uploads for comparison.

2. DPCM2 Evaluation
- **Type 2 Diabetes Prevalence Consistency Measurement (DPCM2)**.
- Compares prevalence patterns between real and synthetic datasets.
- Provides an overall similarity score and per-attribute breakdown.
- Higher scores = stronger similarity.

3. Resemblance Analysis
- Statistical tests to measure distribution similarity:
  - **Jensen–Shannon Divergence (JS)** → lower = better.
  - **Kolmogorov–Smirnov D-Statistic (KS)** → lower = better.
  - **Wasserstein Distance (W)** → lower = better.
- Visualizes results with bar charts.

4. Utility Evaluation
- Assign datasets as **Synthetic** or **Real**.
- Run:
  - TSTR (Train Synthetic, Test Real)** → checks generalization of synthetic-trained models.
  - TRTR (Train Real, Test Real)** → baseline performance.
- Reports metrics: Accuracy, AUC, Recall, Precision, F1-Score.
- Preserves both TSTR and TRTR results for export.

5. Explainable AI (XAI)
- Upload trained model files (`.pkl`).
- Generate **SHAP explanations**:
  - Global importance plots.
  - Dependence plots per feature.
- Plots are stored and embedded into the PDF report.

6. PDF Export
- Consolidates all results into a single **evaluation report**.
- Includes:
  - Dataset filenames and assignments (Synthetic vs Real).
  - Model filename used.
  - DPCM2 results.
  - Resemblance scores.
  - Utility metrics.
  - SHAP plots.
- Provides a timestamped, professional PDF for documentation.


Goal
The project’s primary goal is to evaluate the reliability and usability of synthetic health data in research and clinical contexts. By combining statistical resemblance, prevalence consistency, predictive utility, and explainability, the tool helps determine whether synthetic datasets can serve as valid substitutes for real patient data while preserving privacy.

