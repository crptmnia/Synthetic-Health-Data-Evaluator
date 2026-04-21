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


Installation & Setup

Prerequisites
Before running the application, please make sure you have the following installed on your system:
- [Anaconda](https://www.anaconda.com/download) or Miniconda
- Python 3.9+
- Git

### Steps
1. Clone the repository.
   Open a terminal and run:
   ```bash
   git clone https://github.com/yourusername/synthetic-health-evaluator.git
   cd synthetic-health-data-evaluator
2. Create the environment  
   Instead of manually pinning versions, this project uses a `requirements.txt` file to auto‑detect and install the correct versions:
   ```bash
   conda create -n thesis_env 
   conda activate thesis_env
   pip install -r requirements.txt
3. Run the dashboard
   Start the application (on any terminal )via:
   ```bash
   python thesis_frontend.py
5. Open your browser and access the app via:
   http://127.0.0.1:8050/


References & Inspiration

This thesis project was inspired by and builds upon the work of Santangelo et al. (2025):

> Santangelo, G., et al. **How good is your synthetic data? SynthRO, a dashboard to evaluate and benchmark synthetic tabular data**.
> Published in **National Library of Medicine**.  
> [PMC Article Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC11837667)

Their study provided the foundational concepts and motivation for evaluating synthetic health data, particularly in terms of **data resemblance** and **utility**. This project adapts and extends those ideas into a practical evaluation framework and interactive dashboard.  

A key contribution of this thesis is the addition of an **Explainable AI (XAI) module using SHAP**, which was not part of Santangelo et al.’s original work. This enhancement allows users to interpret predictive models trained on synthetic data, offering transparency into feature importance and model behavior, thereby strengthening the evaluation framework beyond resemblance and utility alone.



