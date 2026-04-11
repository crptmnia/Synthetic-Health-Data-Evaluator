import os

# Core app & UI
import dash
import base64
import io
from dash import dash_table
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# Data & math 
import numpy as np
import pandas as pd
import scipy.stats as stats

# Visualization 
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

# ML / metrics (future tabs)
from sklearn.metrics import mean_squared_error, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Parallelism
from multiprocessing import Pool
from itertools import repeat

#DPCM2 backend function
from dpcm2_backend import compute_dpcm2 

#Resemblance backend function
from resemblance_backend import compute_resemblance

#Utility backend function
from utility_backend import run_trtr, run_tstr, preprocess_dataframe 

# XAI (SHAP)
import shap
from xai_backend import load_model_from_pkl, generate_shap_explanations

# PDF imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from pdf_report import create_pdf_report


# ---------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)

app.title = "Synthetic Health Data Evaluation"

# ---------------------------------------------------------------------
# Layout components
# ---------------------------------------------------------------------

# moved up so it is always present
export_button = dbc.Button(
    "Export Results",
    id="export-button",
    color="warning",
    className="mt-3"
)

navbar = dbc.NavbarSimple(
    brand="Synthetic Health Data Evaluation",
    color="primary",
    dark=True,
    sticky="top",
    children=[export_button, dcc.Download(id="download-pdf")]
)

tabs = dbc.Tabs(
    [
        # Load Data Tab
        dbc.Tab(
            label="Load Data",
            tab_id="tab-load",
            children=[
                html.H4("Load Data Tab"),
                html.P("Upload a dataset and preview the first few rows. Select multiple (via Ctrl) for dataset comparisons.", style={'fontStyle': 'italic'}),
                html.P("Please note that the tests expect all values to be numerical and not categorical."),
                html.P("Supported formats: CSV, Excel (.xls/.xlsx)", style={'fontStyle': 'italic'}),

                dcc.Upload(
                    id='upload-data',
                    children=dbc.Button("Upload Dataset/s", color="primary"),
                    multiple=True
                ),

                html.Br(),
                html.Div(id='preview-tables'),
                html.Div(id='uploaded-filenames', style={'marginTop': '10px'}),

                # Hidden store to keep uploaded data for other tabs
                dcc.Store(id='stored-data')
            ]
        ),
        

        # DPCM2 Tab
        dbc.Tab(
            label="DPCM2",
            tab_id="tab-dpcm2",
            children=[
                html.H4("DPCM2 Tab"),
                html.P("Type 2 Diabetes Prevalence Consistency Measurement tests the prevalence of known characteristics of having type 2 diabetes between the original and synthetic dataset."),
                html.P("It is a supporting measure to check if the synthetic data captures important relationships in the original data, but it is not a comprehensive evaluation on its own."),
                html.P("Must upload one synthetic and one dataset for comparison."),
                dbc.Button("Run DPCM2 Evaluation", id='run-dpcm2', color="success", className="mt-3"),
                dbc.Button("Per attribute similarity", id='show-attributes', color="info", className="mt-3", style={"marginLeft": "10px"}),
                html.Div(id='dpcm2-results', style={'marginTop': '20px'}),
                html.Div(id="dpcm2-attributes", style={"marginTop": "20px"})
            ]
        ),

        # Resemblance Tab
        dbc.Tab(
            label="Resemblance",
            tab_id="tab-resemblance",
            children=[
                html.H4("Resemblance Tab"),
                html.P("This module evaluates how similar two groups of data are. In simple terms, it checks whether two patient populations or clinical variables “look alike” or behave differently."),
                html.P("It compares the overall pattern of values rather than individual patients. For example, it can assess whether age distributions, laboratory results, or risk scores from two hospitals follow similar trends or show meaningful differences."),
                html.P("Must upload one synthetic and one dataset for comparison"),
                dbc.Button("JS Similarity", id="run-js", color="primary", className="m-2"),
                dbc.Button("KS Comparison", id="run-ks", color="secondary", className="m-2"),
                dbc.Button("Wasserstein Distance", id="run-wasserstein", color="info", className="m-2"),
                html.Div(id="resemblance-results", style={"marginTop": "20px"})
            ]
        ),

        # Utility Tab
        dbc.Tab(
            label="Utility",
            tab_id="tab-utility",
            children=[
                html.H4("Utility Tab"),
                html.P("Evaluate how well synthetic data works in predictions."),

                # Step 1: Assign datasets
                html.H5("Step 1: Assign Synthetic and Real"),
                html.Div([
                dbc.Button("Assign Dataset 1 as Synthetic, Dataset 2 as Real", id="assign-1-2", color="primary", className="me-2"),
                dbc.Button("Assign Dataset 2 as Synthetic, Dataset 1 as Real", id="assign-2-1", color="secondary")
                ]),
            dcc.Store(id="store-assignments"),
            # Assignment indicator
            html.Div(id="assignment-indicator", className="mt-2"),

            html.Hr(),

            # Step 2: Run tests
            html.H5("Step 2: Run Utility Tests"),
            html.Div([
                dbc.Button("Run TSTR (Train Synthetic, Test Real)", id="run-tstr", color="success", className="me-2"),
                dbc.Button("Run TRTR (Train Real, Test Real)", id="run-trtr", color="info")
            ]),

            html.Hr(),

            # Results
            html.Div(id="utility-results")
            ]
        ),

        # XAI Tab
        dbc.Tab(
            label="XAI (SHAP)",
            tab_id="tab-xai",
            children=[
                html.H4("Explainable AI (XAI) Tab"),
                html.P("Upload trained model files (.pkl) and generate SHAP explanations."),
                html.P("SHAP tells us how each feature contributed to the predictions. Higher values means they contribute more towards the prediction."),
                html.P("Global Importance Pattern: X-axis are the SHAP values, which represent the impact of each feature on prediction, Y-axis: the list of features, with varying colors per dot which shows the dots value, Dots: each dot is one patient/sample"),
                html.P("Dependence plots: X-axis = feature values, Y-axis: SHAP values (impact on prediction)."),

        dcc.Upload(
            id="upload-model",
            children=html.Div(["Drag and Drop or ", html.A("Select a .pkl file")]),
            multiple=False,
            accept=".pkl",
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
        ),

            dcc.Store(id="stored-model"),

            html.Div(id="upload-status"),

            html.Hr(),

            dbc.Button(
                "Run SHAP Explanation",
                id="run-shap",
                color="primary",
                n_clicks=0
            ),

                html.Div(id="xai-results"),
            ]
        ),
    ]
)



app.layout = dbc.Container(
    [
        navbar,
        html.Br(),
        tabs,

        # Store the results
        dcc.Store(id="store-dpcm2-results"),
        dcc.Store(id="store-dpcm2-attributes"),
        dcc.Store(id="store-resemblance-results"),
        dcc.Store(id="store-utility-results"),
        dcc.Store(id="store-xai-results"),

        html.Hr(),
    ],
    fluid=True
)
# Load Data Tab Function Start ----------
@app.callback(
    [Output('stored-data', 'data'),
    Output('preview-tables', 'children'),
    Output('uploaded-filenames', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filenames):
    if contents is None:
        raise PreventUpdate

    datasets = []
    previews = []
    alerts = []

    for content, fname in zip(contents, filenames):
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)

        try:
            if fname.endswith('.csv'):
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif fname.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                alerts.append(dbc.Alert(f"Unsupported file type: {fname}", color="danger", dismissable=True))
                continue

            datasets.append({'filename': fname, 'data': df.to_dict('records')})

            previews.append(
    dbc.Card(
        [
            dbc.CardHeader(html.H5(f"Preview of {fname}", className="text-primary")),
            dbc.CardBody(
                dash_table.DataTable(
                    data=df.head().to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    page_size=5,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'}
                )
            )
        ],
        className="mb-4 shadow-sm"
    )
)

            alerts.append(dbc.Alert(f"Successfully uploaded: {fname}", color="success", dismissable=True))

        except Exception as e:
            fname = fname if fname else "Unknown file"
            alerts.append(dbc.Alert(f"Failed to upload {fname}: {str(e)}", color="danger", dismissable=True))

    return datasets, previews, alerts
# Load Data Tab Function End ------------

# DPCM2 Tab Function Start ----------
# Final Score callback Start ------
@app.callback(
    [Output('dpcm2-results', 'children'),
    Output('store-dpcm2-results', 'data')],
    Input('run-dpcm2', 'n_clicks'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def run_dpcm2(n_clicks, datasets):
    if not datasets or len(datasets) < 2:
        return (
            dbc.Alert("Please upload 2 datasets before running DPCM2.", color="warning"),
            None
        )

    try:
        results = compute_dpcm2(pd.DataFrame(datasets[0]['data']),
                                pd.DataFrame(datasets[1]['data']),
                                name1=datasets[0]['filename'],
                                name2=datasets[1]['filename'])
    except Exception as e:
        return (
            dbc.Alert(f"Error during DPCM2 computation: {str(e)}", color="danger"),
            None
        )

    return (
        html.Div([
            dbc.Alert(
                f"DPCM2 Score ({datasets[0]['filename']} vs {datasets[1]['filename']}): {results['dpcm2_final_score']:.2f}%",
                color="success"
            ),
            dbc.Alert(
                "Interpretation: A value close to 100% means the two datasets show very high similarity "
                "in Type 2 Diabetes prevalence patterns. Lower values indicate greater divergence.",
                color="info"
            )
        ]),
        {
            "title": "DPCM2 Results",
            "dpcm2_final_score": float(results['dpcm2_final_score'])
        }
    )

# Final Score callback End ------
# Attribute Scores callback Start ------
@app.callback(
    [Output("dpcm2-attributes", "children"),
    Output("store-dpcm2-attributes", "data", allow_duplicate=True)],
    Input("show-attributes", "n_clicks"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def show_attribute_scores(n_clicks, datasets):
    if not datasets or len(datasets) < 2:
        return (
            dbc.Alert("Please upload 2 datasets before running DPCM2.", color="warning"),
            None
        )

    try:
        results = compute_dpcm2(pd.DataFrame(datasets[0]["data"]),
                                pd.DataFrame(datasets[1]["data"]),
                                name1=datasets[0]["filename"],
                                name2=datasets[1]["filename"])
    except Exception as e:
        return (
            dbc.Alert(f"Error during DPCM2 computation: {str(e)}", color="danger"),
            None
        )

    scores = results["attribute_scores"]
    table = dbc.Table.from_dataframe(
        pd.DataFrame({
            "Attribute": list(scores.keys()),
            "Similarity (%)": [f"{v*100:.2f}" for v in scores.values()]
        }),
        striped=True, bordered=True, hover=True
    )

    return (
        html.Div([
            html.H6("Per-attribute similarity breakdown"),
            table
        ]),
        {
            "title": "DPCM2 Attribute Breakdown",
            "attributes": {k: float(v) for k, v in scores.items()}
        }
    )

# Attribute Scores callback End ------
# DPCM2 Tab Function End ------------

# Resemblance Tab Callback Start -------
@app.callback(
    [Output("resemblance-results", "children"),
    Output("store-resemblance-results", "data")],
    [Input("run-js", "n_clicks"),
    Input("run-ks", "n_clicks"),
    Input("run-wasserstein", "n_clicks")],
    State("stored-data", "data"),
    State("store-resemblance-results", "data"),
    prevent_initial_call=True
)
def run_resemblance(js_click, ks_click, w_click, datasets, previous_results):

    if not datasets or len(datasets) < 2:
        return (
            dbc.Alert("Please upload 2 datasets before running resemblance tests.", color="warning"),
            previous_results
        )

    previous_results = previous_results or {"title": "Resemblance Analysis"}

    df1 = pd.DataFrame(datasets[0]["data"])
    df2 = pd.DataFrame(datasets[1]["data"])
    fname1, fname2 = datasets[0]["filename"], datasets[1]["filename"]

    results = compute_resemblance(df1, df2, name1=fname1, name2=fname2)

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # --- JS Similarity ---
    if button_id == "run-js":

        scores = {col: metrics["JS Divergence"] for col, metrics in results["results"].items()}
        overall_js = results["overall"]["JS Divergence"]

        fig = px.bar(
            x=list(scores.keys()),
            y=list(scores.values()),
            labels={"x": "Attribute", "y": "JS Divergence (lower = more similar)"},
            title=f"JS Similarity ({fname1} vs {fname2})"
        )

        previous_results["js"] = float(overall_js)

        return (
            html.Div([
                dbc.Alert(
                    "Jensen-Shannon Similarity measures how similar the distribution shapes (how spread the values) are.",
                    color="info"
                ),
                html.P(f"Overall JS Divergence: {overall_js:.4f}", style={"fontWeight": "bold"}),
                dcc.Graph(figure=fig)
            ]),
            previous_results
        )

    # --- KS Comparison ---
    elif button_id == "run-ks":

        scores = {col: metrics["KS D-Statistic"] for col, metrics in results["results"].items()}
        overall_ks = results["overall"]["KS D-Statistic"]

        fig = px.bar(
            x=list(scores.keys()),
            y=list(scores.values()),
            labels={"x": "Attribute", "y": "KS D-Statistic (lower = more similar)"},
            title=f"KS Comparison ({fname1} vs {fname2})"
        )

        previous_results["ks"] = float(overall_ks)

        return (
            html.Div([
                dbc.Alert(
                    "Kolmogorov-Smirnov Statistic measures the maximum distribution difference (higher bar means higher mismatch/disagreement).",
                    color="info"
                ),
                html.P(f"Overall KS D-Statistic: {overall_ks:.4f}", style={"fontWeight": "bold"}),
                dcc.Graph(figure=fig)
            ]),
            previous_results
        )

    # --- Wasserstein Distance ---
    elif button_id == "run-wasserstein":

        scores = {col: metrics["Wasserstein"] for col, metrics in results["results"].items()}
        overall_w = results["overall"]["Wasserstein"]

        fig = px.bar(
            x=list(scores.keys()),
            y=list(scores.values()),
            labels={"x": "Attribute", "y": "Wasserstein Distance (lower = more similar)"},
            title=f"Wasserstein Distance ({fname1} vs {fname2})"
        )

        previous_results["wasserstein"] = float(overall_w)

        return (
            html.Div([
                dbc.Alert(
                    "Wasserstein Distance measures how much distributions must shift to match (higher bar means more effort needed to make the two match).",
                    color="info"
                ),
                html.P(f"Overall Wasserstein Distance: {overall_w:.4f}", style={"fontWeight": "bold"}),
                dcc.Graph(figure=fig)
            ]),
            previous_results
        )
# Resemblance Tab Callback End --------

# Utility Tab Callback Start -------
# Assignment Callback Start ------
@app.callback(
    [Output("store-assignments", "data"),
    Output("assignment-indicator", "children")],
    [Input("assign-1-2", "n_clicks"),
    Input("assign-2-1", "n_clicks")],
    State("stored-data", "data"),
    prevent_initial_call=True
)
def assign_datasets(assign12, assign21, datasets):
    ctx = dash.callback_context
    if not ctx.triggered or not datasets or len(datasets) < 2:
        return None, dbc.Alert("Please upload 2 datasets first.", color="warning")

    fname1, fname2 = datasets[0]["filename"], datasets[1]["filename"]
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "assign-1-2":
        assignments = {"synthetic": 0, "real": 1}
        message = dbc.Alert(f"{fname1} assigned as Synthetic, {fname2} assigned as Real.", color="info")
    else:
        assignments = {"synthetic": 1, "real": 0}
        message = dbc.Alert(f"{fname2} assigned as Synthetic, {fname1} assigned as Real.", color="info")

    return assignments, message
# Assignment Callback End ------

@app.callback(
    [Output("utility-results", "children"),
    Output("store-utility-results", "data")],
    [Input("run-tstr", "n_clicks"),
    Input("run-trtr", "n_clicks")],
    State("stored-data", "data"),
    State("store-assignments", "data"),
    State("store-utility-results", "data"),   # <-- add previous results
    prevent_initial_call=True
)
def run_utility(tstr_click, trtr_click, datasets, assignments, previous_results):
    if not datasets or len(datasets) < 2 or not assignments:
        return (
            dbc.Alert("Please upload 2 datasets and assign Synthetic/Real first.", color="warning"),
            previous_results
        )

    previous_results = previous_results or {}  # start with empty dict if none

    synth_idx, real_idx = assignments["synthetic"], assignments["real"]
    synth_df = pd.DataFrame(datasets[synth_idx]["data"])
    real_df = pd.DataFrame(datasets[real_idx]["data"])

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "run-tstr":
        metrics = run_tstr(synth_df, real_df)
        description = "TSTR: Train on synthetic data, test on real data. If results are similar to TRTR, then the synthetic data is realistic enough to train models without losing performance."
        previous_results["tstr"] = metrics
    else:
        metrics = run_trtr(real_df)
        description = "TRTR: Train and test on real data. This is the baseline, compare this to TSTR"
        previous_results["trtr"] = metrics

    df_metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Score"])
    fig = px.bar(df_metrics, x="Metric", y="Score",
                title="Utility Test Metrics",
                labels={"Score": "Value"},
                text=df_metrics["Score"].round(3))
    fig.update_traces(textposition="outside")

    return (
        html.Div([
            dbc.Alert(description, color="info"),
            dcc.Graph(figure=fig),
            html.H5("Metric Explanations"),
            html.Ul([
                html.Li("Accuracy: Overall correctness. Note that class imbalance may make this misleading."),
                html.Li("AUC: Ability to distinguish cases (positive vs. negative)."),
                html.Li("Recall: Ability to detect actual positive cases."),
                html.Li("Precision: True positives among predicted positives."),
                html.Li("F1-Score: Balance between precision and recall, provides an overall positive class measure.")
            ])
        ]),
        previous_results   # <-- merged results returned
    )

# Utility Tab Callback End --------

# XAI Upload Callback --------
@app.callback(
    [Output("stored-model", "data"),
    Output("xai-results", "children")],
    Input("upload-model", "contents"),
    State("upload-model", "filename"),
    prevent_initial_call=True
)
def store_uploaded_model(contents, filename):
    if contents is None:
        return (
            dbc.Alert("No file uploaded.", color="warning"),
            None
        )

    import base64
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    stored = {
        "filename": filename,
        "content": base64.b64encode(decoded).decode("utf-8")
    }

    # Confirmation message
    return stored, dbc.Alert(f"Model '{filename}' uploaded successfully.", color="info")

# XAI Tab Start -------
# Upload Callback
@app.callback(
    Output("stored-model", "data", allow_duplicate=True),
    Output("upload-status", "children"),
    Input("upload-model", "contents"),
    State("upload-model", "filename"),
    prevent_initial_call=True
)
def store_uploaded_model(contents, filename):
    if contents is None:
        return None, dbc.Alert("No file uploaded.", color="warning")

    content_type, content_string = contents.split(',')
    stored = {
        "filename": filename,
        "content": content_string
    }
    
    return stored, dbc.Alert(f"Model '{filename}' uploaded successfully.", color="info")

# Run SHAP Callback
@app.callback(
    [Output("xai-results", "children", allow_duplicate=True),
    Output("store-xai-results", "data")],
    Input("run-shap", "n_clicks"),
    State("stored-model", "data"),
    prevent_initial_call=True
)
def run_shap_explanation(n_clicks, model_data):
    if not model_data:
        return (
            dbc.Alert("Please upload a .pkl model file first.", color="warning"),
            None
        )
    
    # Decode and load pipeline
    pkl_bytes = base64.b64decode(model_data["content"])
    model = load_model_from_pkl(pkl_bytes)

    num_features = model.named_steps["preprocess"].transformers_[0][2]
    cat_features = model.named_steps["preprocess"].transformers_[1][2]
    expected_columns = list(num_features) + list(cat_features)

    X_sample = pd.DataFrame(np.random.rand(50, len(expected_columns)), columns=expected_columns)
    if "gender" in cat_features:
        X_sample["gender"] = np.random.choice(["1", "0"], size=50)

    plots = generate_shap_explanations(model, X_sample)

    dependence_images = []
    image_list = []   # collect base64 images for PDF

    # Add global importance plot
    image_list.append(plots["global_importance"])

    for key, img in plots.items():
        if key.startswith("dependence_"):
            feature_name = key.replace("dependence_", "")
            dependence_images.append(html.Div([
                html.H5(f"Dependence Plot: {feature_name}"),
                html.Img(src="data:image/png;base64," + img, style={"width": "80%"})
            ]))
            image_list.append(img)   # add dependence plot image

    return (
        html.Div([
            dbc.Alert("SHAP explanations generated successfully.", color="success"),
            html.H5("1. Global Importance Pattern"),
            html.Img(src="data:image/png;base64," + plots["global_importance"], style={"width": "80%"}),
            html.H5("2. Dependence Plots"),
            html.Div(dependence_images)
        ]),
        {
            "title": "XAI Results",
            "summary": "SHAP explanations generated successfully.",
            "images": image_list   # <-- stored for PDF
        }
    )

# XAI Tab End -------

# PDF Callback Start -------
import base64

@app.callback(
    Output("download-pdf", "data"),
    Input("export-button", "n_clicks"),
    State("store-dpcm2-results", "data"),
    State("store-dpcm2-attributes", "data"),
    State("store-resemblance-results", "data"),
    State("store-utility-results", "data"),
    State("store-xai-results", "data"),
    State("stored-data","data"),
    State("store-assignments", "data"),
    State("stored-model", "data"),
    prevent_initial_call=True
)
def export_pdf(n_clicks, dpcm2_results, dpcm2_attributes, resemblance, utility, xai, datasets, assignments, model_files):
    # Merge DPCM2 results
    dpcm2_data = {}
    if dpcm2_results:
        dpcm2_data.update(dpcm2_results)
    if dpcm2_attributes:
        dpcm2_data.update(dpcm2_attributes)

    # Merge Utility results
    utility_data = {}
    if utility:
        if "tstr" in utility:
            utility_data["tstr"] = utility["tstr"]
        if "trtr" in utility:
            utility_data["trtr"] = utility["trtr"]

    # Create PDF report
    pdf_bytes = create_pdf_report(
        dpcm2=dpcm2_data,
        resemblance=resemblance,
        utility=utility_data,
        xai=xai,
        datasets=datasets,
        assignments=assignments,
        models=model_files
    )

    # Encode to base64 for Dash
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

    return {
        "content": pdf_base64,
        "filename": "thesis_evaluation_report.pdf",
        "type": "application/pdf",
        "base64": True
    }

#PDF Callback End -----

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
