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

navbar = dbc.NavbarSimple(
    brand="Synthetic Health Data Evaluation",
    color="primary",
    dark=True,
    sticky="top"
)

tabs = dbc.Tabs(
    [
        # Load Data Tab
        dbc.Tab(
            label="Load Data",
            tab_id="tab-load",
            children=[
                html.H4("Load Data Tab"),
                html.P("Upload a dataset and preview the first few rows. Select multiple for multiple dataset comparisons.", style={'fontStyle': 'italic'}),
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
                html.P("Must upload one synthetic and one dataset for comparison"),
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
                html.P("Evaluate downstream task performance using synthetic vs real datasets."),

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

            dcc.Upload(
                id="upload-model",
                children=html.Div(["Drag and Drop or ", html.A("Select a .pkl file")]),
                multiple=False,
                accept=".pkl",
                style={
                    "width": "100%", "height": "60px", "lineHeight": "60px",
                    "borderWidth": "1px", "borderStyle": "dashed",
                    "borderRadius": "5px", "textAlign": "center", "margin": "10px"
                }
            ),

            dcc.Store(id="stored-model"),

            html.Hr(),

            dbc.Button("Run SHAP Explanation", id="run-shap", color="primary"),

            html.Div(id="xai-results")
            ]
        )

    ],
    id="main-tabs",
    active_tab="tab-load"
)

export_button = dbc.Button(
    "Export Results",
    id="export-button",
    color="warning",
    className="mt-3"
)

app.layout = dbc.Container(
    [
        navbar,
        html.Br(),

        tabs,

        html.Hr(),
        html.Div(
            export_button,
            style={"textAlign": "right"}
        )
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
    Output('dpcm2-results', 'children'),
    Input('run-dpcm2', 'n_clicks'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def run_dpcm2(n_clicks, datasets):
    if not datasets or len(datasets) < 2:
        return dbc.Alert("Please upload 2 datasets before running DPCM2.", color="warning")

    # Convert stored dicts back to DataFrames
    df1 = pd.DataFrame(datasets[0]['data'])
    df2 = pd.DataFrame(datasets[1]['data'])
    fname1 = datasets[0]['filename']
    fname2 = datasets[1]['filename']

    # Call backend function
    try:
        results = compute_dpcm2(df1, df2, name1=fname1, name2=fname2)
    except Exception as e:
        return dbc.Alert(f"Error during DPCM2 computation: {str(e)}", color="danger")

    # Final Score
    return html.Div([
        dbc.Alert(
            f"DPCM2 Score ({fname1} vs {fname2}): {results['dpcm2_final_score']:.2f}%",
            color = "success"
        ),
        dbc.Alert(
            "Interpretation: A value close to 100% means the two datasets show very high similarity "
            "in Type 2 Diabetes prevalence patterns. Lower calues indicate greater divergence.",
            color = "info"
        )
    ])
# Final Score callback End ------
# Attribute Scores callback Start ------
@app.callback(
    Output("dpcm2-attributes", "children"),
    Input("show-attributes", "n_clicks"),
    State("stored-data", "data"),
    prevent_initial_call=True
)
def show_attribute_scores(n_clicks, datasets):
    if not datasets or len(datasets) < 2:
        return dbc.Alert("Please upload 2 datasets before running DPCM2.", color="warning")

    df1 = pd.DataFrame(datasets[0]["data"])
    df2 = pd.DataFrame(datasets[1]["data"])
    fname1 = datasets[0]["filename"]
    fname2 = datasets[1]["filename"]

    try:
        results = compute_dpcm2(df1, df2, name1=fname1, name2=fname2)
    except Exception as e:
        return dbc.Alert(f"Error during DPCM2 computation: {str(e)}", color="danger")

    scores = results["attribute_scores"]

    # Format as a simple table
    table = dbc.Table.from_dataframe(
        pd.DataFrame({
            "Attribute": list(scores.keys()),
            "Similarity (%)": [f"{v*100:.2f}" for v in scores.values()]
        }),
        striped=True, bordered=True, hover=True
    )

    return html.Div([
        html.H6("Per‑attribute similarity breakdown"),
        table
    ])
# Attribute Scores callback End ------
# DPCM2 Tab Function End ------------

# Resemblance Tab Callback Start -------
@app.callback(
    Output("resemblance-results", "children"),
    [Input("run-js", "n_clicks"),
    Input("run-ks", "n_clicks"),
    Input("run-wasserstein", "n_clicks")],
    State("stored-data", "data"),
    prevent_initial_call=True
)
def run_resemblance(js_click, ks_click, w_click, datasets):
    if not datasets or len(datasets) < 2:
        return dbc.Alert("Please upload 2 datasets before running resemblance tests.", color="warning")

    df1 = pd.DataFrame(datasets[0]["data"])
    df2 = pd.DataFrame(datasets[1]["data"])
    fname1, fname2 = datasets[0]["filename"], datasets[1]["filename"]

    results = compute_resemblance(df1, df2, name1=fname1, name2=fname2)

    # Identify which button triggered
    ctx = dash.callback_context
    if not ctx.triggered:
        return dbc.Alert("No test selected.", color="warning")
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # --- JS Similarity ---
    if button_id == "run-js":
        scores = {col: metrics["JS Divergence"] for col, metrics in results["results"].items()}
        overall_js = results["overall"]["JS Divergence"]

        description = "JS Similarity: Measures how similar the overall shape of the data is. Lower values mean the datasets look more alike."
        y_label = "JS Divergence (lower = more similar)"
        title = f"JS Similarity ({fname1} vs {fname2})"

        fig = px.bar(x=list(scores.keys()), y=list(scores.values()),
                    labels={"x":"Attribute", "y":y_label}, title=title)

        return html.Div([
            dbc.Alert(description, color="info"),
            dcc.Graph(figure=fig),
            html.H6(f"Overall JS Divergence: {overall_js:.3f} (lower = more similar)")
        ])

    # --- KS Comparison ---
    elif button_id == "run-ks":
        d_scores = {col: metrics["KS D-Statistic"] for col, metrics in results["results"].items()}
        overall_ks = results["overall"]["KS D-Statistic"]

        description = (
            "KS Comparison: The Kolmogorov–Smirnov D-statistic measures the maximum difference between "
            "the two distributions. Lower values mean the datasets resemble each other more closely."
        )

        fig = px.bar(
            x=list(d_scores.keys()),
            y=list(d_scores.values()),
            labels={"x": "Attribute", "y": "KS D-Statistic (lower = more similar)"},
            title=f"KS Comparison ({fname1} vs {fname2})"
        )

        return html.Div([
            dbc.Alert(description, color="info"),
            dcc.Graph(figure=fig),
            html.H6(f"Overall KS D-Statistic: {overall_ks:.3f} (lower = more similar)")
        ])

    # --- Wasserstein Distance ---
    elif button_id == "run-wasserstein":
        scores = {col: metrics["Wasserstein"] for col, metrics in results["results"].items()}
        overall_w = results["overall"]["Wasserstein"]

        description = ("Wasserstein Distance: Measures how far values would need to shift to make the datasets match. "
                    "Lower values mean closer resemblance. Note: values depend on the scale of the variable "
                    "(e.g., years vs. blood pressure), so comparisons across attributes should be interpreted cautiously.")
        y_label = "Wasserstein Distance (lower = more similar)"
        title = f"Wasserstein Distance ({fname1} vs {fname2})"

        fig = px.bar(x=list(scores.keys()), y=list(scores.values()),
                    labels={"x":"Attribute", "y":y_label}, title=title)

        return html.Div([
            dbc.Alert(description, color="info"),
            dcc.Graph(figure=fig),
            html.H6(f"Overall Wasserstein Distance: {overall_w:.3f} (lower = more similar)")
        ])
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
    Output("utility-results", "children"),
    [Input("run-tstr", "n_clicks"),
    Input("run-trtr", "n_clicks")],
    State("stored-data", "data"),
    State("store-assignments", "data"),
    prevent_initial_call=True
)
def run_utility(tstr_click, trtr_click, datasets, assignments):
    if not datasets or len(datasets) < 2 or not assignments:
        return dbc.Alert("Please upload 2 datasets and assign Synthetic/Real first.", color="warning")

    synth_idx, real_idx = assignments["synthetic"], assignments["real"]
    synth_df = pd.DataFrame(datasets[synth_idx]["data"])
    real_df = pd.DataFrame(datasets[real_idx]["data"])

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "run-tstr":
        metrics = run_tstr(synth_df, real_df)
        description = ("TSTR: Train on synthetic data, test on real data. "
                    "This checks whether synthetic data can train a model "
                    "that generalizes well to real-world cases.")
    else:
        metrics = run_trtr(real_df)
        description = ("TRTR: Train and test on real data. "
                    "This is the baseline performance using only real data.")

    # Convert metrics to DataFrame
    df_metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Score"])

    # Bar chart visualization
    fig = px.bar(df_metrics, x="Metric", y="Score",
                title="Utility Test Metrics",
                labels={"Score": "Value"},
                text=df_metrics["Score"].round(3))
    fig.update_traces(textposition="outside")

    # Plain-language descriptions for each metric
    metric_explanations = html.Ul([
        html.Li("Accuracy: Overall correctness — proportion of all predictions that were right."),
        html.Li("AUC: Ability to distinguish between positive and negative cases (higher = better discrimination)."),
        html.Li("Recall: Sensitivity — how well the model detects actual positives."),
        html.Li("Precision: How many predicted positives were truly positive."),
        html.Li("F1-Score: Balance between precision and recall (harmonic mean).")
    ])

    return html.Div([
        dbc.Alert(description, color="info"),
        dcc.Graph(figure=fig),
        html.H5("Metric Explanations"),
        metric_explanations
    ])
# Utility Tab Callback End --------

# XAI Tab Callback Start --------
@app.callback(
    Output("xai-results", "children"),
    Input("run-shap", "n_clicks"),
    State("stored-model", "data"),
    prevent_initial_call=True
)

def store_uploaded_model(contents, filename):
    if contents is None:
        return None
    
    import base64 
    content_type, content_string = contents.split(',') 
    decoded = base64.b64decode(content_string) 
    
    # Store filename and raw content (encoded again for safe transport) return { "filename": filename, "content": base64.b64encode(decoded).decode("utf-8") }
    return{
        "filename": filename,
        "content": base64.b64encode(decoded).decode("utf-8")
    }

def run_shap_explanation(n_clicks, model_data):
    if not model_data:
        return dbc.Alert("Please upload a .pkl model file first.", color="warning")

    # Load model
    model = load_model_from_pkl(base64.b64decode(model_data["content"]))

    # Dummy sample data (replace with real test data if available)
    X_sample = pd.DataFrame(np.random.rand(50, model.n_features_in_),
                            columns=[f"Feature {i}" for i in range(model.n_features_in_)])

    plots = generate_shap_explanations(model, X_sample)

    # Display plots with captions
    return html.Div([
        dbc.Alert("SHAP explanations generated successfully.", color="success"),

        html.H5("1. Global Importance Pattern"),
        html.P("This bar chart shows which features are most influential overall in the model’s predictions."),
        html.Img(src="data:image/png;base64," + plots["global_importance"], style={"width":"80%"}),

        html.H5("2. Dependence Plot"),
        html.P("This scatter plot shows how a single feature’s values relate to its SHAP contribution, revealing feature patterns."),
        html.Img(src="data:image/png;base64," + plots["dependence"], style={"width":"80%"}),

        html.H5("3. Force Plot"),
        html.P("This visualization explains one individual prediction, showing how each feature pushed the prediction higher or lower."),
        html.Img(src="data:image/png;base64," + plots["force"], style={"width":"80%"})
    ])
# XAI Tab Callback End -------


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
