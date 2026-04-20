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


navbar = dbc.NavbarSimple(
    brand="Synthetic Health Data Evaluation",
    color="primary",
    dark=True,
    sticky="top",
    children=[
        html.Div(id="export-button-container"),  # placeholder
        dcc.Download(id="download-pdf")
    ]
)


# Welcome Tab defined globally
welcome_tab = dbc.Tab(
    label="Welcome",
    tab_id="tab-welcome",
    children=[
        html.H3("Synthetic Health Data Evaluator"),
        html.P("This dashboard provides an interactive framework for evaluating synthetic healthcare datasets. It integrates domain-specific checks, resemblance metrics, utility tests, and explainable AI visualizations."),
        html.P("To begin, please upload datasets in the Load Data tab. Clicking the Refresh button unlocks the evaluation modules (DPCM2, Resemblance, Utility, and XAI)."),
    ]
)

# Load Data Tab
load_data_tab = dbc.Tab(
    label="Load Data",
    tab_id="tab-load",
    children=[
        html.H4("Load Data Tab"),
        html.P("Upload a dataset and preview the first few rows. Select multiple (via Ctrl) for dataset comparisons.", style={'fontStyle': 'italic'}),
        html.P("Please note that the tests expect all values to be numerical and not categorical."),
        html.P("Caution: Clicking Refresh after all tabs are unlocked starts an entirely new session (datasets are deleted)"),
        html.P("Supported formats: CSV, Excel (.xls/.xlsx)", style={'fontStyle': 'italic'}),

        dcc.Upload(
            id='upload-data',
            children=dbc.Button("Upload Dataset/s", color="primary"),
            multiple=True
        ),
        dbc.Button("Refresh", id="unlock-tabs", color="primary", className="mt-3"),


        html.Br(),
        html.Div(id='preview-tables'),
        html.Div(id='uploaded-filenames', style={'marginTop': '10px'}),
    ]
)

# Initial Tabs container uses welcome_tab + load_data_tab
tabs_container = html.Div(
    id="tabs-container",
    children=dbc.Tabs([welcome_tab, load_data_tab])
)

# DPCM2 Tab
dpcm2_tab = dbc.Tab(
    label="DPCM2",
    tab_id="tab-dpcm2",
    children=[
        html.H4("DPCM2 Tab"),
        html.P("Type 2 Diabetes Prevalence Consistency Measurement tests the prevalence of known characteristics of having type 2 diabetes between the original and synthetic dataset."),
        html.P("It is a supporting measure to check if the synthetic data captures important relationships in the original data, but it is not a comprehensive evaluation on its own."),
        html.P("Must upload 2 datasets for comparison."),
        dbc.Button("Run DPCM2 Evaluation", id='run-dpcm2', color="success", className="mt-3"),
        dbc.Button("Per attribute similarity", id='show-attributes', color="info", className="mt-3", style={"marginLeft": "10px"}),
        html.Div(id='dpcm2-results', style={'marginTop': '20px'}),
        html.Div(id="dpcm2-attributes", style={"marginTop": "20px"})
    ]
)

# Resemblance Tab
resemblance_tab = dbc.Tab(
    label="Resemblance",
    tab_id="tab-resemblance",
    children=[
        html.H4("Resemblance Tab"),
        html.P("This module evaluates how similar two groups of data are. In simple terms, it checks whether two patient populations or clinical variables “look alike” or behave differently."),
        html.P("It compares the overall pattern of values rather than individual patients. For example, it can assess whether age distributions, laboratory results, or risk scores from two hospitals follow similar trends or show meaningful differences."),
        html.P("Must upload 2 datasets for comparison"),

        html.Div(
            [
                dbc.Button("JS Similarity", id="run-js", color="primary", className="m-2"),
                dbc.Button("KS Comparison", id="run-ks", color="secondary", className="m-2"),
                dbc.Button("Wasserstein Distance", id="run-wasserstein", color="info", className="m-2"),
                dcc.Loading(
                    id="loading-resemblance",
                    type="default",
                    children=html.Div(id="resemblance-results", style={"minWidth": "40px", "minHeight": "40px"})
                )
            ],
            style={"display": "flex", "alignItems": "center", "gap": "20px"}
        )
    ]
)




# Utility Tab
utility_tab = dbc.Tab(
    label="Utility",
    tab_id="tab-utility",
    children=[
        html.H4("Utility Tab"),
        html.P("Check how accurately synthetic data can be used for medical predictions compared to real data. This tests if models trained on synthetic data can perform well on real data, which is crucial for practical use."),

        # Step 1: Assign datasets
        html.H5("Step 1: Assign Synthetic and Real"),
        html.Div([
            dbc.Button("Assign Dataset 1 as Synthetic, Dataset 2 as Real", id="assign-1-2", color="primary", className="me-2"),
            dbc.Button("Assign Dataset 2 as Synthetic, Dataset 1 as Real", id="assign-2-1", color="secondary")
        ]),
        html.Div(id="assignment-indicator", className="mt-2"),

        html.Hr(),

        # Step 2: Run Utility Tests
        html.H5("Step 2: Run Utility Tests"),

        html.Div(
            [
                # Left button: TSTR
                dbc.Button(
                    "Run TSTR (Train Synthetic, Test Real)",
                    id="run-tstr",
                    color="success",
                    className="me-2"
                ),

                # Right button: TRTR + spinner beside it
                html.Div(
                    [
                        dbc.Button(
                            "Run TRTR (Train Real, Test Real)",
                            id="run-trtr",
                            color="info",
                            className="me-2"
                        ),
                        dcc.Loading(
                            id="loading-utility",
                            type="default",
                            children=html.Div(
                                id="utility-results",
                                style={"minWidth": "40px", "minHeight": "40px"}
                            )
                        )
                    ],
                    style={"display": "flex", "alignItems": "center", "gap": "10px"}
                )
            ],
            style={"display": "flex", "alignItems": "center", "gap": "20px"}
        )
    ]
)




# XAI Tab
xai_tab = dbc.Tab(
    label="XAI (SHAP)",
    tab_id="tab-xai",
    children=[
        html.H4("Explainable AI (XAI) Tab"),
        html.P("Upload trained model files (.pkl) and generate SHAP explanations."),
        html.P("SHAP tells us how each feature contributed to the model's predictions, helping us understand if the model is making decisions based on meaningful patterns or just noise."),
        html.P("Global Importance Pattern: Shows which features overall have the most influence on predictions. Dependence Plots: Show how specific feature values affect predictions, revealing if the model captures expected relationships (e.g., higher age increasing diabetes risk)."),

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

        html.Div(id="upload-status"),
        html.Hr(),

        # Button + Loading side by side
        html.Div(
            [
                dbc.Button("Run SHAP", id="run-shap"),
                dcc.Loading(
                    id="loading-xai",
                    type="default",
                    children=html.Div(id="xai-results"),
                )
            ],
            style={"display": "flex", "alignItems": "center", "gap": "20px"}
        ),
    ]
)



# When Refresh button is clicked, session memory should be gone
@app.callback(
    [
        Output("tabs-unlocked", "data"),
        Output("stored-data", "data", allow_duplicate=True),
        Output("store-dpcm2-results", "data", allow_duplicate=True),
        Output("store-dpcm2-attributes", "data", allow_duplicate=True),
        Output("store-resemblance-results", "data", allow_duplicate=True),
        Output("store-utility-results", "data", allow_duplicate=True),
        Output("store-xai-results", "data", allow_duplicate=True),
        Output("store-assignments", "data", allow_duplicate=True),
        Output("stored-model", "data", allow_duplicate=True)
    ],
    Input("unlock-tabs", "n_clicks"),
    prevent_initial_call=True
)
def unlock_tabs(n_clicks):
    if n_clicks:
        # Unlock tabs and reset everything
        return True, [], {}, {}, {}, {}, {}, {}, {}
    return False, [], {}, {}, {}, {}, {}, {}, {}



# Tab Unlocker
@app.callback(
    Output("tabs-container", "children"),
    Input("tabs-unlocked", "data")
)
def update_tabs(unlocked):
    if not unlocked:
        return dbc.Tabs(
            [welcome_tab, load_data_tab],
            active_tab="tab-welcome"   # <-- default selection
        )
    else:
        return dbc.Tabs(
            [load_data_tab, dpcm2_tab, resemblance_tab, utility_tab, xai_tab],
            active_tab="tab-load"      # <-- default selection after unlock
        )


#Layout
app.layout = dbc.Container(
    [
        navbar,
        html.Br(),
        tabs_container,

        # Store the results
        dcc.Store(id='stored-data', data = []), # moved here
        dcc.Store(id="tabs-unlocked", data=False),
        dcc.Store(id="store-dpcm2-results"),
        dcc.Store(id="store-dpcm2-attributes"),
        dcc.Store(id="store-resemblance-results"),
        dcc.Store(id="store-utility-results"),
        dcc.Store(id="store-xai-results"),
        dcc.Store(id="store-assignments"),
        dcc.Store(id="stored-model"),
        # Modal Callback for Error in Export Button
        dbc.Modal(
            [
                dbc.ModalHeader("Export Error"),
                dbc.ModalBody("No evaluation results found. Please run at least one test before exporting."),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-error", className="ms-auto", n_clicks=0)
                ),
            ],
            id="error-modal",
            is_open=False,
        ),

        # PDF error placeholder
        html.Div(id="export-error"),
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

    # Enforce max of 2 files
    if len(filenames) > 2:
        return (
            [],  # no datasets stored
            [],  # no previews shown
            dbc.Alert("Error: Maximum of 2 files allowed.",
                    color="danger", dismissable=True)
        )

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

# Resemblance Tab Callback A: run test
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
        return dbc.Alert("Please upload 2 datasets before running resemblance tests.", color="warning"), previous_results

    previous_results = previous_results or {"title": "Resemblance Analysis"}
    df1, df2 = pd.DataFrame(datasets[0]["data"]), pd.DataFrame(datasets[1]["data"])
    fname1, fname2 = datasets[0]["filename"], datasets[1]["filename"]
    results = compute_resemblance(df1, df2, name1=fname1, name2=fname2)

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "run-js":
        scores = {col: metrics["JS Divergence"] for col, metrics in results["results"].items()}
        overall_js = results["overall"]["JS Divergence"]
        previous_results["metric"] = "JS Divergence"
        previous_results["scores"] = scores
        previous_results["overall"] = float(overall_js)

        fig = px.bar(x=list(scores.keys()), y=list(scores.values()),
                    labels={"x": "Attribute", "y": "JS Divergence"},
                    title=f"JS Similarity ({fname1} vs {fname2})")

        return (
            html.Div([
                dbc.Alert("Jensen-Shannon Similarity: lower values mean the two groups spread out in similar ways.", color="info"),
                html.P(f"Overall JS Divergence: {overall_js:.4f}", style={"fontWeight": "bold"}),
                dcc.Graph(id="resemblance-graph", figure=fig),
                html.H6("Filter attributes by threshold:"),
                dcc.Slider(id="resemblance-threshold", min=0, max=1, step=0.05, value=0.2,
                        marks={0: "0", 0.5: "0.5", 1: "1"},
                        tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Button("Sort by Value", id="sort-button", color="dark", className="m-2"),  # <-- added here
                html.P("Low values mean the two datasets look alike for that specific column/attribute. High values mean they differ more.", style={"fontStyle": "italic"})
            ]),
            previous_results
        )

    elif button_id == "run-ks":
        scores = {col: metrics["KS D-Statistic"] for col, metrics in results["results"].items()}
        overall_ks = results["overall"]["KS D-Statistic"]
        previous_results["metric"] = "KS D-Statistic"
        previous_results["scores"] = scores
        previous_results["overall"] = float(overall_ks)

        fig = px.bar(x=list(scores.keys()), y=list(scores.values()),
                    labels={"x": "Attribute", "y": "KS D-Statistic"},
                    title=f"KS Comparison ({fname1} vs {fname2})")

        return (
            html.Div([
                dbc.Alert("Kolmogorov-Smirnov Statistic: lower values mean less disagreement between groups.", color="info"),
                html.P(f"Overall KS D-Statistic: {overall_ks:.4f}", style={"fontWeight": "bold"}),
                dcc.Graph(id="resemblance-graph", figure=fig),
                html.H6("Filter attributes by threshold:"),
                dcc.Slider(id="resemblance-threshold", min=0, max=1, step=0.05, value=0.2,
                        marks={0: "0", 0.5: "0.5", 1: "1"},
                        tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Button("Sort by Value", id="sort-button", color="dark", className="m-2"),
                html.P("Low values mean the two datasets look alike for that specific column/attribute. High values mean they differ more.", style={"fontStyle": "italic"})
            ]),
            previous_results
        )

    elif button_id == "run-wasserstein":
        scores = {col: metrics["Wasserstein"] for col, metrics in results["results"].items()}
        overall_w = results["overall"]["Wasserstein"]
        previous_results["metric"] = "Wasserstein Distance"
        previous_results["scores"] = scores
        previous_results["overall"] = float(overall_w)

        fig = px.bar(x=list(scores.keys()), y=list(scores.values()),
                    labels={"x": "Attribute", "y": "Wasserstein Distance"},
                    title=f"Wasserstein Distance ({fname1} vs {fname2})")

        return (
            html.Div([
                dbc.Alert("Wasserstein Distance: lower values mean less shifting needed to match distributions.", color="info"),
                html.P(f"Overall Wasserstein Distance: {overall_w:.4f}", style={"fontWeight": "bold"}),
                dcc.Graph(id="resemblance-graph", figure=fig),
                html.H6("Filter attributes by threshold:"),
                dcc.Slider(id="resemblance-threshold", min=0, max=5, step=0.1, value=1.0,
                        marks={0: "0", 2.5: "2.5", 5: "5"},
                        tooltip={"placement": "bottom", "always_visible": True}),
                dbc.Button("Sort by Value", id="sort-button", color="dark", className="m-2"),
                html.P("Low values mean the two datasets look alike for that specific column/attribute. High values mean they differ more.", style={"fontStyle": "italic"})
            ]),
            previous_results
        )


# Resemblance Tab Callback B: filter slider
@app.callback(
    Output("resemblance-graph", "figure", allow_duplicate=True),
    Input("resemblance-threshold", "value"),
    State("store-resemblance-results", "data"),
    prevent_initial_call=True
)
def update_resemblance_plot(threshold, stored_results):
    if not stored_results or "scores" not in stored_results:
        raise PreventUpdate

    scores = stored_results["scores"]
    metric = stored_results.get("metric", "")
    filtered_scores = {col: val for col, val in scores.items() if val < threshold}
    sorted_items = sorted(filtered_scores.items(), key=lambda x: x[1]) # automatically sorts lowest to highest from left to right

    fig = px.bar(x=list(filtered_scores.keys()), y=list(filtered_scores.values()),
                labels={"x": "Attribute", "y": f"{metric} (lower = more similar)"},
                title=f"Filtered {metric} Results")
    return fig

# Resemblance Tab Callback C: sort button
@app.callback(
    Output("resemblance-graph", "figure"),
    Input("sort-button", "n_clicks"),
    State("store-resemblance-results", "data"),
    State("resemblance-threshold", "value"),
    prevent_initial_call=True
)
def sort_resemblance_plot(sort_clicks, stored_results, threshold):
    if not stored_results or "scores" not in stored_results:
        raise PreventUpdate

    scores = stored_results["scores"]
    metric = stored_results.get("metric", "")

    # Apply threshold filter first
    filtered_scores = {col: val for col, val in scores.items() if val < threshold}

    # Sort by value ascending
    sorted_items = sorted(filtered_scores.items(), key=lambda x: x[1])

    fig = px.bar(
        x=[col for col, val in sorted_items],
        y=[val for col, val in sorted_items],
        labels={"x": "Attribute", "y": f"{metric} (lower = more similar)"},
        title=f"{metric} Results (sorted lowest → highest)"
    )
    return fig


# Resemblance Tab Callback End ---------

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

    # Collect images for PDF
    image_list = [plots["global_importance"]]
    dependence_dict = {}
    for key, img in plots.items():
        if key.startswith("dependence_"):
            feature_name = key.replace("dependence_", "")
            dependence_dict[feature_name] = img
            image_list.append(img)

    # Store everything for later use
    stored_data = {
        "title": "XAI Results",
        "summary": "SHAP explanations generated successfully.",
        "global_importance": plots["global_importance"],
        "dependence_plots": dependence_dict,
        "images": image_list
    }

    # Only show global importance + filter control initially
    return (
        html.Div([
            dbc.Alert("SHAP explanations generated successfully.", color="success"),
            html.H5("1. Global Importance Pattern"),
            html.Img(src="data:image/png;base64," + plots["global_importance"], style={"width": "80%"}),
            html.H5("2. Select an Attribute to View Dependence Plot"),
            dcc.Dropdown(
                id="xai-attribute-filter",
                options=[{"label": f, "value": f} for f in dependence_dict.keys()],
                placeholder="Choose an attribute..."
            ),
            html.Div(id="xai-filtered-plot", style={"marginTop": "20px"})
        ]),
        stored_data
    )


# Filter Callback
@app.callback(
    Output("xai-filtered-plot", "children"),
    Input("xai-attribute-filter", "value"),
    State("store-xai-results", "data"),
    prevent_initial_call=True
)
def show_dependence_plot(attribute, shap_results):
    if not attribute or not shap_results:
        return dbc.Alert("Select an attribute to view its dependence plot.", color="info")
    
    img = shap_results["dependence_plots"].get(attribute)
    if not img:
        return dbc.Alert("No plot available for this attribute.", color="warning")

    return html.Div([
        html.H5(f"Dependence Plot: {attribute}"),
        html.Img(src="data:image/png;base64," + img, style={"width": "80%"})
    ])


# XAI Tab End -------

# PDF Callback Start -------
import base64

@app.callback(
    [Output("download-pdf", "data"),
    Output("error-modal", "is_open", allow_duplicate=True)],
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

    if not n_clicks or n_clicks == 0:
        raise PreventUpdate

    if not (dpcm2_results or dpcm2_attributes or resemblance or utility or xai):
        return None, True

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

    # ✅ Extract only overall resemblance values
    resemblance_data = {}
    if resemblance:
        metric = resemblance.get("metric")
        overall = resemblance.get("overall")

        if metric and overall is not None:
            if "JS" in metric:
                resemblance_data["js"] = overall
            elif "KS" in metric:
                resemblance_data["ks"] = overall
            elif "Wasserstein" in metric:
                resemblance_data["wasserstein"] = overall

    # Create PDF report
    pdf_bytes = create_pdf_report(
        dpcm2=dpcm2_data,
        resemblance=resemblance_data,   # <-- only overall values
        utility=utility_data,
        xai=xai,
        datasets=datasets,
        assignments=assignments,
        models=model_files
    )

    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

    return {
        "content": pdf_base64,
        "filename": "thesis_evaluation_report.pdf",
        "type": "application/pdf",
        "base64": True
    }, False
# PDF Callback End -----



# Modal Callback for Error in Export Button
@app.callback(
    Output("error-modal", "is_open", allow_duplicate=True),
    Input("close-error", "n_clicks"),
    State("error-modal", "is_open"),
    prevent_initial_call=True
)
def close_modal(n_clicks, is_open):
    if n_clicks:
        return False
    return is_open

# PDF Button Appears after Refresh callback
@app.callback(
    Output("export-button-container", "children"),
    Input("tabs-unlocked", "data")
)
def toggle_export_button(unlocked):
    if unlocked:
        return dbc.Button(
            "Export Results",
            id="export-button",
            color="warning",
            className="mt-3"
        )
    return None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
