# Run this app with `python main.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html, Input, Output, dash_table
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

import plotly.express as px

from dashboard_apply_ml.models import *

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash(__name__, external_stylesheets=external_stylesheets)

df_chronofast = pd.read_csv('insert_path_to_chronofast_here', sep='\t')

df_parofastin = pd.read_csv('inser_path_to_parofastin_here', sep='\t')

###########################################################
# Some pre-processing

df_chronofast = clean_acceleration_data(df_chronofast)

df_chronofast = preprocess_chronofast_data(df_chronofast)

df_parofastin = preprocess_parofastin_data(df_parofastin)

############################################################

# Split data for frontend depending on intervention phase

df_screening = df_chronofast[(df_chronofast['phase'] == "screening")]
df_eTRE = df_chronofast[(df_chronofast['phase'] == "eTRE")]
df_lTRE = df_chronofast[(df_chronofast['phase'] == "lTRE")]

df_parofastin_screening = df_parofastin[(df_parofastin["phase"] == "V1")]
df_parofastin_intervention = df_parofastin[(df_parofastin["phase"] == "V2")]

options_glucose = list(df_chronofast.id.unique())
options_glucose_parofastin = list(df_parofastin.id.unique())

colors = ['#BB2E10', '#3498DB']

colors = {"graphBackground": "#F5F5F5", "background": "#ffffff", "text": "#000000"}

tab_style = {
    'borderTop': '5px solid #5DADE2',
    'fontWeight': 'bold',
    "primary": "#5DADE2",
    "background": "#5DADE2"
}
tab_selected_style = {
    'borderTop': '5px solid #2874A6',
    'fontWeight': 'bold',
    # 'backgroundColor': '#2874A6',
    "primary": "#5DADE2",
    "background": "#5DADE2"
}

tab_style_data = {
    'borderTop': '5px solid #EC7063',
    'fontWeight': 'bold',
    "primary": "#EC7063",
    "background": "#EC7063"
}
tab_selected_style_data = {
    'borderTop': '5px solid #B03A2E',
    'fontWeight': 'bold',
    # 'backgroundColor': '#B03A2E',
    "primary": "#EC7063",
    "background": "#EC7063"
}

tab_style_about = {
    'borderTop': '5px solid #9eaab6',
    'fontWeight': 'bold',
    "primary": "#9eaab6",
    "background": "#9eaab6"
}
tab_selected_style_about = {
    'borderTop': '5px solid #778899',
    'fontWeight': 'bold',
    # 'backgroundColor': '#B03A2E',
    "primary": "#9eaab6",
    "background": "#9eaab6"
}

app.layout = html.Div(children=[
    html.H1(children='Fasting State Prediction: Visualization and Configuration Tool'),

    html.Div([
        dcc.Tabs([
            dcc.Tab(label='About', id="about_tab", style=tab_style_about, selected_style=tab_selected_style_about,
                    children=[
                        html.Div([
                            html.H6(children='About this project: '),
                            html.P(
                                'This project was implemented as part of the master`s thesis at the Hasso Plattner '
                                'Institute. The goal of the project is to build a prototype that classifies the fasting '
                                'state of input data. Training data for the machine and time series classification'
                                ' models was the ChronoFast study conducted by Deutsches Institut für '
                                'Ernährungsforschung Potsdam-Rehbrücke (DIfE).'),
                            html.P('Author: Tanja Manlik'),
                            html.P('Advisors of the master thesis are:'),
                            html.Li('Prof. Dr. Bert Arnrich'),
                            html.Li('Dr. Nico Steckhan'),
                            html.Li('Dr. Olga Ramich')
                        ], style={'max-width': '800px', 'margin': '0 auto', 'vertical-align': 'middle'})
                    ]),
            dcc.Tab(label='ChronoFast Data (Training data)', style=tab_style_data,
                    selected_style=tab_selected_style_data, children=[
                    html.H4(children='Screening Data'),
                    html.Div([
                        dcc.Dropdown(options_glucose, id='pandas-dropdown-1'),
                        html.Div(id='pandas-output-container-1')
                    ]),
                    dcc.Graph(id="scatter-plot-screening"),
                    dcc.Graph(id="scatter-plot-act-screening"),
                    html.H4(children='Early-Time-Restricted-Eating Data'),
                    html.Div([
                        dcc.Dropdown(options_glucose, id='pandas-dropdown-2'),
                        html.Div(id='pandas-output-container-2')
                    ]),
                    dcc.Graph(id="scatter-plot-eTRE"),
                    dcc.Graph(id="scatter-plot-act-eTRE"),
                    html.H4(children='Late-Time-Restricted-Eating Data'),
                    html.Div([
                        dcc.Dropdown(options_glucose, id='pandas-dropdown-3'),
                        html.Div(id='pandas-output-container-3')
                    ]),
                    dcc.Graph(id="scatter-plot-lTRE"),
                    dcc.Graph(id="scatter-plot-act-lTRE")
                ]),
            dcc.Tab(label='ParoFastin Data', style=tab_style_data, selected_style=tab_selected_style_data, children=[
                html.H4(children='Bahai Screening data'),
                html.Div([
                    dcc.Dropdown(options_glucose_parofastin, id='pandas-dropdown-4'),
                    html.Div(id='pandas-output-container-4')
                ]),
                dcc.Graph(id="scatter-plot-bahai-screening"),
                html.H4(children='Bahai Intervention data'),
                html.Div([
                    dcc.Dropdown(options_glucose_parofastin, id='pandas-dropdown-5'),
                    html.Div(id='pandas-output-container-5')
                ]),
                dcc.Graph(id="scatter-plot-bahai-intervention"),
                html.H4(children='16:8 Screening data'),
                html.Div([
                    dcc.Dropdown(options_glucose_parofastin, id='pandas-dropdown-11'),
                    html.Div(id='pandas-output-container-11')
                ]),
                dcc.Graph(id="scatter-plot-168-screening"),
                html.H4(children='16:8 Intervention data'),
                html.Div([
                    dcc.Dropdown(options_glucose_parofastin, id='pandas-dropdown-12'),
                    html.Div(id='pandas-output-container-12')
                ]),
                dcc.Graph(id="scatter-plot-168-intervention")
            ]),
            dcc.Tab(label='Prediction Hutchison', style=tab_style, selected_style=tab_selected_style, children=[
                html.Div([
                    html.H4(children='Description of the Hutchison Method'),
                    html.P(
                        'The Hutchison method is for CGM analysis where the fed and fasting windows are determined '
                        'from meal log data. The “fasting” window is defined by 4 hours after the latest occasion a '
                        'participant ate in a given condition until the earliest occasion a participant ate in that '
                        'condition. The mean fasting glucose is calculated over this time window and across all days '
                        'the participant spent in that condition. The “fed” window is calculated from the earliest '
                        'eating occasion until the latest meal occasion, plus 4 hours to allow for postprandial '
                        'fluctuations in blood glucose." '),
                    html.P(
                        '- Hutchison AT, Regmi P, Manoogian ENC, Fleischer JG, Wittert GA, Panda S, Heilbronn LK. '
                        'Time-Restricted Feeding Improves Glucose Tolerance in Men at Risk for Type 2 Diabetes: A '
                        'Randomized Crossover Trial. Obesity (Silver Spring). 2019 May;27(5):724-732.'
                        ' doi: 10.1002/oby.22449.')
                ], style={'max-width': '1200px', 'margin': '0 auto', 'vertical-align': 'middle'}),
                html.Div([
                    html.H4(children='Upload files for Hutchison classification'),
                    html.P(
                        'Two files are needed for the hutchison classification. Please select both files and press '
                        '"open".'),
                    html.Li(
                        'CGM data file with the columns: "id" for participant id, "phase" for trial condition phase,'
                        ' "gl" for glucose in (mg/dL), optional: "fasting_state" as categorical variable used for '
                        'compliance'),
                    html.Li(
                        'Meal data log: "id" for participant id, "phase" for trial condition phase, "Beginn erste '
                        'Mahlzeit" for start of meal time, "Beginn letzte Mahlzeit" for end of meal time')
                ], style={'max-width': '1200px', 'margin': '0 auto', 'vertical-align': 'middle'}),
                dcc.Upload(
                    id="upload-data-hutchison",
                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
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
                    # Allow multiple files to be uploaded
                    multiple=True,
                ),
                dcc.Graph(id="Mygraph-hutchison"),
                html.H4(children='The compliance score for the selected data is: '),
                html.Div(id='hutchison-output-compliance-score'),
                html.Div(id="output-data-upload-hutchison")

            ]),
            dcc.Tab(label='Prediction with Machine Learning Models', style=tab_style, selected_style=tab_selected_style,
                    children=[
                        dcc.Tabs(id="subtabs_ml", value='Upload_ml_tab', children=[
                            dcc.Tab(label='Data Upload', id="upload_ml_tab", value="Upload_ml_tab", style=tab_style,
                                    selected_style=tab_selected_style, children=[
                                    dcc.Store(id='ml_data'),
                                    dcc.Store(id='actual_ml_data'),
                                    html.Div([
                                        html.H4(children='Select options for machine learning classification:'),
                                        "Inserted data contains:",
                                        dcc.RadioItems(['Glucose and Acceleration', 'Glucose', 'Acceleration'],
                                                       'Glucose and Acceleration', id='ml_data_options'),
                                        html.H6(children=''),
                                        dcc.Checklist(['Postprocessing and smoothing data (Recommended)'],
                                                      ['Postprocessing and smoothing data (Recommended)'],
                                                      id='ml_smoothing_checklist'),
                                        html.H6(children=''),
                                        "Length of each window (in samples) e.g. 3 equals 3*3 samples per 15 minutes "
                                        "= 45 min:",
                                        dcc.Dropdown(['2', '3', '4', '5', '6', '7', '8'], '3',
                                                     placeholder="Length of windowed data (in samples) e.g. 3 equals "
                                                                 "3*3 samples per 15 minutes = 45 min",
                                                     id='ml_length_dropdown'),
                                        html.H4(children='Upload file for machine learning classification'),
                                        html.P(
                                            'File with the columns: "timestamp" as time instance for the entry, '
                                            '"id" for participant id, (per configuration) "gl" for glucose in (mg/dL), '
                                            '(per configuration) "axis1" "axis2" "axis3" for the acceleration axis, '
                                            'optional: "phase" for trial condition phase, optional: "fasting_state" '
                                            'as categorical variable used for compliance'),
                                    ], style={'max-width': '800px', 'margin': '0 auto', 'vertical-align': 'middle'}),
                                    dcc.Upload(
                                        id="upload-data-dashboard_apply_ml",
                                        children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
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
                                        # Allow multiple files to be uploaded
                                        multiple=True,
                                    ),
                                    html.Div(id="output-data-upload-dashboard_apply_ml")
                                ]),
                            dcc.Tab(label='Decision Tree', id="DT_ml_tab", value="DT_ml_tab", children=[
                                html.H6(
                                    children='The predicted fasting state will be shown with coloured rectangle'
                                             ' intervals in the plot.'),
                                dcc.Graph(id="Mygraph-dashboard_apply_ml-dt"),
                                html.H4(children='The compliance score for the selected data is: '),
                                html.Div(id='dashboard_apply_ml-dt-output-compliance-score'),
                                html.H4(children='Confusion Matrix of the underlying model: '),
                                html.Img(id='dashboard_apply_ml-image-dt')
                            ]),
                            dcc.Tab(label='K-Nearest Neighbors', id="KN_ml_tab", value="KN_ml_tab", children=[
                                html.H6(
                                    children='The predicted fasting state will be shown with coloured rectangle'
                                             ' intervals in the plot.'),
                                dcc.Graph(id="Mygraph-dashboard_apply_ml-kn"),
                                html.H4(children='The compliance score for the selected data is: '),
                                html.Div(id='dashboard_apply_ml-kn-output-compliance-score'),
                                html.H4(children='Confusion Matrix of the underlying model: '),
                                html.Img(id='dashboard_apply_ml-image-knn')
                            ]),
                            dcc.Tab(label='Stochastic Gradient Descent', id="SGD_ml_tab", value="SGD_ml_tab", children=[
                                html.H6(
                                    children='The predicted fasting state will be shown with coloured rectangle'
                                             ' intervals in the plot.'),
                                dcc.Graph(id="Mygraph-dashboard_apply_ml-sgd"),
                                html.H4(children='The compliance score for the selected data is: '),
                                html.Div(id='dashboard_apply_ml-sgd-output-compliance-score'),
                                html.H4(children='Confusion Matrix of the underlying model: '),
                                html.Img(id='dashboard_apply_ml-image-sdg')
                            ]),
                            dcc.Tab(label='Logistic Regression', id="LR_ml_tab", value="LR_ml_tab", children=[
                                html.H6(
                                    children='The predicted fasting state will be shown with coloured rectangle '
                                             'intervals in the plot.'),
                                dcc.Graph(id="Mygraph-dashboard_apply_ml-lr"),
                                html.H4(children='The compliance score for the selected data is: '),
                                html.Div(id='dashboard_apply_ml-lr-output-compliance-score'),
                                html.H4(children='Confusion Matrix of the underlying model: '),
                                html.Img(id='dashboard_apply_ml-image-lr')
                            ]),
                            dcc.Tab(label='Multi-layer perceptron (MLP)', id="MLP_ml_tab", value="MLP_ml_tab",
                                    children=[
                                        html.H6(
                                            children='The predicted fasting state will be shown with coloured rectangle'
                                                     ' intervals in the plot.'),
                                        dcc.Graph(id="Mygraph-dashboard_apply_ml-mlp"),
                                        html.H4(children='The compliance score for the selected data is: '),
                                        html.Div(id='dashboard_apply_ml-mlp-output-compliance-score'),
                                        html.H4(children='Confusion Matrix of the underlying model: '),
                                        html.Img(id='dashboard_apply_ml-image-mlp')
                                    ]),
                            dcc.Tab(label='Random Forest', id="RF_ml_tab", value="RF_ml_tab", children=[
                                html.H6(
                                    children='The predicted fasting state will be shown with coloured rectangle'
                                             ' intervals in the plot.'),
                                dcc.Graph(id="Mygraph-dashboard_apply_ml-rf"),
                                html.H4(children='The compliance score for the selected data is: '),
                                html.Div(id='dashboard_apply_ml-rf-output-compliance-score'),
                                html.H4(children='Confusion Matrix of the underlying model: '),
                                html.Img(id='dashboard_apply_ml-image-rf')
                            ]),
                            dcc.Tab(label='Support Vector Machine', id="SVM_ml_tab", value="SVM_ml_tab", children=[
                                html.H6(
                                    children='The predicted fasting state will be shown with coloured rectangle '
                                             'intervals in the plot.'),
                                dcc.Graph(id="Mygraph-dashboard_apply_ml-svm"),
                                html.H4(children='The compliance score for the selected data is: '),
                                html.Div(id='dashboard_apply_ml-svm-output-compliance-score'),
                                html.H4(children='Confusion Matrix of the underlying model: '),
                                html.Img(id='dashboard_apply_ml-image-svm')
                            ])
                        ])
                    ]),
            dcc.Tab(label='Prediction Time Series Classification on Raw Data', id="tsc_tab_raw", style=tab_style,
                    selected_style=tab_selected_style, children=[
                    dcc.Tabs(id="subtabs_tsc_raw", value='Upload_tsc_tab_raw', children=[
                        dcc.Tab(label='Data Upload', id="upload_tsc_tab_raw", value="Upload_tsc_tab_raw",
                                style=tab_style,
                                selected_style=tab_selected_style, children=[
                                dcc.Store(id='tsc_data_raw'),
                                dcc.Store(id='actual_tsc_data_raw'),
                                html.Div([
                                    html.H4(children='Select options for time series classification:'),
                                    "Inserted data contains:",
                                    dcc.RadioItems(['Glucose and Acceleration', 'Glucose', 'Acceleration'],
                                                   'Glucose and Acceleration', id='tsc_data_options_raw'),
                                    html.H6(children=''),
                                    dcc.Checklist(['Postprocessing and smoothing data (Recommended)'],
                                                  ['Postprocessing and smoothing data (Recommended)'],
                                                  id='tsc_smoothing_checklist_raw'),
                                    html.H6(children=''),
                                    "Length of time series item (in samples) e.g. 3 equals 3*3 samples per 15 minutes "
                                    "= 45 min of time series input length:",
                                    dcc.Dropdown(['3', '4', '5'], '5', placeholder="Length of time series item",
                                                 id='tsc_length_dropdown_raw'),
                                    html.H4(children='Upload file for time series classification:'),
                                    html.P(
                                        'CGM data file with the columns: "timestamp" as time instance for the entry, '
                                        '"id" for participant id, (per configuration) "gl" for glucose in (mg/dL), '
                                        '(per configuration) "axis1" "axis2" "axis3" for the acceleration axis, '
                                        'optional: "phase" for trial condition phase, optional: "fasting_state" as '
                                        'categorical variable used for compliance'),
                                ], style={'max-width': '800px', 'margin': '0 auto', 'vertical-align': 'middle'}),
                                dcc.Upload(
                                    id="upload-data-tsc-raw",
                                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
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
                                    # Allow multiple files to be uploaded
                                    multiple=True,
                                ),
                                html.Div(id="output-data-upload-tsc-raw")
                            ]),
                        dcc.Tab(label='KNeighborsTimeSeriesClassifier', id="KNTSC_tab_raw", value="KNTSC_tab_raw",
                                children=[
                                    html.H6(
                                        children='The predicted fasting state will be shown with coloured rectangle'
                                                 ' intervals in the plot.'),
                                    dcc.Graph(id="Mygraph-tsc-raw"),
                                    html.H4(children='The compliance score for the selected data is: '),
                                    html.Div(id='tsc-KNTSC-output-compliance-score-raw'),
                                    html.H4(children='Confusion Matrix of the underlying model: '),
                                    html.Img(id='tsc-image-knn-raw')
                                ]),
                        dcc.Tab(label='TimeSeriesSVC', id="SVC_tsc_tab_raw", value="SVC_tsc_tab_raw", children=[
                            html.H6(
                                children='The predicted fasting state will be shown with coloured rectangle intervals'
                                         ' in the plot.'),
                            dcc.Graph(id="Mygraph-tsc-SVC-raw"),
                            html.H4(children='The compliance score for the selected data is: '),
                            html.Div(id='tsc-SVC-output-compliance-score-raw'),
                            html.H4(children='Confusion Matrix of the underlying model: '),
                            html.Img(id='tsc-image-svc-raw')
                        ])
                    ])
                ]),
            dcc.Tab(label='Prediction Time Series Classification on Features', id="tsc_tab", style=tab_style,
                    selected_style=tab_selected_style, children=[
                    dcc.Tabs(id="subtabs_tsc", value='Upload_tsc_tab', children=[
                        dcc.Tab(label='Data Upload', id="upload_tsc_tab", value="Upload_tsc_tab", style=tab_style,
                                selected_style=tab_selected_style, children=[
                                dcc.Store(id='tsc_data'),
                                dcc.Store(id='actual_tsc_data'),
                                html.Div([
                                    html.H4(children='Select options for time series classification:'),
                                    "Inserted data contains:",
                                    dcc.RadioItems(['Glucose and Acceleration', 'Glucose', 'Acceleration'],
                                                   'Glucose and Acceleration', id='tsc_data_options'),
                                    html.H6(children=''),
                                    dcc.Checklist(['Postprocessing and smoothing data (Recommended)'],
                                                  ['Postprocessing and smoothing data (Recommended)'],
                                                  id='tsc_smoothing_checklist'),
                                    html.H6(children=''),
                                    "Length of each window (in samples) e.g. 3 equals 3*3 samples per 15 minutes = "
                                    "45 min:",
                                    dcc.Dropdown(['3', '4', '5'], '3', placeholder="Length of windowed data ",
                                                 id='tsc_length_window_dropdown'),
                                    html.H6(children=''),
                                    "Length of time series (in samples) e.g. 3 equals 3*3 samples per 15 minutes = "
                                    "45 min of "
                                    "time series input length:",
                                    dcc.Dropdown(['3', '4', '5'], '5', placeholder="Length of time series item",
                                                 id='tsc_length_dropdown'),
                                    html.H4(children='Upload file for time series classification:'),
                                    html.P(
                                        'CGM data file with the columns: "timestamp" as time instance for the entry, '
                                        '"id" for participant id, (per configuration) "gl" for glucose in (mg/dL), '
                                        '(per configuration) "axis1" "axis2" "axis3" for the acceleration axis, '
                                        'optional:  "phase" for trial condition phase, optional: "fasting_state" '
                                        'as categorical variable used for compliance'),
                                ], style={'max-width': '800px', 'margin': '0 auto', 'vertical-align': 'middle'}),
                                dcc.Upload(
                                    id="upload-data-tsc",
                                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
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
                                    # Allow multiple files to be uploaded
                                    multiple=True,
                                ),
                                html.Div(id="output-data-upload-tsc")
                            ]),
                        dcc.Tab(label='KNeighborsTimeSeriesClassifier', id="KNTSC_tab", value="KNTSC_tab", children=[
                            html.H6(
                                children='The predicted fasting state will be shown with coloured rectangle intervals '
                                         'in the plot.'),
                            dcc.Graph(id="Mygraph-tsc"),
                            html.H4(children='The compliance score for the selected data is: '),
                            html.Div(id='tsc-KNTSC-output-compliance-score'),
                            html.H4(children='Confusion Matrix of the underlying model: '),
                            html.Img(id='tsc-image-knn')
                        ]),
                        dcc.Tab(label='TimeSeriesSVC', id="SVC_tsc_tab", value="SVC_tsc_tab", children=[
                            html.H6(
                                children='The predicted fasting state will be shown with coloured rectangle intervals '
                                         'in the plot.'),
                            dcc.Graph(id="Mygraph-tsc-SVC"),
                            html.H4(children='The compliance score for the selected data is: '),
                            html.Div(id='tsc-SVC-output-compliance-score'),
                            html.H4(children='Confusion Matrix of the underlying model: '),
                            html.Img(id='tsc-image-svc')
                        ])
                    ])
                ])
        ])
    ])
])


#########TAB ONE################

##### SCREENING ##########

@app.callback(
    Output("scatter-plot-screening", "figure"),
    Input('pandas-dropdown-1', 'value'))
def update_bar_chart(value):
    if value is None:
        raise PreventUpdate
    else:
        df = df_screening[(df_screening["id"] == value)]
        fig = px.scatter(
            df, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Patient reported fasting state"
            })

        return fig


@app.callback(
    Output('pandas-output-container-1', 'children'),
    Input('pandas-dropdown-1', 'value')
)
def update_output(value):
    return f'You have selected {value}'


@app.callback(
    Output("scatter-plot-act-screening", "figure"),
    Input('pandas-dropdown-1', 'value'))
def update_scatter_chart(value):
    if value is None:
        raise PreventUpdate
    else:
        df = df_screening[(df_screening["id"] == value)]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["gl"], name="glucose in mg/dl"), secondary_y=False)
        fig.add_trace(go.Line(
            x=df["time"], y=df["acc_magn"], name="Acceleration magnitude"), secondary_y=True)

        # fig = go.Figure(data=fig1.data + fig2.data)
        return fig


#########  ETRE  ################

@app.callback(
    Output("scatter-plot-eTRE", "figure"),
    Input('pandas-dropdown-2', 'value'))
def update_bar_chart(value):
    if value is None:
        raise PreventUpdate
    else:
        df = df_eTRE[(df_eTRE["id"] == value)]
        fig = px.scatter(
            df, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Patient reported fasting state"
            })

        return fig


@app.callback(
    Output('pandas-output-container-2', 'children'),
    Input('pandas-dropdown-2', 'value')
)
def update_output(value):
    return f'You have selected {value}'


@app.callback(
    Output("scatter-plot-act-eTRE", "figure"),
    Input('pandas-dropdown-2', 'value'))
def update_scatter_chart(value):
    if value is None:
        raise PreventUpdate
    else:

        df = df_eTRE[(df_eTRE["id"] == value)]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["gl"], name="glucose in mg/dl"), secondary_y=False)
        fig.add_trace(go.Line(
            x=df["time"], y=df["acc_magn"], name="Acceleration magnitude"), secondary_y=True)

        # fig = go.Figure(data=fig1.data + fig2.data)
        return fig


######### LTRE ################
@app.callback(
    Output("scatter-plot-lTRE", "figure"),
    Input('pandas-dropdown-3', 'value'))
def update_bar_chart(value):
    if value is None:
        raise PreventUpdate
    else:
        df = df_lTRE[(df_lTRE["id"] == value)]
        fig = px.scatter(
            df, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Patient reported fasting state"
            })

        return fig


@app.callback(
    Output('pandas-output-container-3', 'children'),
    Input('pandas-dropdown-3', 'value')
)
def update_output(value):
    return f'You have selected {value}'


@app.callback(
    Output("scatter-plot-act-lTRE", "figure"),
    Input('pandas-dropdown-3', 'value'))
def update_scatter_chart(value):
    if value is None:
        raise PreventUpdate
    else:
        df = df_lTRE[(df_lTRE["id"] == value)]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["gl"], name="glucose in mg/dl"), secondary_y=False)
        fig.add_trace(go.Line(
            x=df["time"], y=df["acc_magn"], name="Acceleration magnitude"), secondary_y=True)
        return fig


#########TAB TWO ################

@app.callback(
    Output("scatter-plot-bahai-screening", "figure"),
    Input('pandas-dropdown-4', 'value'))
def update_bar_chart(value):
    if value is None:
        raise PreventUpdate
    else:

        df = df_parofastin_screening[(df_parofastin_screening["id"] == value)]
        fig1 = px.scatter(
            df, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Patient reported fasting state"
            })
        return fig1


@app.callback(
    Output('pandas-output-container-4', 'children'),
    Input('pandas-dropdown-4', 'value')
)
def update_output(value):
    return f'You have selected {value}'


@app.callback(
    Output("scatter-plot-bahai-intervention", "figure"),
    Input('pandas-dropdown-5', 'value'))
def update_bar_chart(value):
    if value is None:
        raise PreventUpdate
    else:

        df = df_parofastin_intervention[(df_parofastin_intervention["id"] == value)]
        fig1 = px.scatter(
            df, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Patient reported fasting state"
            })
        return fig1


@app.callback(
    Output('pandas-output-container-5', 'children'),
    Input('pandas-dropdown-5', 'value')
)
def update_output(value):
    return f'You have selected {value}'


########## 16 8 #########

@app.callback(
    Output("scatter-plot-168-screening", "figure"),
    Input('pandas-dropdown-11', 'value'))
def update_bar_chart(value):
    if value is None:
        raise PreventUpdate
    else:
        df = df_parofastin_screening[(df_parofastin_screening["id"] == value)]
        fig1 = px.scatter(
            df, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Patient reported fasting state"
            })
        return fig1


@app.callback(
    Output('pandas-output-container-11', 'children'),
    Input('pandas-dropdown-11', 'value')
)
def update_output(value):
    return f'You have selected {value}'


@app.callback(
    Output("scatter-plot-168-intervention", "figure"),
    Input('pandas-dropdown-12', 'value'))
def update_bar_chart(value):
    if value is None:
        raise PreventUpdate
    else:
        df = df_parofastin_intervention[(df_parofastin_intervention["id"] == value)]
        fig1 = px.scatter(
            df, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Patient reported fasting state"
            })
        return fig1


@app.callback(
    Output('pandas-output-container-12', 'children'),
    Input('pandas-dropdown-12', 'value')
)
def update_output(value):
    return f'You have selected {value}'


################# TAB FIVE #####################


@app.callback(
    [Output("Mygraph-hutchison", "figure"), Output('hutchison-output-compliance-score', 'children')],
    [Input("upload-data-hutchison", "contents"), Input("upload-data-hutchison", "filename")]
)
def update_graph(contents, filename):
    if contents is None:
        raise PreventUpdate
    else:

        if contents:
            content = contents[1]
            file = filename[1]
            df = parse_data(content, file)
            content_state = contents[0]
            filename_state = filename[0]
            df_state = parse_data(content_state, filename_state)

        df_predicted, compliance_score = hutchison_prediction(df, df_state)

        fig = px.scatter(
            df_predicted, x="time", y="gl", color="pred_fasting_state",
            color_discrete_map={"fasting": "red", "non-fasting": "blue", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "pred_fasting_state": "Predicted fasting state"
            },
            title="Inserted glucose data with applied fasting state prediction"
        )

        return fig, compliance_score




@app.callback(
    Output("output-data-upload-hutchison", "children"),
    [Input("upload-data-hutchison", "contents"), Input("upload-data-hutchison", "filename")]
)
def update_table(contents, filename):
    table = html.Div()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)

        table = html.Div(
            [
                html.H5(filename),
                dash_table.DataTable(
                    data=df.to_dict("rows"),
                    columns=[{"name": i, "id": i} for i in df.columns],
                ),
                html.Hr(),
                html.Div("Raw Content"),
                html.Pre(
                    contents[0:200] + "...",
                    style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"},
                ),
            ]
        )

    return table


################# TAB SIX #####################


@app.callback(
    [Output('ml_data', 'data'), Output('actual_ml_data', 'data')],
    [Input("upload-data-dashboard_apply_ml", "contents"), Input("upload-data-dashboard_apply_ml", "filename"), Input('ml_length_dropdown', 'value'),
     Input('ml_data_options', 'value')])
def clean_data_ml(contents, filename, window_size, data_option):
    if contents is None:
        raise PreventUpdate
    else:

        if contents:
            contents = contents[0]
            filename = filename[0]
            df = parse_data(contents, filename)

        window_size = int(window_size)
        df_prediction = df

        df_prediction['Time'] = pd.to_datetime(df_prediction['time'], format='%Y-%m-%dT%H:%M:%S')
        df_prediction['Day'] = df_prediction["Time"].dt.date

        if data_option == "Glucose and Acceleration":
            df_prediction['Glucose'] = pd.to_numeric(df_prediction['gl'])

            try:
                df_prediction = df_prediction.sort_values(by=['id', 'time', "phase"], ascending=True)
            except:
                df_prediction = df_prediction.sort_values(by=['id', 'time'], ascending=True)

            df_prediction = df_prediction.reset_index(drop=True)
            df_prediction = df_prediction.drop(["Unnamed: 0"], axis=1)

            print("start windowing")
            step_size = 1
            x_list_pred, prediction_labels = windowing_ml_gl_acc(df_prediction, window_size, step_size)
            print("finished windowing")

            X_pred = feat_statistical_measures_gl_acc_ml(x_list_pred)

        elif data_option == "Glucose":

            df_prediction['Glucose'] = pd.to_numeric(df_prediction['gl'])

            try:
                df_prediction = df_prediction.sort_values(by=['id', 'time', "phase"], ascending=True)
            except:
                df_prediction = df_prediction.sort_values(by=['id', 'time'], ascending=True)

            df_prediction = df_prediction.reset_index(drop=True)
            df_prediction = df_prediction.drop(["Unnamed: 0"], axis=1)

            print("start windowing")
            step_size = 1
            x_list_pred, prediction_labels = windowing_ml_gl(df_prediction, window_size, step_size)
            print("finished windowing")
            X_pred = feat_statistical_measures_gl_ml(x_list_pred)

        elif data_option == "Acceleration":

            try:
                df_prediction = df_prediction.sort_values(by=['id', 'time', "phase"], ascending=True)
            except:
                df_prediction = df_prediction.sort_values(by=['id', 'time'], ascending=True)

            df_prediction = df_prediction.reset_index(drop=True)
            df_prediction = df_prediction.drop(["Unnamed: 0"], axis=1)

            print("start windowing")
            step_size = 1
            x_list_pred, prediction_labels = windowing_ml_acc(df_prediction, window_size, step_size)
            print("finished windowing")
            X_pred = feat_statistical_measures_acc_ml(x_list_pred)

        X_pred["labels"] = prediction_labels
        X_pred_json = X_pred.to_json(date_format='iso', orient='split')
        df_json = df.to_json(date_format='iso', orient='split')

        return X_pred_json, df_json


@app.callback(
    Output("output-data-upload-dashboard_apply_ml", "children"),
    [Input("upload-data-dashboard_apply_ml", "contents"), Input("upload-data-dashboard_apply_ml", "filename")]
)
def update_table(contents, filename):
    table = html.Div()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)

        table = html.Div(
            [
                html.H5(filename),
                dash_table.DataTable(
                    data=df.to_dict("rows"),
                    columns=[{"name": i, "id": i} for i in df.columns],
                ),
                html.Hr(),
                html.Div("Raw Content"),
                html.Pre(
                    contents[0:200] + "...",
                    style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"},
                ),
            ]
        )

    return table


@app.callback(
    [Output("Mygraph-dashboard_apply_ml-lr", "figure"), Output("dashboard_apply_ml-lr-output-compliance-score", "children")],
    [Input("ml_data", "data"), Input('actual_ml_data', 'data'), Input('ml_smoothing_checklist', 'value'),
     Input('ml_data_options', 'value'), Input('ml_length_dropdown', 'value')]
)
def update_graph(df_ml, df_original, smoothing_flag, data_option, window_size):
    if df_ml is None:
        raise PreventUpdate
    else:

        df_ml = pd.read_json(df_ml, orient='split')
        df_original = pd.read_json(df_original, orient='split')

        ml_type = "ML_LR"

        df_predicted, accuray = make_prediction_from_loaded_ml(df_ml, data_option, ml_type, smoothing_flag)

        map_dict_color = {"non-fasting": "blue", "fasting": "red"}
        df_predicted["pred_fasting_state_color"] = df_predicted["pred_fasting_state"].map(map_dict_color)

        fig = px.scatter(
            df_original, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Actual fasting state"
            },
            title="Inserted glucose data with applied fasting state prediction"
        )

        try:
            df_original = df_original.sort_values(by=['id', 'time', "phase"], ascending=True).reset_index()
        except:
            df_original = df_original.sort_values(by=['id', 'time'], ascending=True).reset_index()

        window_size = int(window_size)
        step_after = window_size // 2

        if window_size % 2 == 0:
            step_before = (window_size // 2) - 1
        else:
            step_before = window_size // 2

        end = df_predicted.shape[0] - (window_size + 1)

        for i in range(window_size - step_before, end, 1):
            fig.add_vrect(
                x0=df_original.loc[i - step_before]["time"],
                x1=df_original.loc[i + step_after]["time"],
                fillcolor=df_predicted.loc[i]["pred_fasting_state_color"],
                opacity=0.15,
                line_width=0)

        return [fig, accuray]


@app.callback(
    Output("dashboard_apply_ml-image-lr", "src"),
    Input('ml_data_options', 'value')
)
def update_image(data_option):
    if data_option == "Glucose and Acceleration":
        encoded_image = base64.b64encode(open("assets/ML_LR.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Glucose":
        encoded_image = base64.b64encode(open("assets/ML_LR_gl.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Acceleration":
        encoded_image = base64.b64encode(open("assets/ML_LR_acc.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return filename


@app.callback(
    [Output("Mygraph-dashboard_apply_ml-sgd", "figure"), Output("dashboard_apply_ml-sgd-output-compliance-score", "children")],
    [Input("ml_data", "data"), Input('actual_ml_data', 'data'), Input('ml_smoothing_checklist', 'value'),
     Input('ml_data_options', 'value'), Input('ml_length_dropdown', 'value')])
def update_graph(df_ml, df_original, smoothing_flag, data_option, window_size):
    if df_ml is None:
        raise PreventUpdate
    else:

        df_ml = pd.read_json(df_ml, orient='split')
        df_original = pd.read_json(df_original, orient='split')

        ml_type = "ML_SDG"

        df_predicted, accuray = make_prediction_from_loaded_ml(df_ml, data_option, ml_type, smoothing_flag)

        map_dict_color = {"non-fasting": "blue", "fasting": "red"}
        df_predicted["pred_fasting_state_color"] = df_predicted["pred_fasting_state"].map(map_dict_color)

        fig = px.scatter(
            df_original, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Actual fasting state"
            },
            title="Inserted glucose data with applied fasting state prediction"
        )

        try:
            df_original = df_original.sort_values(by=['id', 'time', "phase"], ascending=True).reset_index()
        except:
            df_original = df_original.sort_values(by=['id', 'time'], ascending=True).reset_index()

        window_size = int(window_size)
        step_after = window_size // 2

        if window_size % 2 == 0:
            step_before = (window_size // 2) - 1
        else:
            step_before = window_size // 2

        end = df_predicted.shape[0] - (window_size + 1)

        for i in range(window_size - step_before, end, 1):
            fig.add_vrect(
                x0=df_original.loc[i - step_before]["time"],
                x1=df_original.loc[i + step_after]["time"],
                fillcolor=df_predicted.loc[i]["pred_fasting_state_color"],
                opacity=0.15,
                line_width=0)

        return [fig, accuray]


@app.callback(
    Output("dashboard_apply_ml-image-sdg", "src"),
    Input('ml_data_options', 'value')
)
def update_image(data_option):
    if data_option == "Glucose and Acceleration":
        encoded_image = base64.b64encode(open("assets/ML_SGD.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Glucose":
        encoded_image = base64.b64encode(open("assets/ML_SGD_gl.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Acceleration":
        encoded_image = base64.b64encode(open("assets/ML_SGD_acc.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return filename


@app.callback(
    [Output("Mygraph-dashboard_apply_ml-svm", "figure"), Output("dashboard_apply_ml-svm-output-compliance-score", "children")],
    [Input("ml_data", "data"), Input('actual_ml_data', 'data'), Input('ml_smoothing_checklist', 'value'),
     Input('ml_data_options', 'value'), Input('ml_length_dropdown', 'value')]
)
def update_graph(df_ml, df_original, smoothing_flag, data_option, window_size):
    if df_ml is None:
        raise PreventUpdate
    else:

        df_ml = pd.read_json(df_ml, orient='split')
        df_original = pd.read_json(df_original, orient='split')

        ml_type = "ML_SVM"

        df_predicted, accuray = make_prediction_from_loaded_ml(df_ml, data_option, ml_type, smoothing_flag)

        map_dict_color = {"non-fasting": "blue", "fasting": "red"}
        df_predicted["pred_fasting_state_color"] = df_predicted["pred_fasting_state"].map(map_dict_color)

        fig = px.scatter(
            df_original, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Actual fasting state"
            },
            title="Inserted glucose data with applied fasting state prediction"
        )

        try:
            df_original = df_original.sort_values(by=['id', 'time', "phase"], ascending=True).reset_index()
        except:
            df_original = df_original.sort_values(by=['id', 'time'], ascending=True).reset_index()

        window_size = int(window_size)
        step_after = window_size // 2

        if window_size % 2 == 0:
            step_before = (window_size // 2) - 1
        else:
            step_before = window_size // 2

        end = df_predicted.shape[0] - (window_size + 1)

        for i in range(window_size - step_before, end, 1):
            fig.add_vrect(
                x0=df_original.loc[i - step_before]["time"],
                x1=df_original.loc[i + step_after]["time"],
                fillcolor=df_predicted.loc[i]["pred_fasting_state_color"],
                opacity=0.15,
                line_width=0)

        return [fig, accuray]


@app.callback(
    Output("dashboard_apply_ml-image-svm", "src"),
    Input('ml_data_options', 'value')
)
def update_image(data_option):
    if data_option == "Glucose and Acceleration":
        encoded_image = base64.b64encode(open("assets/ML_SVM.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Glucose":
        encoded_image = base64.b64encode(open("assets/ML_SVM_gl.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Acceleration":
        encoded_image = base64.b64encode(open("assets/ML_SVM_acc.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return filename


@app.callback(
    [Output("Mygraph-dashboard_apply_ml-dt", "figure"), Output("dashboard_apply_ml-dt-output-compliance-score", "children")],
    [Input("ml_data", "data"), Input('actual_ml_data', 'data'), Input('ml_smoothing_checklist', 'value'),
     Input('ml_data_options', 'value'), Input('ml_length_dropdown', 'value')]
)
def update_graph(df_ml, df_original, smoothing_flag, data_option, window_size):
    if df_ml is None:
        raise PreventUpdate
    else:

        df_ml = pd.read_json(df_ml, orient='split')
        df_original = pd.read_json(df_original, orient='split')

        ml_type = "ML_DT"

        df_predicted, accuray = make_prediction_from_loaded_ml(df_ml, data_option, ml_type, smoothing_flag)

        map_dict_color = {"non-fasting": "blue", "fasting": "red"}
        df_predicted["pred_fasting_state_color"] = df_predicted["pred_fasting_state"].map(map_dict_color)

        fig = px.scatter(
            df_original, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Actual fasting state"
            },
            title="Inserted glucose data with applied fasting state prediction"
        )

        try:
            df_original = df_original.sort_values(by=['id', 'time', "phase"], ascending=True).reset_index()
        except:
            df_original = df_original.sort_values(by=['id', 'time'], ascending=True).reset_index()

        window_size = int(window_size)
        step_after = window_size // 2

        if window_size % 2 == 0:
            step_before = (window_size // 2) - 1
        else:
            step_before = window_size // 2

        end = df_predicted.shape[0] - (window_size + 1)

        for i in range(window_size - step_before, end, 1):
            fig.add_vrect(
                x0=df_original.loc[i - step_before]["time"],
                x1=df_original.loc[i + step_after]["time"],
                fillcolor=df_predicted.loc[i]["pred_fasting_state_color"],
                opacity=0.15,
                line_width=0)

        print(accuray)

        return [fig, accuray]


@app.callback(
    Output("dashboard_apply_ml-image-dt", "src"),
    Input('ml_data_options', 'value')
)
def update_image(data_option):
    if data_option == "Glucose and Acceleration":
        encoded_image = base64.b64encode(open("assets/ML_DT.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Glucose":
        encoded_image = base64.b64encode(open("assets/ML_DT_gl.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Acceleration":
        encoded_image = base64.b64encode(open("assets/ML_DT_acc.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return filename


@app.callback(
    [Output("Mygraph-dashboard_apply_ml-rf", "figure"), Output("dashboard_apply_ml-rf-output-compliance-score", "children")],
    [Input("ml_data", "data"), Input('actual_ml_data', 'data'), Input('ml_smoothing_checklist', 'value'),
     Input('ml_data_options', 'value'), Input('ml_length_dropdown', 'value')]
)
def update_graph(df_ml, df_original, smoothing_flag, data_option, window_size):
    if df_ml is None:
        raise PreventUpdate
    else:

        df_ml = pd.read_json(df_ml, orient='split')
        df_original = pd.read_json(df_original, orient='split')

        ml_type = "ML_RF"

        df_predicted, accuray = make_prediction_from_loaded_ml(df_ml, data_option, ml_type, smoothing_flag)

        map_dict_color = {"non-fasting": "blue", "fasting": "red"}
        df_predicted["pred_fasting_state_color"] = df_predicted["pred_fasting_state"].map(map_dict_color)

        fig = px.scatter(
            df_original, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Actual fasting state"
            },
            title="Inserted glucose data with applied fasting state prediction"
        )

        try:
            df_original = df_original.sort_values(by=['id', 'time', "phase"], ascending=True).reset_index()
        except:
            df_original = df_original.sort_values(by=['id', 'time'], ascending=True).reset_index()

        window_size = int(window_size)
        step_after = window_size // 2

        if window_size % 2 == 0:
            step_before = (window_size // 2) - 1
        else:
            step_before = window_size // 2

        end = df_predicted.shape[0] - (window_size + 1)

        for i in range(window_size - step_before, end, 1):
            fig.add_vrect(
                x0=df_original.loc[i - step_before]["time"],
                x1=df_original.loc[i + step_after]["time"],
                fillcolor=df_predicted.loc[i]["pred_fasting_state_color"],
                opacity=0.15,
                line_width=0)

        return [fig, accuray]


@app.callback(
    Output("dashboard_apply_ml-image-rf", "src"),
    Input('ml_data_options', 'value')
)
def update_image(data_option):
    if data_option == "Glucose and Acceleration":
        encoded_image = base64.b64encode(open("assets/ML_RF.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Glucose":
        encoded_image = base64.b64encode(open("assets/ML_RF_gl.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Acceleration":
        encoded_image = base64.b64encode(open("assets/ML_RF_acc.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return filename


@app.callback(
    [Output("Mygraph-dashboard_apply_ml-kn", "figure"), Output("dashboard_apply_ml-kn-output-compliance-score", "children")],
    [Input("ml_data", "data"), Input('actual_ml_data', 'data'), Input('ml_smoothing_checklist', 'value'),
     Input('ml_data_options', 'value'), Input('ml_length_dropdown', 'value')]
)
def update_graph(df_ml, df_original, smoothing_flag, data_option, window_size):
    if df_ml is None:
        raise PreventUpdate
    else:

        df_ml = pd.read_json(df_ml, orient='split')
        df_original = pd.read_json(df_original, orient='split')

        ml_type = "ML_KNN"

        df_predicted, accuray = make_prediction_from_loaded_ml(df_ml, data_option, ml_type, smoothing_flag)

        map_dict_color = {"non-fasting": "blue", "fasting": "red"}
        df_predicted["pred_fasting_state_color"] = df_predicted["pred_fasting_state"].map(map_dict_color)

        fig = px.scatter(
            df_original, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Actual fasting state"
            },
            title="Inserted glucose data with applied fasting state prediction"
        )

        try:
            df_original = df_original.sort_values(by=['id', 'time', "phase"], ascending=True).reset_index()
        except:
            df_original = df_original.sort_values(by=['id', 'time'], ascending=True).reset_index()

        window_size = int(window_size)
        step_after = window_size // 2

        if window_size % 2 == 0:
            step_before = (window_size // 2) - 1
        else:
            step_before = window_size // 2

        end = df_predicted.shape[0] - (window_size + 1)

        for i in range(window_size - step_before, end, 1):
            fig.add_vrect(
                x0=df_original.loc[i - step_before]["time"],
                x1=df_original.loc[i + step_after]["time"],
                fillcolor=df_predicted.loc[i]["pred_fasting_state_color"],
                opacity=0.15,
                line_width=0)

        return [fig, accuray]


@app.callback(
    Output("dashboard_apply_ml-image-knn", "src"),
    Input('ml_data_options', 'value')
)
def update_image(data_option):
    if data_option == "Glucose and Acceleration":
        encoded_image = base64.b64encode(open("assets/ML_KNN.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Glucose":
        encoded_image = base64.b64encode(open("assets/ML_KNN_gl.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Acceleration":
        encoded_image = base64.b64encode(open("assets/ML_KNN_acc.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return filename


@app.callback(
    [Output("Mygraph-dashboard_apply_ml-mlp", "figure"), Output("dashboard_apply_ml-mlp-output-compliance-score", "children")],
    [Input("ml_data", "data"), Input('actual_ml_data', 'data'), Input('ml_smoothing_checklist', 'value'),
     Input('ml_data_options', 'value'), Input('ml_length_dropdown', 'value')]
)
def update_graph(df_ml, df_original, smoothing_flag, data_option, window_size):
    if df_ml is None:
        raise PreventUpdate
    else:

        df_ml = pd.read_json(df_ml, orient='split')
        df_original = pd.read_json(df_original, orient='split')

        ml_type = "ML_MLP"

        df_predicted, accuray = make_prediction_from_loaded_ml(df_ml, data_option, ml_type, smoothing_flag)

        map_dict_color = {"non-fasting": "blue", "fasting": "red"}
        df_predicted["pred_fasting_state_color"] = df_predicted["pred_fasting_state"].map(map_dict_color)

        fig = px.scatter(
            df_original, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Actual fasting state"
            },
            title="Inserted glucose data with applied fasting state prediction"
        )

        try:
            df_original = df_original.sort_values(by=['id', 'time', "phase"], ascending=True).reset_index()
        except:
            df_original = df_original.sort_values(by=['id', 'time'], ascending=True).reset_index()

        window_size = int(window_size)
        step_after = window_size // 2

        if window_size % 2 == 0:
            step_before = (window_size // 2) - 1
        else:
            step_before = window_size // 2

        end = df_predicted.shape[0] - (window_size + 1)

        for i in range(window_size - step_before, end, 1):
            fig.add_vrect(
                x0=df_original.loc[i - step_before]["time"],
                x1=df_original.loc[i + step_after]["time"],
                fillcolor=df_predicted.loc[i]["pred_fasting_state_color"],
                opacity=0.15,
                line_width=0)

        return [fig, accuray]


@app.callback(
    Output("dashboard_apply_ml-image-mlp", "src"),
    Input('ml_data_options', 'value')
)
def update_image(data_option):
    if data_option == "Glucose and Acceleration":
        encoded_image = base64.b64encode(open("assets/ML_MLP.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Glucose":
        encoded_image = base64.b64encode(open("assets/ML_MLP_gl.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Acceleration":
        encoded_image = base64.b64encode(open("assets/ML_MLP_acc.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return filename


############ Tab SIX ##################

@app.callback(
    [Output('tsc_data_raw', 'data'), Output('actual_tsc_data_raw', 'data')],
    [Input("upload-data-tsc-raw", "contents"), Input("upload-data-tsc-raw", "filename"),
     Input("tsc_data_options_raw", "value")])
def clean_data_tsc(contents, filename, data_option):
    if contents is None:
        raise PreventUpdate
    else:
        if contents:
            contents = contents[0]
            filename = filename[0]
            df = parse_data(contents, filename)

        try:
            df = df.sort_values(by=['id', 'time', "phase"], ascending=True)
            df = df.drop(["Unnamed: 0"], axis=1)
        except:
            df = df.sort_values(by=['id', 'time'], ascending=True)

        df = df.reset_index(drop=True)
        X_pred_json = df.to_json(date_format='iso', orient='split')
        df_json = df.to_json(date_format='iso', orient='split')

        return X_pred_json, df_json

# Table
@app.callback(
    Output("output-data-upload-tsc-raw", "children"),
    [Input("upload-data-tsc-raw", "contents"), Input("upload-data-tsc-raw", "filename")]
)
def update_table(contents, filename):
    table = html.Div()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)

        table = html.Div(
            [
                html.H5(filename),
                dash_table.DataTable(
                    data=df.to_dict("rows"),
                    columns=[{"name": i, "id": i} for i in df.columns],
                ),
                html.Hr(),
                html.Div("Raw Content"),
                html.Pre(
                    contents[0:200] + "...",
                    style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"},
                ),
            ]
        )

    return table


@app.callback(
    [Output("Mygraph-tsc-raw", "figure"), Output("tsc-KNTSC-output-compliance-score-raw", "children")],
    [Input("tsc_data_raw", "data"), Input('actual_tsc_data_raw', 'data'), Input('tsc_length_dropdown_raw', 'value'),
     Input('tsc_smoothing_checklist_raw', 'value'), Input("tsc_data_options_raw", "value")]
)
def update_graph(df_tsc, df_original, tsc_length, smoothing_flag, data_option):
    if df_tsc is None:
        raise PreventUpdate
    else:
        tsc_length = int(tsc_length)

        df_tsc = pd.read_json(df_tsc, orient='split')
        df_original = pd.read_json(df_original, orient='split')

        tsc_type = "KNN_TSC"

        df_predicted, accuray = make_prediction_from_loaded_tsc_raw_KNN(df_tsc, tsc_type, tsc_length, smoothing_flag,
                                                                        data_option)

        map_dict_color = {"non-fasting": "blue", "fasting": "red", "nan": "yellow"}
        df_predicted["pred_fasting_state_color"] = df_predicted["pred_fasting_state"].map(map_dict_color)

        fig = px.scatter(
            df_original, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Actual fasting state"
            },
            title="Inserted glucose data with applied fasting state prediction"
        )

        max_length = df_original.shape[0] - tsc_length
        steps = tsc_length - 1
        counter = 0
        for i in range(0, max_length, steps):
            fig.add_vrect(
                x0=df_original.loc[i]["time"],
                x1=df_original.loc[i + steps]["time"],
                fillcolor=df_predicted.loc[counter]["pred_fasting_state_color"],
                opacity=0.15,
                line_width=0)
            counter += 1

        return fig, accuray


@app.callback(
    Output("tsc-image-knn-raw", "src"),
    Input('tsc_data_options_raw', 'value')
)
def update_image(data_option):
    if data_option == "Glucose and Acceleration":
        encoded_image = base64.b64encode(open("assets/TSC_KNN_RAW.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Glucose":
        encoded_image = base64.b64encode(open("assets/TSC_KNN_RAW_gl.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Acceleration":
        encoded_image = base64.b64encode(open("assets/TSC_KNN_RAW_acc.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return filename


@app.callback(
    [Output("Mygraph-tsc-SVC-raw", "figure"), Output("tsc-SVC-output-compliance-score-raw", "children")],
    [Input("tsc_data_raw", "data"), Input('actual_tsc_data_raw', 'data'), Input('tsc_length_dropdown_raw', 'value'),
     Input('tsc_smoothing_checklist_raw', 'value'), Input("tsc_data_options_raw", "value")]
)
def update_graph(df_tsc, df_original, tsc_length, smoothing_flag, data_option):
    if df_tsc is None:
        raise PreventUpdate
    else:

        tsc_length = int(tsc_length)

        df_tsc = pd.read_json(df_tsc, orient='split')
        df_original = pd.read_json(df_original, orient='split')

        tsc_type = "CLF_TSC"

        df_predicted, accuray = make_prediction_from_loaded_tsc_raw_SVC(df_tsc, tsc_type, tsc_length, smoothing_flag,
                                                                        data_option)

        map_dict_color = {"non-fasting": "blue", "fasting": "red", "nan": "yellow"}
        df_predicted["pred_fasting_state_color"] = df_predicted["pred_fasting_state"].map(map_dict_color)

        fig = px.scatter(
            df_original, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Actual fasting state"
            },
            title="Inserted glucose data with applied fasting state prediction"
        )

        max_length = df_original.shape[0] - tsc_length
        steps = tsc_length - 1

        counter = 0
        for i in range(0, max_length, steps):
            fig.add_vrect(
                x0=df_original.loc[i]["time"],
                x1=df_original.loc[i + steps]["time"],
                fillcolor=df_predicted.loc[counter]["pred_fasting_state_color"],
                opacity=0.15,
                line_width=0)
            counter += 1

        return fig, accuray


@app.callback(
    Output("tsc-image-svc-raw", "src"),
    Input('tsc_data_options_raw', 'value')
)
def update_image(data_option):
    if data_option == "Glucose and Acceleration":
        encoded_image = base64.b64encode(open("assets/TSC_SVC_RAW.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Glucose":
        encoded_image = base64.b64encode(open("assets/TSC_SVC_RAW_gl.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Acceleration":
        encoded_image = base64.b64encode(open("assets/TSC_SVC_RAW_acc.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return filename


################# TAB SEVEN #####################


@app.callback(
    [Output('tsc_data', 'data'), Output('actual_tsc_data', 'data')],
    [Input("upload-data-tsc", "contents"), Input("upload-data-tsc", "filename"),
     Input("tsc_length_window_dropdown", "value"), Input("tsc_data_options", "value")])
def clean_data_tsc(contents, filename, window_size, data_option):
    if contents is None:
        raise PreventUpdate
    else:
        if contents:
            contents = contents[0]
            filename = filename[0]
            df = parse_data(contents, filename)

        df_prediction = df
        window_size = int(window_size)

        df_prediction['Time'] = pd.to_datetime(df_prediction['time'], format='%Y-%m-%dT%H:%M:%S')
        df_prediction['Day'] = df_prediction["Time"].dt.date

        df['Time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S')
        df = df.sort_values(by=['id', 'Time'], ascending=True).reset_index(drop=True)

        if data_option == "Glucose and Acceleration":
            df_prediction['Glucose'] = pd.to_numeric(df_prediction['gl'])

            try:
                df_prediction = df_prediction.sort_values(by=['id', 'time', "phase"], ascending=True)
            except:
                df_prediction = df_prediction.sort_values(by=['id', 'time'], ascending=True)

            df_prediction = df_prediction.reset_index(drop=True)
            df_prediction = df_prediction.drop(["Unnamed: 0"], axis=1)

            print("start windowing")
            step_size = 1
            x_list_pred, prediction_labels = windowing_tsc_gl_acc(df_prediction, window_size, step_size)
            print("finished windowing")

            X_pred = feat_statistical_measures_gl_acc_tsc(x_list_pred)

        elif data_option == "Glucose":

            df_prediction['Glucose'] = pd.to_numeric(df_prediction['gl'])

            try:
                df_prediction = df_prediction.sort_values(by=['id', 'time', "phase"], ascending=True)
            except:
                df_prediction = df_prediction.sort_values(by=['id', 'time'], ascending=True)

            df_prediction = df_prediction.reset_index(drop=True)
            df_prediction = df_prediction.drop(["Unnamed: 0"], axis=1)

            print("start windowing")
            step_size = 1
            x_list_pred, prediction_labels = windowing_tsc_gl(df_prediction, window_size, step_size)
            print("finished windowing")
            X_pred = feat_statistical_measures_gl_tsc(x_list_pred)

        elif data_option == "Acceleration":

            try:
                df_prediction = df_prediction.sort_values(by=['id', 'time', "phase"], ascending=True)
            except:
                df_prediction = df_prediction.sort_values(by=['id', 'time'], ascending=True)

            df_prediction = df_prediction.reset_index(drop=True)
            df_prediction = df_prediction.drop(["Unnamed: 0"], axis=1)

            print("start windowing")
            step_size = 1
            x_list_pred, prediction_labels = windowing_tsc_acc(df_prediction, window_size, step_size)
            print("finished windowing")
            X_pred = feat_statistical_measures_acc_tsc(x_list_pred)

        X_pred["labels"] = prediction_labels
        X_pred_json = X_pred.to_json(date_format='iso', orient='split')
        df_json = df.to_json(date_format='iso', orient='split')

        return X_pred_json, df_json


# Table
@app.callback(
    Output("output-data-upload-tsc", "children"),
    [Input("upload-data-tsc", "contents"), Input("upload-data-tsc", "filename")]
)
def update_table(contents, filename):
    table = html.Div()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)

        table = html.Div(
            [
                html.H5(filename),
                dash_table.DataTable(
                    data=df.to_dict("rows"),
                    columns=[{"name": i, "id": i} for i in df.columns],
                ),
                html.Hr(),
                html.Div("Raw Content"),
                html.Pre(
                    contents[0:200] + "...",
                    style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"},
                ),
            ]
        )

    return table


@app.callback(
    [Output("Mygraph-tsc", "figure"), Output("tsc-KNTSC-output-compliance-score", "children")],
    [Input("tsc_data", "data"), Input('actual_tsc_data', 'data'), Input('tsc_length_dropdown', 'value'),
     Input("tsc_length_window_dropdown", "value"), Input('tsc_smoothing_checklist', 'value'),
     Input("tsc_data_options", "value")]
)
def update_graph(df_tsc, df_original, tsc_length, window_size, smoothing_flag, data_option):
    if df_tsc is None:
        raise PreventUpdate
    else:
        tsc_length = int(tsc_length)

        df_tsc = pd.read_json(df_tsc, orient='split')
        df_original = pd.read_json(df_original, orient='split')
        df_original['time'] = pd.to_datetime(df_original['time'], format='%Y-%m-%dT%H:%M:%S')
        df_original = df_original.sort_values(by=['id', 'time'], ascending=True).reset_index(drop=True)

        tsc_type = "KNN_TSC"

        df_predicted, accuray = make_prediction_from_loaded_tsc_KNN(df_tsc, tsc_type, tsc_length, smoothing_flag,
                                                                    data_option)

        map_dict_color = {"non-fasting": "blue", "fasting": "red", "nan": "yellow"}
        df_predicted["pred_fasting_state_color"] = df_predicted["pred_fasting_state"].map(map_dict_color)

        fig = px.scatter(
            df_original, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Actual fasting state"
            },
            title="Inserted glucose data with applied fasting state prediction"
        )

        window_size = int(window_size)

        if window_size == 5:
            steps = tsc_length + 3
        elif window_size == 4:
            steps = tsc_length + 2
        elif window_size == 3:
            steps = tsc_length + 1

        max_length = df_original.shape[0] - (tsc_length * 2)

        counter = 0
        for i in range(0, max_length, tsc_length):
            fig.add_vrect(
                x0=df_original.loc[i]["time"],
                x1=df_original.loc[i + steps]["time"],
                fillcolor=df_predicted.loc[counter]["pred_fasting_state_color"],
                opacity=0.15,
                line_width=0)
            counter += 1

        return fig, accuray


@app.callback(
    Output("tsc-image-knn", "src"),
    Input('tsc_data_options', 'value')
)
def update_image(data_option):
    if data_option == "Glucose and Acceleration":
        encoded_image = base64.b64encode(open("assets/TSC_KNN.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Glucose":
        encoded_image = base64.b64encode(open("assets/TSC_KNN_gl.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Acceleration":
        encoded_image = base64.b64encode(open("assets/TSC_KNN_acc.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return filename


@app.callback(
    [Output("Mygraph-tsc-SVC", "figure"), Output("tsc-SVC-output-compliance-score", "children")],
    [Input("tsc_data", "data"), Input('actual_tsc_data', 'data'), Input('tsc_length_dropdown', 'value'),
     Input("tsc_length_window_dropdown", "value"), Input('tsc_smoothing_checklist', 'value'),
     Input("tsc_data_options", "value")]
)
def update_graph(df_tsc, df_original, tsc_length, window_size, smoothing_flag, data_option):
    if df_tsc is None:
        raise PreventUpdate
    else:

        tsc_length = int(tsc_length)

        df_tsc = pd.read_json(df_tsc, orient='split')
        df_original = pd.read_json(df_original, orient='split')

        df_original['time'] = pd.to_datetime(df_original['time'], format='%Y-%m-%dT%H:%M:%S')
        df_original = df_original.sort_values(by=['id', 'time'], ascending=True)

        tsc_type = "CLF_TSC"

        df_predicted, accuray = make_prediction_from_loaded_tsc_SVC(df_tsc, tsc_type, tsc_length, smoothing_flag,
                                                                    data_option)

        map_dict_color = {"non-fasting": "blue", "fasting": "red", "nan": "yellow"}
        df_predicted["pred_fasting_state_color"] = df_predicted["pred_fasting_state"].map(map_dict_color)

        fig = px.scatter(
            df_original, x="time", y="gl", color="fasting_state",
            color_discrete_map={"fasting": "#BB2E10", "non-fasting": "#3498DB", "Undefined": "#2ECC71"},
            hover_data=['gl'],
            labels={
                "gl": "glucose in mg/dl",
                "fasting_state": "Actual fasting state"
            },
            title="Inserted glucose data with applied fasting state prediction"
        )

        window_size = int(window_size)

        if window_size == 5:
            steps = tsc_length + 3
        elif window_size == 4:
            steps = tsc_length + 2
        elif window_size == 3:
            steps = tsc_length + 1

        max_length = df_original.shape[0] - (tsc_length * 2)

        counter = 0
        for i in range(0, max_length, tsc_length):
            fig.add_vrect(
                x0=df_original.loc[i]["time"],
                x1=df_original.loc[i + steps]["time"],
                fillcolor=df_predicted.loc[counter]["pred_fasting_state_color"],
                opacity=0.15,
                line_width=0)
            counter += 1

        return fig, accuray


@app.callback(
    Output("tsc-image-svc", "src"),
    Input('tsc_data_options', 'value')
)
def update_image(data_option):
    if data_option == "Glucose and Acceleration":
        encoded_image = base64.b64encode(open("assets/TSC_SVC.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Glucose":
        encoded_image = base64.b64encode(open("assets/TSC_SVC_gl.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    elif data_option == "Acceleration":
        encoded_image = base64.b64encode(open("assets/TSC_SVC_acc.png", 'rb').read())
        filename = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return filename


#################################################


if __name__ == '__main__':
    app.run_server(debug=True)
