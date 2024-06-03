import pickle

from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from dashboard_apply_ml.ml_utils import *

base_path = "insert_data_folder_with_models_here"

#################### HUTCHISON ############################


def hutchison_prediction(dataframe_glucose, dataframe_fasting_times):

    def compliance_score(x):
        if x['gl'] >= x['gl_mean'] and x['fasting_state'] == "non-fasting":
            return "compliant"
        elif x['gl'] > x['gl_mean'] and x['fasting_state'] == "fasting":
            return "non-compliant"
        elif x['gl'] <= x['gl_mean'] and x['fasting_state'] == "non-fasting":
            return "non-compliant"
        elif x['gl'] < x['gl_mean'] and x['fasting_state'] == "fasting":
            return "compliant"

    df_labeled_all = get_fasting_states_hutchison(dataframe_glucose, dataframe_fasting_times)

    df_labeled_all['time'] = pd.to_datetime(df_labeled_all['time'], errors='coerce')
    fasting_states_to_keep = ['fasting', 'non-fasting']
    df_labeled_all = df_labeled_all[df_labeled_all.fasting_state.isin(fasting_states_to_keep)]

    df_labeled_all = df_labeled_all.sort_values(by=['id', 'time'], ascending=True)
    df_labeled_all['Day'] = df_labeled_all["time"].dt.date

    try:
        mean_fasting_glucose_per_phase = df_labeled_all.groupby(["id", "phase", "fasting_state"])["gl"].mean().reset_index(
            name='gl_mean')
    except:
        mean_fasting_glucose_per_phase = df_labeled_all.groupby(["id", "fasting_state"])["gl"].mean().reset_index(
            name='gl_mean')

    mean_non_fasting_glucose = mean_fasting_glucose_per_phase[
        mean_fasting_glucose_per_phase.fasting_state != "non-fasting"]
    mean_non_fasting_glucose = mean_non_fasting_glucose.drop(["fasting_state"], axis=1)

    try:
        df_merged_pred = pd.merge(df_labeled_all, mean_non_fasting_glucose, on=["id", "phase"])
    except:
        df_merged_pred = pd.merge(df_labeled_all, mean_non_fasting_glucose, on=["id"])

    df_merged_pred["pred_fasting_state"] = df_merged_pred.apply(make_prediction_hutchison, axis=1)

    df_merged_pred["compliance_score"] = df_merged_pred.apply(compliance_score, axis=1)
    result = df_merged_pred.compliance_score.value_counts()

    compliance_score = '%.3f'%(result[0] / (result[0] + result[1]) * 100)

    return df_merged_pred, compliance_score


def make_prediction_hutchison(x):
    if x['gl'] >= x['gl_mean']:
        return "non-fasting"
    elif x['gl'] < x['gl_mean']:
        return "fasting"

########################### ML #########################

def building_return_df(y_pred,y_pred_actual ):
    df_return = pd.DataFrame({'pred_fasting_state': y_pred, 'actual_fasting_state': y_pred_actual})
    map_dict = {1: "non-fasting", 0:"fasting"}
    df_return["pred_fasting_state"] = df_return["pred_fasting_state"].map(map_dict)
    df_return["actual_fasting_state"] =  df_return["actual_fasting_state"].map(map_dict)

    return df_return


def to_float(input_string):
    if input_string == "fasting":
        return 0
    else:
        return 1


def make_prediction_from_loaded_ml(df_prediction, data_option, ml_type ,smoothing_flag):
    features_to_keep_gl_acc = ["cgm_interdaycv", "J_index", "maximum", "z_mean", "y_max", "z_energy"]
    features_to_keep_acc = ["z_mean", "y_max", "z_energy"]

    prediction_labels = df_prediction["labels"]
    X_pred = df_prediction.drop(["labels"], axis=1)

    y_pred_labels = np.array(prediction_labels)
    func_float = np.vectorize(to_float)
    y_pred_actual_float = func_float(y_pred_labels)

    if data_option == "Glucose and Acceleration":
        scaler = pickle.load(open(base_path + "models_ML\\both\\ML_scaler_adapt.pkl", 'rb'))
        X_pred = X_pred[features_to_keep_gl_acc]
        X_pred = scaler.transform(X_pred)
        folder = "both\\"

    elif data_option == "Glucose":

        folder = "gl\\"

    elif data_option == "Acceleration":
        X_pred = X_pred[features_to_keep_acc]

        scaler = pickle.load(open(base_path + "models_ML\\acc\\ML_scaler_adapt.pkl", 'rb'))
        if ml_type != "ML_MLP":
            X_pred = scaler.transform(X_pred)
        folder = "acc\\"

    start = base_path + "models_ML\\"
    end = "_adapt.pkl"

    path = start + folder + ml_type + end

    with open(path, "rb") as f:
        model = pickle.load(f)

    # Prediction
    y_pred = model.predict(X_pred)

    # If list is empty and smoothing not set
    if smoothing_flag:
        y_pred = smoothing(y_pred)

    # Compute Accuracy score
    acc_score = accuracy_score(y_pred_actual_float, y_pred)
    acc_score = "{:.2f}".format(acc_score)

    # Building return dataframe with actual and predicted labels
    y_pred = np.array(y_pred)
    y_pred_actual = np.array(y_pred_actual_float)

    df_return = pd.DataFrame({'pred_fasting_state': y_pred, 'actual_fasting_state': y_pred_actual})
    map_dict = {1: "non-fasting", 0:"fasting"}
    df_return["pred_fasting_state"] = df_return["pred_fasting_state"].map(map_dict)
    df_return["actual_fasting_state"] =  df_return["actual_fasting_state"].map(map_dict)

    return df_return, acc_score

########################### TIME SERIES CLASSIFICATION RAW #########################

def make_prediction_from_loaded_tsc_raw_SVC(df_prediction, ml_type, tsc_length, smoothing_flag, data_option):

    prediction_labels = df_prediction["fasting_state"]
    X_pred = df_prediction.drop(["fasting_state"], axis=1)

    y_pred_labels = np.array(prediction_labels)
    func_float = np.vectorize(to_float)
    y_pred_actual_float = func_float(y_pred_labels)

    if data_option == "Glucose and Acceleration":
        X_pred_tsc, y_pred_actual_float = formatting_to_tsc_format_raw_gl_acc(X_pred, y_pred_actual_float, tsc_length)
        folder = "both\\"
    elif data_option == "Glucose":
        X_pred_tsc, y_pred_actual_float = formatting_to_tsc_format_raw_gl(X_pred, y_pred_actual_float, tsc_length)
        folder = "gl\\"

    elif data_option == "Acceleration":
        X_pred_tsc, y_pred_actual_float = formatting_to_tsc_format_raw_acc(X_pred, y_pred_actual_float, tsc_length)
        folder = "acc\\"

    start = base_path + "TSC_data\\models_raw_TSC\\"
    end = ".pkl"

    path = start + folder+ ml_type + end

    with open(path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_pred_tsc)

    # if list is empty and smoothing not set
    if smoothing_flag:
        y_pred = smoothing(y_pred)

    acc_score = accuracy_score(y_pred_actual_float, y_pred)
    acc_score = "{:.2f}".format(acc_score)

    y_pred = np.array(y_pred)
    y_pred_actual_float = np.array(y_pred_actual_float)

    df_return = pd.DataFrame({'pred_fasting_state': y_pred, 'actual_fasting_state': y_pred_actual_float})

    map_dict = {1: "non-fasting", 0:"fasting"}
    df_return["pred_fasting_state"] = df_return["pred_fasting_state"].map(map_dict)
    df_return["actual_fasting_state"] = df_return["actual_fasting_state"].map(map_dict)

    return df_return, acc_score


def make_prediction_from_loaded_tsc_raw_KNN(df_prediction, ml_type, tsc_length, smoothing_flag, data_option):

    prediction_labels = df_prediction["fasting_state"]
    X_pred = df_prediction.drop(["fasting_state"], axis=1)

    y_pred_labels = np.array(prediction_labels)
    func_float = np.vectorize(to_float)
    y_pred_actual_float = func_float(y_pred_labels)

    if data_option == "Glucose and Acceleration":
        X_pred_tsc, y_pred_actual_float = formatting_to_tsc_format_raw_gl_acc(X_pred, y_pred_actual_float, tsc_length)
        folder = "both\\"
    elif data_option == "Glucose":
        X_pred_tsc, y_pred_actual_float = formatting_to_tsc_format_raw_gl(X_pred, y_pred_actual_float, tsc_length)
        folder = "gl\\"

    elif data_option == "Acceleration":
        X_pred_tsc, y_pred_actual_float = formatting_to_tsc_format_raw_acc(X_pred, y_pred_actual_float, tsc_length)
        folder = "acc\\"

    start = base_path + "TSC_data\\models_raw_TSC\\"
    end = ".pkl"

    path = start + folder+ ml_type + end

    with open(path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_pred_tsc)

    # if list is empty and smoothing not set
    if smoothing_flag:
        y_pred = smoothing(y_pred)

    acc_score = accuracy_score(y_pred_actual_float, y_pred)
    acc_score = "{:.2f}".format(acc_score)

    y_pred = np.array(y_pred)
    y_pred_actual_float = np.array(y_pred_actual_float)

    df_return = pd.DataFrame({'pred_fasting_state': y_pred, 'actual_fasting_state': y_pred_actual_float})

    map_dict = {1: "non-fasting", 0:"fasting"}
    df_return["pred_fasting_state"] = df_return["pred_fasting_state"].map(map_dict)
    df_return["actual_fasting_state"] =  df_return["actual_fasting_state"].map(map_dict)

    return df_return, acc_score

########################### TIME SERIES CLASSIFICATION FEATURES #########################

def make_prediction_from_loaded_tsc_KNN(df_prediction, ml_type, tsc_length, smoothing_flag, data_option):

    prediction_labels = df_prediction["labels"]
    X_pred = df_prediction.drop(["labels"], axis=1)


    y_pred_labels = np.array(prediction_labels)
    func_float = np.vectorize(to_float)
    y_pred_actual_float = func_float(y_pred_labels)

    if data_option == "Glucose and Acceleration":
        X_pred_tsc, y_pred_actual_float = formatting_to_tsc_format_gl_acc(X_pred, y_pred_actual_float, tsc_length)
        folder = "both\\"
    elif data_option == "Glucose":
        X_pred_tsc, y_pred_actual_float = formatting_to_tsc_format_gl(X_pred, y_pred_actual_float, tsc_length)
        folder = "gl\\"
    elif data_option == "Acceleration":
        X_pred_tsc, y_pred_actual_float = formatting_to_tsc_format_acc(X_pred, y_pred_actual_float, tsc_length)
        folder = "acc\\"

    start = base_path + "TSC_data\\models_TSC\\"
    end = ".pkl"

    path = start + folder + ml_type + end

    with open(path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_pred_tsc)

    # if list is empty and smoothing not set
    if smoothing_flag:
        y_pred = smoothing(y_pred)

    acc_score = accuracy_score(y_pred_actual_float, y_pred)
    acc_score = "{:.2f}".format(acc_score)

    y_pred = np.array(y_pred)
    y_pred_actual_float = np.array(y_pred_actual_float)

    df_return = pd.DataFrame({'pred_fasting_state': y_pred, 'actual_fasting_state': y_pred_actual_float})

    map_dict = {1: "non-fasting", 0:"fasting"}
    df_return["pred_fasting_state"] = df_return["pred_fasting_state"].map(map_dict)
    df_return["actual_fasting_state"] =  df_return["actual_fasting_state"].map(map_dict)

    return df_return, acc_score


def make_prediction_from_loaded_tsc_SVC(df_prediction, ml_type, tsc_length, smoothing_flag, data_option):

    prediction_labels = df_prediction["labels"]
    X_pred = df_prediction.drop(["labels"], axis=1)

    y_pred_labels = np.array(prediction_labels)
    func_float = np.vectorize(to_float)
    y_pred_actual_float = func_float(y_pred_labels)

    if data_option == "Glucose and Acceleration":
        X_pred_tsc, y_pred_actual_float = formatting_to_tsc_format_gl_acc(X_pred, y_pred_actual_float, tsc_length)
        folder = "both\\"
    elif data_option == "Glucose":
        X_pred_tsc, y_pred_actual_float = formatting_to_tsc_format_gl(X_pred, y_pred_actual_float, tsc_length)
        folder = "gl\\"
    elif data_option == "Acceleration":
        X_pred_tsc, y_pred_actual_float = formatting_to_tsc_format_acc(X_pred, y_pred_actual_float, tsc_length)
        folder = "acc\\"

    start = base_path + "TSC_data\\models_TSC\\"
    end = ".pkl"

    path = start + folder + ml_type + end

    with open(path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_pred_tsc)

    # if list is empty and smoothing not set
    if smoothing_flag:
        y_pred = smoothing(y_pred)

    acc_score = accuracy_score(y_pred_actual_float, y_pred)
    acc_score = "{:.2f}".format(acc_score)

    y_pred = np.array(y_pred)
    y_pred_actual_float = np.array(y_pred_actual_float)

    df_return = pd.DataFrame({'pred_fasting_state': y_pred, 'actual_fasting_state': y_pred_actual_float})

    map_dict = {1: "non-fasting", 0:"fasting"}
    df_return["pred_fasting_state"] = df_return["pred_fasting_state"].map(map_dict)
    df_return["actual_fasting_state"] =  df_return["actual_fasting_state"].map(map_dict)

    return df_return, acc_score