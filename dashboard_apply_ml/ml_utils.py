import base64
from datetime import timedelta, datetime

import pandas as pd
import cgmquantify as cgm
from scipy.stats import stats
import numpy as np
from tslearn.utils import to_time_series_dataset
from numpy.lib.stride_tricks import sliding_window_view


def clean_acceleration_data(dataframe):

    dataframe = dataframe.dropna(subset=['axis1', 'axis2', 'axis3'], how='all')

    dataframe['axis1'] = dataframe['axis1'].fillna(0)
    dataframe['axis2'] = dataframe['axis2'].fillna(0)
    dataframe['axis3'] = dataframe['axis3'].fillna(0)

    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.drop(["Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0"], axis=1)

    return dataframe


def preprocess_chronofast_data(df):

    def magnitude(axis1, axis2, axis3):
        vector = numpy.empty((0))
        vector = numpy.append(vector,
                              axis1)
        vector = numpy.append(vector, axis2)
        vector = numpy.append(vector, axis3)

        return math.sqrt(sum(pow(element, 2) for element in vector))

    df['gl_norm'] = sklearn.preprocessing.minmax_scale(df['gl'], feature_range=(0, 100))
    df['acc_magn'] = df.apply(lambda x: magnitude(x.axis1, x.axis2, x.axis3), axis=1)
    df['savgol'] = savgol_filter(df["acc_magn"], 3, 1)

    return df


def preprocess_parofastin_data(df):

    # convert to mg/dl
    df["time"] = pd.to_datetime(df['time'], format="%d-%m-%Y %H:%M")
    df["gl"] = df["gl"].str.replace(',', '.').astype(float)
    df["gl"] = df["gl"] * 18

    return df


def parse_data(contents, filename):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep='\t')
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif "txt" or "tsv" in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return df


def compare_times(row, starthour_sleep, endhour_sleep):

    try:

        val = "None"

        if not pd.isna(row["time"]):

                time =datetime.strptime(str(row["time"]), "%Y-%m-%d %H:%M:%S%z")
                time = time.strftime('%H:%M:%S')
                time = datetime.strptime(time,"%H:%M:%S").time()
                starthour_sleep = datetime.strptime(starthour_sleep,"%H:%M:%S").time()
                endhour_sleep = datetime.strptime(endhour_sleep,"%H:%M:%S").time()

                # starthour = 18 Uhr
                # endhour = 7 Uhr

                if row["phase"] == "eTRE":

                    eTRE_start = datetime.strptime("9:00:00","%H:%M:%S").time()
                    eTRE_ende = datetime.strptime("17:00:00", "%H:%M:%S").time()

                    if time < endhour_sleep:
                        val = "fasting"
                    elif time > starthour_sleep:
                        val = "fasting"
                    elif time > eTRE_start and time < eTRE_ende:
                        val = "non-fasting"
                    else: # time > endhour_sleep and time <starthour_sleep:
                        val = "Unrelevant"
                    return val

                elif row["phase"] == "lTRE":

                    lTRE_start = datetime.strptime("14:00:00", "%H:%M:%S").time()
                    lTRE_ende = datetime.strptime("22:00:00", "%H:%M:%S").time()

                    if time < endhour_sleep:
                        val = "fasting"
                    elif time > starthour_sleep:
                        val = "fasting"
                    elif time > lTRE_start and time < lTRE_ende:
                        val = "non-fasting"
                    else:  # time > endhour_sleep and time <starthour_sleep:
                        val = "Unrelevant"
                    return val

                else:
                    if time < endhour_sleep:
                        val = "fasting"
                    elif time > starthour_sleep:
                        val = "fasting"
                    else:  # time > endhour_sleep and time <starthour_sleep:
                        val = "non-fasting"
                    return val

        else:
            return val
    except:
        return val


def get_fasting_states_hutchison(df_glucose, df_fasting_times_all):
    df_fasting_times_all["Beginn erste Mahlzeit"] = pd.to_datetime(df_fasting_times_all["Beginn erste Mahlzeit"],
                                                                   format="%H:%M:%S")
    df_fasting_times_all["Beginn erste Mahlzeit"] = df_fasting_times_all["Beginn erste Mahlzeit"] + timedelta(hours=0)
    df_fasting_times_all["Beginn erste Mahlzeit"] = df_fasting_times_all["Beginn erste Mahlzeit"].dt.strftime(
        '%H:%M:%S')

    df_fasting_times_all["Beginn letzte Mahlzeit"] = pd.to_datetime(df_fasting_times_all["Beginn letzte Mahlzeit"],
                                                                    format="%H:%M:%S")
    df_fasting_times_all["Beginn letzte Mahlzeit"] = df_fasting_times_all["Beginn letzte Mahlzeit"] + timedelta(hours=4)
    df_fasting_times_all["Beginn letzte Mahlzeit"] = df_fasting_times_all["Beginn letzte Mahlzeit"].dt.strftime(
        '%H:%M:%S')

    df_fasting_times_all = df_fasting_times_all.dropna()
    df_fasting_times_all = df_fasting_times_all.sort_values(by=['id', 'time'], ascending=True)

    df_glucose["fasting_state"] = "Undefined"

    for index_1, row_df in df_glucose.iterrows():
        for index_2, row_df_fasting in df_fasting_times_all.iterrows():

            ###### from merged df ######
            try:
                time_df_merged = datetime.strptime(str(row_df["time"]), "%Y-%m-%d %H:%M:%S")
                date_df_merged = time_df_merged.strftime('%Y-%m-%d')
                date_df_merged = datetime.strptime(date_df_merged, '%Y-%m-%d')

                min_df_merged = time_df_merged.strftime('%H:%M:%S')
                min_df_merged = datetime.strptime(min_df_merged, '%H:%M:%S').time()
            except:
                time_df_merged = datetime.strptime(str(row_df["time"]), "%Y-%m-%d %H:%M:%S+%z")
                date_df_merged = time_df_merged.strftime('%Y-%m-%d')
                date_df_merged = datetime.strptime(date_df_merged, '%Y-%m-%d')

                min_df_merged = time_df_merged.strftime('%H:%M:%S')
                min_df_merged = datetime.strptime(min_df_merged, '%H:%M:%S').time()

            ###### from fasting df #######
            # day

            time_fasting = datetime.strptime(str(row_df_fasting["time"]), "%Y-%m-%d %H:%M:%S")
            date_fasting = time_fasting.strftime('%Y-%m-%d')
            date_fasting = datetime.strptime(date_fasting, '%Y-%m-%d')

            # minutes

            begin_non_fasting = datetime.strptime(str(row_df_fasting["Beginn erste Mahlzeit"]), "%H:%M:%S")
            begin_non_fasting = begin_non_fasting.strftime('%H:%M:%S')
            begin_non_fasting = datetime.strptime(begin_non_fasting, '%H:%M:%S').time()

            if not row_df_fasting["Beginn letzte Mahlzeit"] == "nan":
                end_non_fasting = datetime.strptime(str(row_df_fasting["Beginn letzte Mahlzeit"]), "%H:%M:%S")
                end_non_fasting = end_non_fasting.strftime('%H:%M:%S')
                end_non_fasting = datetime.strptime(end_non_fasting, '%H:%M:%S').time()

            try:

                if row_df["id"] == row_df_fasting["id"] and row_df["phase"] == row_df_fasting["phase"] and\
                        date_df_merged == date_fasting:

                    if begin_non_fasting < end_non_fasting:

                        # Zeit vorm Beginn und nach dem Ende vom Tag davor
                        if min_df_merged < begin_non_fasting:
                            # fastenzustand
                            df_glucose.at[index_1, 'fasting_state'] = "fasting"
                        # Zwischen Beginn und Ende am selben Tag
                        elif min_df_merged >= begin_non_fasting and min_df_merged <= end_non_fasting:
                            # nicht fastenzustand
                            df_glucose.at[index_1, 'fasting_state'] = "non-fasting"
                        # Zeit nach dem Ende der Fastenzeit
                        elif min_df_merged > end_non_fasting:
                            df_glucose.at[index_1, 'fasting_state'] = "fasting"
                            # fastenzustand

                    elif begin_non_fasting > end_non_fasting:
                        # begin_non_fasting 10 Uhr
                        # end_non_fasting  2 Uhr

                        # Zeit vorm Beginn und nach dem Ende vom Tag davor
                        if min_df_merged < begin_non_fasting:
                            # fastenzustand
                            df_glucose.at[index_1, 'fasting_state'] = "fasting"
                        # Zeit vorm Beginn und vor dem Ende vom Tag davor
                        elif min_df_merged >= begin_non_fasting:
                            # nicht fastenzustand
                            df_glucose.at[index_1, 'fasting_state'] = "non-fasting"

            except:
                if row_df["id"] == row_df_fasting["id"] and date_df_merged == date_fasting:

                    if begin_non_fasting < end_non_fasting:

                        # Zeit vorm Beginn und nach dem Ende vom Tag davor
                        if min_df_merged < begin_non_fasting:
                            # fastenzustand
                            df_glucose.at[index_1, 'fasting_state'] = "fasting"
                        # Zwischen Beginn und Ende am selben Tag
                        elif min_df_merged >= begin_non_fasting and min_df_merged <= end_non_fasting:
                            # nicht fastenzustand
                            df_glucose.at[index_1, 'fasting_state'] = "non-fasting"
                        # Zeit nach dem Ende der Fastenzeit
                        elif min_df_merged > end_non_fasting:
                            df_glucose.at[index_1, 'fasting_state'] = "fasting"
                            # fastenzustand

                    elif begin_non_fasting > end_non_fasting:
                        # begin_non_fasting 10 Uhr
                        # end_non_fasting  2 Uhr

                        # Zeit vorm Beginn und nach dem Ende vom Tag davor
                        if min_df_merged < begin_non_fasting:
                            # fastenzustand
                            df_glucose.at[index_1, 'fasting_state'] = "fasting"
                        # Zeit vorm Beginn und vor dem Ende vom Tag davor
                        elif min_df_merged >= begin_non_fasting:
                            # nicht fastenzustand
                            df_glucose.at[index_1, 'fasting_state'] = "non-fasting"

    return df_glucose

def windowing_tsc_gl_acc(df_prediction, window_size, step_size):

    x_list_pred = []
    prediction_labels = []

    # creating overlaping windows of size window-size 100
    for i in range(0, df_prediction.shape[0] - window_size, step_size):
        time = df_prediction['Time'].values[i: i + window_size]
        times = df_prediction['Day'].values[i: i + window_size]
        gls = df_prediction['Glucose'].values[i: i + window_size]
        steps = df_prediction['steps'].values[i: i + window_size]
        inclineStanding = df_prediction['inclineStanding'].values[i: i + window_size]
        inclineSitting = df_prediction['inclineSitting'].values[i: i + window_size]
        inclineLying = df_prediction['inclineLying'].values[i: i + window_size]

        xs = df_prediction['axis1'].values[i: i + window_size]
        ys = df_prediction['axis2'].values[i: i + window_size]
        zs = df_prediction['axis3'].values[i: i + window_size]

        label = stats.mode(df_prediction['fasting_state'][i: i + window_size])[0][0]

        df_slice = pd.DataFrame(time, columns=['Time'])
        df_slice["Day"] = times
        df_slice["Glucose"] = gls
        df_slice["axis1"] = xs
        df_slice["axis2"] = ys
        df_slice["axis3"] = zs
        df_slice["steps"] = steps
        df_slice["inclineStanding"] = inclineStanding
        df_slice["inclineSitting"] = inclineSitting
        df_slice["inclineLying"] = inclineLying

        x_list_pred.append(df_slice)

        prediction_labels.append(label)

    return x_list_pred, prediction_labels


def windowing_tsc_gl(df_prediction, window_size, step_size):
    """
    Function for windowing a prediction dataframe

    :param df_prediction: df of
    :param window_size: size of the window
    :param step_size: amount of vales

    :return: list of smoothed values
    """
    x_list_pred = []
    prediction_labels = []

    # creating overlaping windows of size window-size 100
    for i in range(0, df_prediction.shape[0] - window_size, step_size):
        time = df_prediction['Time'].values[i: i + window_size]
        times = df_prediction['Day'].values[i: i + window_size]
        gls = df_prediction['Glucose'].values[i: i + window_size]

        label = stats.mode(df_prediction['fasting_state'][i: i + window_size])[0][0]

        df_slice = pd.DataFrame(time, columns=['Time'])
        df_slice["Day"] = times
        df_slice["Glucose"] = gls

        x_list_pred.append(df_slice)

        prediction_labels.append(label)

    return x_list_pred, prediction_labels


def windowing_tsc_acc(df_prediction, window_size, step_size):

    x_list_pred = []
    prediction_labels = []

    # creating overlaping windows of size window-size 100
    for i in range(0, df_prediction.shape[0] - window_size, step_size):
        time = df_prediction['Time'].values[i: i + window_size]
        steps = df_prediction['steps'].values[i: i + window_size]
        inclineStanding = df_prediction['inclineStanding'].values[i: i + window_size]
        inclineSitting = df_prediction['inclineSitting'].values[i: i + window_size]
        inclineLying = df_prediction['inclineLying'].values[i: i + window_size]

        xs = df_prediction['axis1'].values[i: i + window_size]
        ys = df_prediction['axis2'].values[i: i + window_size]
        zs = df_prediction['axis3'].values[i: i + window_size]

        label = stats.mode(df_prediction['fasting_state'][i: i + window_size])[0][0]

        df_slice = pd.DataFrame(time, columns=['Time'])
        df_slice["axis1"] = xs
        df_slice["axis2"] = ys
        df_slice["axis3"] = zs
        df_slice["steps"] = steps
        df_slice["inclineStanding"] = inclineStanding
        df_slice["inclineSitting"] = inclineSitting
        df_slice["inclineLying"] = inclineLying

        x_list_pred.append(df_slice)

        prediction_labels.append(label)

    return x_list_pred, prediction_labels


def formatting_to_tsc_format_gl_acc(X_pred, y_pred, window_size):

    x_list_pred_tsc_1st = []
    x_list_pred_tsc_2nd = []
    pred_labels_tsc = []
    counter = 1

    for i in range(0, X_pred.shape[0], 1):

        time = X_pred['time'].values[i]
        cgm_interdaycv = X_pred['cgm_interdaycv'].values[i]
        J_index = X_pred['J_index'].values[i]
        maximum = X_pred['maximum'].values[i]
        z_mean = X_pred['z_mean'].values[i]
        y_max = X_pred['y_max'].values[i]
        z_energy = X_pred['z_energy'].values[i]

        x_list_pred_tsc_3rd = [cgm_interdaycv, J_index, maximum, z_mean, y_max, z_energy]

        x_list_pred_tsc_2nd.append(x_list_pred_tsc_3rd)

        if counter >= window_size:
            pred_labels_tsc.append(y_pred[i])

            x_list_pred_tsc_1st.append(x_list_pred_tsc_2nd)
            x_list_pred_tsc_2nd = []
            counter = 0

        counter += 1

    x_list_pred_tsc_1st = to_time_series_dataset(x_list_pred_tsc_1st)
    return x_list_pred_tsc_1st, pred_labels_tsc


def formatting_to_tsc_format_gl(X_pred, y_pred, window_size):

    x_list_pred_tsc_1st = []
    x_list_pred_tsc_2nd = []
    pred_labels_tsc = []
    counter = 1

    for i in range(0, X_pred.shape[0], 1):

        cgm_interdaycv = X_pred['cgm_interdaycv'].values[i]
        J_index = X_pred['J_index'].values[i]
        maximum = X_pred['maximum'].values[i]

        x_list_pred_tsc_3rd = [cgm_interdaycv, J_index, maximum]

        x_list_pred_tsc_2nd.append(x_list_pred_tsc_3rd)

        if counter >= window_size:
            pred_labels_tsc.append(y_pred[i])

            x_list_pred_tsc_1st.append(x_list_pred_tsc_2nd)
            x_list_pred_tsc_2nd = []
            counter = 0

        counter += 1

    x_list_pred_tsc_1st = to_time_series_dataset(x_list_pred_tsc_1st)
    return x_list_pred_tsc_1st, pred_labels_tsc


def formatting_to_tsc_format_acc(X_pred, y_pred, window_size):

    x_list_pred_tsc_1st = []
    x_list_pred_tsc_2nd = []
    pred_labels_tsc = []
    counter = 1

    for i in range(0, X_pred.shape[0], 1):

        time = X_pred['time'].values[i]
        z_mean = X_pred['z_mean'].values[i]
        y_max = X_pred['y_max'].values[i]
        z_energy = X_pred['z_energy'].values[i]

        x_list_pred_tsc_3rd = [z_mean, y_max, z_energy]

        x_list_pred_tsc_2nd.append(x_list_pred_tsc_3rd)

        if counter >= window_size:
            pred_labels_tsc.append(y_pred[i])

            x_list_pred_tsc_1st.append(x_list_pred_tsc_2nd)
            x_list_pred_tsc_2nd = []
            counter = 0

        counter += 1

    x_list_pred_tsc_1st = to_time_series_dataset(x_list_pred_tsc_1st)
    return x_list_pred_tsc_1st, pred_labels_tsc


def formatting_to_tsc_format_raw_gl_acc(X_pred, y_pred, window_size):
    x_list_pred_tsc_1st = []
    x_list_pred_tsc_2nd = []
    pred_labels_tsc = []
    counter = 1

    for i in range(0, X_pred.shape[0], 1):

        gl = X_pred['gl'].values[i]
        axis1 = X_pred['axis1'].values[i]
        axis2 = X_pred['axis2'].values[i]
        axis3 = X_pred['axis3'].values[i]
        steps = X_pred['steps'].values[i]
        lux = X_pred['lux'].values[i]
        inclineStanding = X_pred['inclineStanding'].values[i]
        inclineSitting = X_pred['inclineSitting'].values[i]
        inclineLying = X_pred['inclineLying'].values[i]
        vm = X_pred['vm'].values[i]

        x_list_pred_tsc_3rd = [gl, axis1, axis2, axis3, steps, lux, inclineStanding, inclineSitting, inclineLying, vm]

        x_list_pred_tsc_2nd.append(x_list_pred_tsc_3rd)

        if counter >= window_size:
            pred_labels_tsc.append(y_pred[i])

            x_list_pred_tsc_1st.append(x_list_pred_tsc_2nd)
            x_list_pred_tsc_2nd = []

            x_list_pred_tsc_3rd = [gl, axis1, axis2, axis3, steps, lux, inclineStanding, inclineSitting, inclineLying,
                                   vm]

            x_list_pred_tsc_2nd.append(x_list_pred_tsc_3rd)

            counter = 1

        counter += 1

    x_list_pred_tsc_1st = to_time_series_dataset(x_list_pred_tsc_1st)
    return x_list_pred_tsc_1st, pred_labels_tsc


def formatting_to_tsc_format_raw_gl(X_pred, y_pred, window_size):
    x_list_pred_tsc_1st = []
    x_list_pred_tsc_2nd = []
    pred_labels_tsc = []
    counter = 1

    for i in range(0, X_pred.shape[0], 1):

        gl = X_pred['gl'].values[i]
        x_list_pred_tsc_3rd = [gl]

        x_list_pred_tsc_2nd.append(x_list_pred_tsc_3rd)

        if counter >= window_size:
            pred_labels_tsc.append(y_pred[i])

            x_list_pred_tsc_1st.append(x_list_pred_tsc_2nd)
            x_list_pred_tsc_2nd = []

            x_list_pred_tsc_3rd = [gl]
            x_list_pred_tsc_2nd.append(x_list_pred_tsc_3rd)

            counter = 1

        counter += 1

    x_list_pred_tsc_1st = to_time_series_dataset(x_list_pred_tsc_1st)
    return x_list_pred_tsc_1st, pred_labels_tsc


def formatting_to_tsc_format_raw_acc(X_pred, y_pred, window_size):
    x_list_pred_tsc_1st = []
    x_list_pred_tsc_2nd = []
    pred_labels_tsc = []
    counter = 1

    for i in range(0, X_pred.shape[0], 1):
        axis1 = X_pred['axis1'].values[i]
        axis2 = X_pred['axis2'].values[i]
        axis3 = X_pred['axis3'].values[i]
        steps = X_pred['steps'].values[i]
        lux = X_pred['lux'].values[i]
        inclineStanding = X_pred['inclineStanding'].values[i]
        inclineSitting = X_pred['inclineSitting'].values[i]
        inclineLying = X_pred['inclineLying'].values[i]
        vm = X_pred['vm'].values[i]

        x_list_pred_tsc_3rd = [axis1, axis2, axis3, steps, lux, inclineStanding, inclineSitting, inclineLying, vm]

        x_list_pred_tsc_2nd.append(x_list_pred_tsc_3rd)

        if counter >= window_size:
            pred_labels_tsc.append(y_pred[i])

            x_list_pred_tsc_1st.append(x_list_pred_tsc_2nd)
            x_list_pred_tsc_2nd = []

            x_list_pred_tsc_3rd = [axis1, axis2, axis3, steps, lux, inclineStanding, inclineSitting, inclineLying, vm]

            x_list_pred_tsc_2nd.append(x_list_pred_tsc_3rd)

            counter = 1

        counter += 1

    x_list_pred_tsc_1st = to_time_series_dataset(x_list_pred_tsc_1st)
    return x_list_pred_tsc_1st, pred_labels_tsc


def smoothing(y_pred):
    """
    Function for smoothing the prediction curve by sliding window of 5

    :param y_pred: list of predicted values
    :return: list of smoothed values
    """
    outcomes = np.array([])
    last_four_values = y_pred[-4:]

    y_pred_windowed = sliding_window_view((y_pred), window_shape=5)

    for window in y_pred_windowed:
        pred_one = np.count_nonzero(window == 1)
        pred_zero = 5 - pred_one
        if pred_one > pred_zero:
            outcome = int(1)
            outcomes = np.append(outcomes, outcome)
        else:
            outcome = int(0)
            outcomes = np.append(outcomes, outcome)

    outcomes = np.append(outcomes, last_four_values)

    return outcomes


def windowing_ml_gl_acc(df_prediction, window_size, step_size):

    x_list_pred = []
    prediction_labels = []

    # creating overlaping windows of size window-size 100
    for i in range(0, df_prediction.shape[0] - window_size, step_size):
        time = df_prediction['Time'].values[i: i + window_size]
        day = df_prediction['Day'].values[i: i + window_size]
        gls = df_prediction['Glucose'].values[i: i + window_size]

        xs = df_prediction['axis1'].values[i: i + window_size]
        ys = df_prediction['axis2'].values[i: i + window_size]
        zs = df_prediction['axis3'].values[i: i + window_size]

        label = stats.mode(df_prediction['fasting_state'][i: i + window_size])[0][0]

        df_slice = pd.DataFrame(time, columns=['Time'])
        df_slice["Day"] = day
        df_slice["Glucose"] = gls
        df_slice["axis1"] = xs
        df_slice["axis2"] = ys
        df_slice["axis3"] = zs

        x_list_pred.append(df_slice)

        prediction_labels.append(label)

    return x_list_pred, prediction_labels


def windowing_ml_acc(df_prediction, window_size, step_size):

    x_list_pred = []
    prediction_labels = []

    # creating overlaping windows of size window-size 100
    for i in range(0, df_prediction.shape[0] - window_size, step_size):
        time = df_prediction['Time'].values[i: i + window_size]
        day = df_prediction['Day'].values[i: i + window_size]

        xs = df_prediction['axis1'].values[i: i + window_size]
        ys = df_prediction['axis2'].values[i: i + window_size]
        zs = df_prediction['axis3'].values[i: i + window_size]

        label = stats.mode(df_prediction['fasting_state'][i: i + window_size])[0][0]

        df_slice = pd.DataFrame(time, columns=['Time'])
        df_slice["Day"] = day
        df_slice["axis1"] = xs
        df_slice["axis2"] = ys
        df_slice["axis3"] = zs

        x_list_pred.append(df_slice)

        prediction_labels.append(label)

    return x_list_pred, prediction_labels


def windowing_ml_gl(df_prediction, window_size, step_size):

    x_list_pred = []
    prediction_labels = []

    # creating overlaping windows of size window-size 100
    for i in range(0, df_prediction.shape[0] - window_size, step_size):
        time = df_prediction['Time'].values[i: i + window_size]
        day = df_prediction['Day'].values[i: i + window_size]
        gls = df_prediction['Glucose'].values[i: i + window_size]

        label = stats.mode(df_prediction['fasting_state'][i: i + window_size])[0][0]

        df_slice = pd.DataFrame(time, columns=['Time'])
        df_slice["Day"] = day
        df_slice["Glucose"] = gls

        x_list_pred.append(df_slice)

        prediction_labels.append(label)

    return x_list_pred, prediction_labels


def feat_statistical_measures_gl_ml(x_list):
    X_train = pd.DataFrame()

    for df_temp in x_list:
        cgm_summary = list(cgm.summary(df_temp))
        cgm_LBGI = cgm.LBGI(df_temp)
        cgm_HBGI = cgm.HBGI(df_temp)
        cgm_ADRR = cgm.ADRR(df_temp)
        cgm_GMI = cgm.GMI(df_temp)
        cgm_J_index = cgm.J_index(df_temp)
        cgm_eA1c = cgm.eA1c(df_temp)
        cgm_interdaysd = cgm.interdaysd(df_temp)
        cgm_interdaycv = cgm.interdaycv(df_temp)
        cgm_TOR = cgm.TOR(df_temp, sd=1, sr=15)
        cgm_TIR = cgm.TIR(df_temp, sd=1, sr=15)
        cgm_POR = cgm.POR(df_temp, sd=1, sr=15)

        cgm_summary.append(cgm_LBGI)
        cgm_summary.append(cgm_HBGI)
        cgm_summary.append(cgm_ADRR)
        cgm_summary.append(cgm_GMI)
        cgm_summary.append(cgm_J_index)
        cgm_summary.append(cgm_eA1c)
        cgm_summary.append(cgm_interdaysd)
        cgm_summary.append(cgm_interdaycv)
        cgm_summary.append(cgm_TOR)
        cgm_summary.append(cgm_TIR)
        cgm_summary.append(cgm_POR)

        X_train_temp = pd.DataFrame([cgm_summary],
                                    columns=["mean", "median", "minimum", "maximum", "first_quartile", "third_quartile",
                                             "LBGI", "HBGI", "ADRR", "GMI", "J_index", "eA1c", "interdaysd",
                                             "cgm_interdaycv",
                                             # "cgm_intradaysd", "cgm_intradaycv",
                                             "cgm_TOR", "cgm_TIR",
                                             "cgm_POR"])

        X_train = pd.concat([X_train, X_train_temp], ignore_index=True)

    return X_train


def feat_statistical_measures_gl_acc_ml(x_list):
    X_train = pd.DataFrame()

    for df_temp in x_list:
        cgm_summary = list(cgm.summary(df_temp))
        cgm_J_index = cgm.J_index(df_temp)
        cgm_interdaycv = cgm.interdaycv(df_temp)

        cgm_summary.append(cgm_J_index)
        cgm_summary.append(cgm_interdaycv)

        z_mean = df_temp["axis3"].mean()
        y_max = df_temp["axis2"].max()
        z_energy = ((df_temp["axis3"] ** 2) / 100).sum()

        acc_summary = []
        acc_summary.extend(
            [z_mean, y_max, z_energy])

        features = cgm_summary
        features.extend(acc_summary)

        X_train_temp = pd.DataFrame([features],
                                    columns=["mean", "median", "minimum", "maximum", "first_quartile", "third_quartile",
                                             "J_index", "cgm_interdaycv", "z_mean", "y_max", "z_energy"])

        X_train = pd.concat([X_train, X_train_temp], ignore_index=True)

    return X_train


def feat_statistical_measures_acc_ml(x_list):
    X_train = pd.DataFrame()

    for df_temp in x_list:
        # mean
        z_mean = df_temp["axis3"].mean()
        y_max = df_temp["axis2"].max()
        z_energy = ((df_temp["axis3"] ** 2) / 100).sum()

        acc_summary = []
        acc_summary.extend(
            [z_mean, y_max, z_energy])

        X_train_temp = pd.DataFrame([acc_summary],
                                    columns=["z_mean", "y_max", "z_energy"])

        X_train = pd.concat([X_train, X_train_temp], ignore_index=True)

    return X_train


def feat_statistical_measures_gl_acc_tsc(x_list):
    X_train = pd.DataFrame()

    for df_temp in x_list:
        time_median = df_temp.loc[1,'Time']
        cgm_summary = list(cgm.summary(df_temp))
        cgm_J_index = cgm.J_index(df_temp)

        cgm_interdaycv = cgm.interdaycv(df_temp)
        cgm_summary.append(cgm_J_index)
        cgm_summary.append(cgm_interdaycv)

        # mean
        z_mean = df_temp["axis3"].mean()

        # max
        y_max = df_temp["axis2"].max()

        # energy
        z_energy = ((df_temp["axis3"] ** 2) / 100).sum()

        general = []
        general.extend([time_median])

        acc_summary = []
        acc_summary.extend([z_mean, y_max, z_energy])

        features = general
        features.extend(cgm_summary)
        features.extend(acc_summary)

        X_train_temp = pd.DataFrame([features], columns=["time", "mean", "median", "minimum", "maximum",
                                                         "first_quartile", "third_quartile",
                                                         "J_index", "cgm_interdaycv",
                                                         "z_mean", "y_max", "z_energy"])

        X_train = pd.concat([X_train, X_train_temp], ignore_index=True)

    return X_train


def feat_statistical_measures_gl_tsc(x_list):
    X_train = pd.DataFrame()

    for df_temp in x_list:
        time_median = df_temp.loc[1,'Time']
        cgm_summary = list(cgm.summary(df_temp))
        cgm_J_index = cgm.J_index(df_temp)

        cgm_interdaycv = cgm.interdaycv(df_temp)
        cgm_summary.append(cgm_J_index)
        cgm_summary.append(cgm_interdaycv)

        general = []
        general.extend([time_median])
        general.extend(cgm_summary)

        X_train_temp = pd.DataFrame([general],
                                    columns=["time","mean", "median", "minimum", "maximum", "first_quartile", "third_quartile",
                                             "J_index", "cgm_interdaycv"])

        X_train = pd.concat([X_train, X_train_temp], ignore_index=True)

    return X_train


def feat_statistical_measures_acc_tsc(x_list):
    X_train = pd.DataFrame()

    for df_temp in x_list:
        time_median = df_temp.loc[1,'Time']

        # mean

        z_mean = df_temp["axis3"].mean()
        y_max = df_temp["axis2"].max()
        z_energy = ((df_temp["axis3"] ** 2) / 100).sum()

        general = []
        general.extend([time_median])

        acc_summary = []
        acc_summary.extend(
            [z_mean, y_max, z_energy])

        features = general
        features.extend(acc_summary)

        X_train_temp = pd.DataFrame([features],
                                    columns=["time",
                                             "z_mean", "y_max", "z_energy"])

        X_train = pd.concat([X_train, X_train_temp], ignore_index=True)

    return X_train


def to_float(input_string):
    if input_string == "fasting":
        return 0
    else:
        return 1

