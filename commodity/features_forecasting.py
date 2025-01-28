import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from pmdarima.arima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import optuna
import random
from tqdm import tqdm
import os
import logging
import warnings


from result import calculating_metric

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("prophet").propagate = False
warnings.filterwarnings("ignore")


# Data Settig
valid_length = 3  # fix this
test_length = 3  # fix this
future_length = 7  # fix this


def _expand_ts(df_input):
    df_future = df_input.copy()
    last_date = df_future["dt"].iloc[-1]
    # 향후 날짜 생성
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_length, freq="MS")
    additional_df = pd.DataFrame({"dt": future_dates})
    df_future = pd.concat([df_future, additional_df], ignore_index=True)
    return df_future


def _merging_pred_actual(df_actual, df_pred):
    df_actual = df_actual.rename(columns={"y": "Actual"})
    df_output = pd.concat([df_actual, df_pred], axis=1)
    return df_output


def _preprocessing_ARIMA(df_input, target):
    df_input = df_input[["dt", target]]
    df_input = df_input.rename(columns={target: "y"})
    # Train Set
    df_train = df_input.iloc[: -(test_length + future_length)]
    # Test Set
    df_test = df_input.iloc[-(test_length + future_length) : -future_length]
    # Future Set
    df_future = df_input.iloc[-future_length:]
    return df_train, df_test, df_future


def _predicting_ARIMA(df_expanded, df_x):
    model_name = "ARIMA"
    print(f"Feature Forecasting - {model_name}")

    WhatToPredict = df_x.columns

    # Reset Result DF
    df_best_params_result = pd.DataFrame()
    df_metric_test_result = pd.DataFrame()
    df_pred_values_result = pd.DataFrame()

    for target_idx in tqdm(WhatToPredict):
        # df_tmp = _expand_ts[["dt", target_idx]]
        df_train, df_test, df_future = _preprocessing_ARIMA(df_expanded, target_idx)
        # Model
        model = auto_arima(
            df_train["y"],
            start_p=0,
            start_q=0,
            max_p=3,
            max_q=3,
            seasonal=True,
            m=12,
            max_P=3,
            max_Q=3,
            start_d=0,
            max_d=2,
            start_D=0,
            max_D=2,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=False,
        )

        # Prediction
        pred = model.predict(len(df_test) + len(df_future))
        pred = pred.apply(lambda x: max(x, 0))
        # Prediction - test
        pred_test = pred[: len(df_test)]
        df_pred_test = pd.DataFrame(pred_test)
        df_pred_test = df_pred_test.rename(columns={"index": "ts", 0: "Prediction"})
        # Prediction - future
        pred_future = pred[len(df_test) : (len(df_test) + len(df_future))]
        pred_future = pd.DataFrame(pred_future)
        pred_future = pred_future.rename(columns={"index": "ts", 0: "Prediction"})

        # Metric
        df_pred_actual_test = _merging_pred_actual(df_test, df_pred_test)
        df_metric_test = calculating_metric(df_pred_actual_test, target_idx)
        df_metric_test["Model"] = model_name

        # Prediction Value
        df_pred_future = pd.concat([df_future, pred_future], axis=1)
        df_pred_future = df_pred_future.drop(["y"], axis=1)
        df_pred_future["Target"] = target_idx
        df_pred_future["Model"] = model_name

        # Parameters
        best_params = model.get_params()
        df_best_params = pd.DataFrame([best_params])
        df_best_params["Target"] = target_idx

        # Result Saving
        df_best_params_result = pd.concat([df_best_params_result, df_best_params])
        df_metric_test_result = pd.concat([df_metric_test_result, df_metric_test])
        df_pred_values_result = pd.concat([df_pred_values_result, df_pred_future])

    # Result Saving
    save_path = f"./output/FeaturesForecasting/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df_best_params_result.to_csv(save_path + f"{model_name}_Parameters.csv", index=True)
    df_metric_test_result.to_csv(save_path + f"{model_name}_Metric.csv", index=True)
    df_pred_values_result.to_csv(save_path + f"{model_name}_Prediction.csv", index=True)
    print("ARIMA Done")

    return df_metric_test_result, df_pred_values_result


def _preprocessing_Prophet(df_input, target):
    df_input = df_input[["dt", target]]
    df_input = df_input.rename(columns={"dt": "ds", target: "y"})
    # Train Set
    df_train = df_input.iloc[: -(valid_length + test_length + future_length)]
    df_train = df_train.reset_index().drop(["index"], axis=1)
    # Validation Set
    df_valid = df_input.iloc[-(valid_length + test_length + future_length) : -(test_length + future_length)]
    df_valid = df_valid.reset_index().drop(["index"], axis=1)
    # Test Set
    df_test = df_input.iloc[-(test_length + future_length) : -future_length]
    df_test = df_test.reset_index().drop(["index"], axis=1)
    # Future Set
    df_future = df_input.iloc[-future_length:]
    df_future = df_future.reset_index().drop(["index"], axis=1)
    return df_train, df_valid, df_test, df_future


def _predicting_Prophet(df_expanded, df_x):
    model_name = "Prophet"
    print(f"Feature Forecasting - {model_name}")

    WhatToPredict = df_x.columns

    # Reset Result DF
    df_best_params_result = pd.DataFrame()
    df_metric_test_result = pd.DataFrame()
    df_pred_values_result = pd.DataFrame()

    for target_idx in tqdm(WhatToPredict):
        df_tmp = df_expanded[["dt", target_idx]]
        df_train, df_valid, df_test, df_future = _preprocessing_Prophet(df_tmp, target_idx)

        def objective(trial):
            params = {
                "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.005, 5),
                "changepoint_range": trial.suggest_float("changepoint_range", 0.7, 0.9),
                "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 20),
                "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["multiplicative", "additive"]),
                "yearly_seasonality": trial.suggest_int("yearly_seasonality", 1, 30),
            }
            # Model
            model_optuna = Prophet(**params)
            model_optuna.fit(df_train)
            # Prediction
            pred_optuna = model_optuna.predict(df_valid)
            # Loss Function
            mse = mean_squared_error(df_valid["y"], pred_optuna["yhat"])
            return mse

        # HyperParameters Tuning w/ Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        # Best HyperParameters from the optimization
        best_params = study.best_params

        # Modeling
        model = Prophet(**best_params)
        model.fit(df_train)

        # Prediction - test
        pred_test = model.predict(df_test)
        pred_test = pred_test.loc[:, ["ds", "yhat"]]
        pred_test.loc[pred_test["yhat"] < 0, "yhat"] = 0
        df_pred_test = pred_test.rename(columns={"yhat": "Prediction"})
        # Prediction - future
        pred_future = model.predict(df_future)
        pred_future = pred_future.loc[:, ["ds", "yhat"]]
        pred_future.loc[pred_future["yhat"] < 0, "yhat"] = 0
        df_pred_future = pred_future.rename(columns={"yhat": "Prediction"})

        # Metric
        df_pred_actual_test = _merging_pred_actual(df_test, df_pred_test)
        df_metric_test = calculating_metric(df_pred_actual_test, target_idx)
        df_metric_test["Model"] = model_name

        # Prediction Value
        df_pred_future = pd.concat([df_future, df_pred_future], axis=1)
        df_pred_future = df_pred_future.drop(["y"], axis=1)
        df_pred_future = df_pred_future.iloc[:, 1:]
        df_pred_future = df_pred_future.rename(columns={"ds": "dt"})
        df_pred_future["Target"] = target_idx
        df_pred_future["Model"] = model_name

        # Parameters
        df_best_params = pd.DataFrame([best_params])
        df_best_params["Target"] = target_idx

        # Result Saving
        df_best_params_result = pd.concat([df_best_params_result, df_best_params])
        df_metric_test_result = pd.concat([df_metric_test_result, df_metric_test])
        df_pred_values_result = pd.concat([df_pred_values_result, df_pred_future])
    
    # Result Saving
    save_path = f"./output/FeaturesForecasting/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df_best_params_result.to_csv(save_path + f"{model_name}_Parameters.csv", index=True)
    df_metric_test_result.to_csv(save_path + f"{model_name}_Metric.csv", index=True)
    df_pred_values_result.to_csv(save_path + f"{model_name}_Prediction.csv", index=True)
    print("Prophet Done")

    return df_metric_test_result, df_pred_values_result


def _create_sequences(data, window_size):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(x), np.array(y)


def _preprocessing(df_input, target):
    df_input = df_input.rename(columns={target: "Actual"})
    # Train Set
    df_train = df_input.iloc[: -(test_length + future_length)]
    df_train = df_train.reset_index()
    # Test Set
    df_test = df_input.iloc[-(test_length + future_length) : -future_length]
    df_test = df_test.reset_index()
    # Future Set
    df_future = df_input.iloc[-future_length:]
    df_future = df_future.reset_index()
    return df_train, df_test, df_future


def _predicting_sliding_window(model, scaler, x_set, forecasting_length, window_size):
    predictions = []
    current_input = x_set[0]

    for i in range(forecasting_length):
        current_prediction = model.predict(
            current_input.reshape(1, window_size, x_set.shape[2]), verbose=0
        )
        predictions.append(current_prediction[0])
        current_input = np.append(current_input[1:], current_prediction, axis=0)
    
    pred = scaler.inverse_transform(predictions)  # descaling
    return pred


def _preprocessing_LSTM(x, y):
    # Train Set
    x_train = x[: -(test_length + future_length)]
    y_train = y[: -(test_length + future_length)]
    # Test Set
    x_test = x[-(test_length + future_length) : -future_length]
    y_test = y[-(test_length + future_length) : -future_length]
    # Future Set
    x_future = x[-future_length:]
    y_future = y[-future_length:]
    return x_train, y_train, x_test, y_test, x_future, y_future


def _predicting_LSTM(df_expanded, df_x):
    model_name = "LSTM"

    # Model HyperParameters
    units_num = 100
    epochs_num = 50
    batch_num = 32

    # window_size
    window_size = test_length

    # Seed Setting
    user_seed = 42
    np.random.seed(user_seed)
    tf.random.set_seed(user_seed)
    random.seed(user_seed)

    print(f"Feature Forecasting - {model_name}")

    WhatToPredict = df_x.columns

    # Reset Result DF
    df_best_params_result = pd.DataFrame()
    df_metric_test_result = pd.DataFrame()
    df_pred_values_result = pd.DataFrame()

    for target_idx in tqdm(WhatToPredict):
        df_tmp = df_expanded[["dt", target_idx]]
        df_tmp = df_tmp.set_index("dt")
        df_train, df_test, df_future = _preprocessing(df_tmp, target_idx)

        # Scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_tmp)
        # Making Sequence
        x, y = _create_sequences(scaled_data, window_size)
        # Data Split
        x_train, y_train, x_test, y_test, x_future, y_future = _preprocessing_LSTM(x, y)

        # Model
        model = Sequential()
        model.add(
            LSTM(
                units=units_num,
                input_shape=(x_train.shape[1], x_train.shape[2]),
                return_sequences=True,
            )
        )
        model.add(Dropout(0.2))
        model.add(LSTM(units=units_num))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        # Compiling
        model.compile(optimizer="adam", loss="mean_squared_error")
        # Fitting
        model.fit(x_train, y_train, epochs=epochs_num, batch_size=batch_num, verbose=0)

        # Prediction
        pred = _predicting_sliding_window(model, scaler, x_test, (len(x_test) + len(x_future)), window_size)
        # Prediction - test
        pred_test = pred[: len(x_test)]
        df_pred_test = pd.DataFrame(pred_test)
        df_pred_test = df_pred_test.rename(columns={0: "Prediction"})
        df_pred_test["Target"] = target_idx
        df_pred_test["Model"] = model_name
        df_test_ts = pd.DataFrame(df_expanded.iloc[-(test_length + future_length) : -(future_length),]["dt"])
        df_test_ts = df_test_ts.reset_index().drop(["index"], axis=1)
        df_pred_test = pd.concat([df_test_ts, df_pred_test], axis=1)
        # Prediction - future
        pred_future = pred[len(x_test) : (len(x_test) + len(x_future))]
        df_pred_future = pd.DataFrame(pred_future)
        df_pred_future = df_pred_future.rename(columns={0: "Prediction"})
        df_pred_future["Target"] = target_idx
        df_pred_future["Model"] = model_name
        df_future_ts = pd.DataFrame(df_expanded.iloc[-(future_length):,]["dt"])
        df_future_ts = df_future_ts.reset_index().drop(["index"], axis=1)
        df_pred_future = pd.concat([df_future_ts, df_pred_future], axis=1)

        # Metric
        df_pred_actual_test = _merging_pred_actual(df_test, df_pred_test)
        df_metric_test = calculating_metric(df_pred_actual_test, target_idx)
        df_metric_test["Model"] = model_name

        # Result Saving
        df_metric_test_result = pd.concat([df_metric_test_result, df_metric_test])
        df_pred_values_result = pd.concat([df_pred_values_result, df_pred_future])

    # Result Saving
    save_path = f"./output/FeaturesForecasting/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df_best_params_result.to_csv(save_path + f"{model_name}_Parameters.csv", index=True)
    df_metric_test_result.to_csv(save_path + f"{model_name}_Metric.csv", index=True)
    df_pred_values_result.to_csv(save_path + f"{model_name}_Prediction.csv", index=True)
    print("LSTM Done")

    return df_metric_test_result, df_pred_values_result


def _predicting_BiLSTM(df_expanded, df_x):
    model_name = "BI-LSTM"

    # Model HyperParameters
    units_num = 100
    epochs_num = 50
    batch_num = 32

    # window_size
    window_size = test_length

    # Seed Setting
    user_seed = 42
    np.random.seed(user_seed)
    tf.random.set_seed(user_seed)
    random.seed(user_seed)

    print(f"Feature Forecasting - {model_name}")

    WhatToPredict = df_x.columns

    # Reset Result DF
    df_best_params_result = pd.DataFrame()
    df_metric_test_result = pd.DataFrame()
    df_pred_values_result = pd.DataFrame()

    for target_idx in tqdm(WhatToPredict):
        df_tmp = df_expanded[["dt", target_idx]]
        df_tmp = df_tmp.set_index("dt")
        df_train, df_test, df_future = _preprocessing(df_tmp, target_idx)

        # Scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_tmp)
        # Making Sequence
        x, y = _create_sequences(scaled_data, window_size)
        # Data Split
        x_train, y_train, x_test, y_test, x_future, y_future = _preprocessing_LSTM(x, y)

        # Model
        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(
                    units=units_num,
                    input_shape=(x_train.shape[1], x_train.shape[2]),
                    return_sequences=True,
                )
            )
        )
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(units=units_num)))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        # Compiling
        model.compile(optimizer="adam", loss="mean_squared_error")
        # Fitting
        model.fit(x_train, y_train, epochs=epochs_num, batch_size=batch_num, verbose=0)

        # Prediction
        pred = _predicting_sliding_window(model, scaler, x_test, (len(x_test) + len(x_future)), window_size)
        # Prediction - test
        pred_test = pred[: len(x_test)]
        df_pred_test = pd.DataFrame(pred_test)
        df_pred_test = df_pred_test.rename(columns={0: "Prediction"})
        df_pred_test["Target"] = target_idx
        df_pred_test["Model"] = model_name
        df_test_ts = pd.DataFrame(df_expanded.iloc[-(test_length + future_length) : -(future_length),]["dt"])
        df_test_ts = df_test_ts.reset_index().drop(["index"], axis=1)
        df_pred_test = pd.concat([df_test_ts, df_pred_test], axis=1)
        # Prediction - future
        pred_future = pred[len(x_test) : (len(x_test) + len(x_future))]
        df_pred_future = pd.DataFrame(pred_future)
        df_pred_future = df_pred_future.rename(columns={0: "Prediction"})
        df_pred_future["Target"] = target_idx
        df_pred_future["Model"] = model_name
        df_future_ts = pd.DataFrame(df_expanded.iloc[-(future_length):,]["dt"])
        df_future_ts = df_future_ts.reset_index().drop(["index"], axis=1)
        df_pred_future = pd.concat([df_future_ts, df_pred_future], axis=1)

        # Metric
        df_pred_actual_test = _merging_pred_actual(df_test, df_pred_test)
        df_metric_test = calculating_metric(df_pred_actual_test, target_idx)
        df_metric_test["Model"] = model_name

        # Result Saving
        df_metric_test_result = pd.concat([df_metric_test_result, df_metric_test])
        df_pred_values_result = pd.concat([df_pred_values_result, df_pred_future])

    # Result Saving
    save_path = f"./output/FeaturesForecasting/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df_best_params_result.to_csv(save_path + f"{model_name}_Parameters.csv", index=True)
    df_metric_test_result.to_csv(save_path + f"{model_name}_Metric.csv", index=True)
    df_pred_values_result.to_csv(save_path + f"{model_name}_Prediction.csv", index=True)
    print("BI-LSTM Done")

    return df_metric_test_result, df_pred_values_result


def _predicting_GRU(df_expanded, df_x):
    model_name = "GRU"

    # Model HyperParameters
    units_num = 100
    epochs_num = 50
    batch_num = 32

    # window_size
    window_size = test_length

    # Seed Setting
    user_seed = 42
    np.random.seed(user_seed)
    tf.random.set_seed(user_seed)
    random.seed(user_seed)

    print(f"Feature Forecasting - {model_name}")

    WhatToPredict = df_x.columns

    # Reset Result DF
    df_best_params_result = pd.DataFrame()
    df_metric_test_result = pd.DataFrame()
    df_pred_values_result = pd.DataFrame()

    for target_idx in tqdm(WhatToPredict):
        df_tmp = df_expanded[["dt", target_idx]]
        df_tmp = df_tmp.set_index("dt")
        df_train, df_test, df_future = _preprocessing(df_tmp, target_idx)

        # Scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_tmp)
        # Making Sequence
        x, y = _create_sequences(scaled_data, window_size)
        # Data Split
        x_train, y_train, x_test, y_test, x_future, y_future = _preprocessing_LSTM(x, y)

        # Model
        model = Sequential()
        model.add(
            GRU(
                units=units_num,
                input_shape=(x_train.shape[1], x_train.shape[2]),
                return_sequences=True,
            )
        )
        model.add(Dropout(0.2))
        model.add(GRU(units=units_num))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        # Compiling
        model.compile(optimizer="adam", loss="mean_squared_error")
        # Fitting
        model.fit(x_train, y_train, epochs=epochs_num, batch_size=batch_num, verbose=0)

        # Prediction
        pred = _predicting_sliding_window(model, scaler, x_test, (len(x_test) + len(x_future)), window_size)
        # Prediction - test
        pred_test = pred[: len(x_test)]
        df_pred_test = pd.DataFrame(pred_test)
        df_pred_test = df_pred_test.rename(columns={0: "Prediction"})
        df_pred_test["Target"] = target_idx
        df_pred_test["Model"] = model_name
        df_test_ts = pd.DataFrame(df_expanded.iloc[-(test_length + future_length) : -(future_length),]["dt"])
        df_test_ts = df_test_ts.reset_index().drop(["index"], axis=1)
        df_pred_test = pd.concat([df_test_ts, df_pred_test], axis=1)
        # Prediction - future
        pred_future = pred[len(x_test) : (len(x_test) + len(x_future))]
        df_pred_future = pd.DataFrame(pred_future)
        df_pred_future = df_pred_future.rename(columns={0: "Prediction"})
        df_pred_future["Target"] = target_idx
        df_pred_future["Model"] = model_name
        df_future_ts = pd.DataFrame(df_expanded.iloc[-(future_length):,]["dt"])
        df_future_ts = df_future_ts.reset_index().drop(["index"], axis=1)
        df_pred_future = pd.concat([df_future_ts, df_pred_future], axis=1)

        # Metric
        df_pred_actual_test = _merging_pred_actual(df_test, df_pred_test)
        df_metric_test = calculating_metric(df_pred_actual_test, target_idx)
        df_metric_test["Model"] = model_name

        # Result Saving
        df_metric_test_result = pd.concat([df_metric_test_result, df_metric_test])
        df_pred_values_result = pd.concat([df_pred_values_result, df_pred_future])

    # Result Saving
    save_path = f"./output/FeaturesForecasting/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df_best_params_result.to_csv(save_path + f"{model_name}_Parameters.csv", index=True)
    df_metric_test_result.to_csv(save_path + f"{model_name}_Metric.csv", index=True)
    df_pred_values_result.to_csv(save_path + f"{model_name}_Prediction.csv", index=True)
    print("GRU Done")

    return df_metric_test_result, df_pred_values_result


def _finding_best(df_expanded, df_x):
    WhatToPredict = df_x.columns

    # Load Metric Data
    df_all_metric_result = pd.DataFrame()
    models = ["ARIMA", "Prophet", "LSTM", "BI-LSTM", "GRU"]

    for model_idx in models:
        base_path = f"./output/FeaturesForecasting/"
        file_name = f"{model_idx}_Metric.csv"
        df_tmp = pd.read_csv(base_path + file_name)
        df_all_metric_result = pd.concat([df_all_metric_result, df_tmp], axis=0)

    df_all_metric_result = df_all_metric_result.rename(columns={"Unnamed: 0": "Target"})

    # Load Prediction Data
    df_all_Prediction_result = pd.DataFrame()

    for model_idx in models:
        base_path = f"./output/FeaturesForecasting/"
        file_name = f"{model_idx}_Prediction.csv"
        df_tmp = pd.read_csv(base_path + file_name)
        df_all_Prediction_result = pd.concat([df_all_Prediction_result, df_tmp], axis=0)
    df_all_Prediction_result = df_all_Prediction_result.drop(["Unnamed: 0"], axis=1)

    # Find Best Model & Prediction
    df_best_metric_result = pd.DataFrame()
    df_pred_values_result = pd.DataFrame()

    df_all_metric_result = df_all_metric_result.reset_index().drop(["index"], axis=1)
    for target_idx in WhatToPredict:
        # Best Metric
        df_tmp = df_all_metric_result[df_all_metric_result["Target"] == target_idx]
        df_tmp_sorted = df_tmp.sort_values(by="nRMSE(Max-Min)")
        df_best_metric = df_tmp_sorted.head(1)
        df_best_metric_result = pd.concat([df_best_metric_result, df_best_metric])
        # Best Model
        best_model = df_best_metric["Model"].iloc[0]
        finding_target = df_best_metric["Target"].iloc[0]
        # Prediction of Best Model
        df_best_prediction = df_all_Prediction_result[
            (df_all_Prediction_result["Target"] == finding_target)
            & (df_all_Prediction_result["Model"] == best_model)
        ]
        df_pred_values_result = pd.concat([df_pred_values_result, df_best_prediction])

        # preprocessing - Update
        df_best_prediction["dt"] = pd.to_datetime(df_best_prediction["dt"])
        df_tmp_update = df_best_prediction[["dt", "Prediction"]]
        df_tmp_update = df_tmp_update.rename(columns={"Prediction": target_idx})
        # Update
        df_expanded = df_expanded.merge(df_tmp_update[["dt", target_idx]], on="dt", how="left")
        df_expanded[target_idx] = df_expanded[f"{target_idx}_y"].combine_first(df_expanded[f"{target_idx}_x"])
        df_expanded = df_expanded.drop([f"{target_idx}_x", f"{target_idx}_y"], axis=1)

    # Saving Result
    save_path = f"./output/FeaturesForecasting/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df_expanded.to_csv(save_path + f"_Result_FeatureForecasting.csv", index=False)
    df_best_metric_result.to_csv(save_path + f"_BestModel_Metric.csv", index=False)
    df_pred_values_result.to_csv(save_path + f"_BestModel_Prediction.csv", index=False)
    print("Feature Forecasting - Done")

    return df_expanded


def forecasting_features(df, target_list):
    print("Feature Forecasting - Start")
    df_expanded = _expand_ts(df)
    df_x = df_expanded.drop(target_list, axis=1).drop("dt", axis=1)
    save_path = f"./output/FeaturesForecasting/"

    # ARIMA
    if not os.path.exists(save_path + "ARIMA_Prediction.csv"):
        _predicting_ARIMA(df_expanded, df_x)
    else:
        print("'ARIMA' Result already exists. Skipping Forecasting.")

    # Prophet
    if not os.path.exists(save_path + "Prophet_Prediction.csv"):
        _predicting_Prophet(df_expanded, df_x)
    else:
        print("'Prophet' Result already exists. Skipping Forecasting.")

    # LSTM
    if not os.path.exists(save_path + "LSTM_Prediction.csv"):
        _predicting_LSTM(df_expanded, df_x)
    else:
        print("'LSTM' Result already exists. Skipping Forecasting.")

    # BI-LSTM
    if not os.path.exists(save_path + "BI-LSTM_Prediction.csv"):
        _predicting_BiLSTM(df_expanded, df_x)
    else:
        print("'BI-LSTM' Result already exists. Skipping Forecasting.")

    # GRU
    if not os.path.exists(save_path + "GRU_Prediction.csv"):
        _predicting_GRU(df_expanded, df_x)
    else:
        print("'GRU' Result already exists. Skipping Forecasting.")

    # Find Best Model & Prediction Value
    if not os.path.exists(save_path + "_Result_FeatureForecasting.csv"):
        df_expanded = _finding_best(df_expanded, df_x)
    else:
        df_expanded = pd.read_csv(save_path + "_Result_FeatureForecasting.csv")
        print("'Feature Forecasting' Result already exists. Skipping Forecasting.")

    print("Feature Forecasting - End")
    
    return df_expanded
