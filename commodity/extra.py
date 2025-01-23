import os
import warnings
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import optuna
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

warnings.filterwarnings("ignore")


def pre_processing(df, target, lag_number):
    # "date"를 월의 첫 날짜로 변경 - 추후 삭제
    df["Date"] = (pd.to_datetime(df["Date"]) - pd.offsets.MonthEnd() + pd.offsets.MonthBegin(1))
    # Rename - 추후 삭제
    df = df.rename(columns={"Date": "dt"})

    # Target(y)
    target_list = [
        "IronOre",
        "Nickel",
        "Coal",
        "CokingCoal",
        "Steel",
        "Copper",
        "Aluminum",
    ]
    df_y = pd.DataFrame(df[target])

    # Features(X)
    df_x = df.drop(target_list, axis=1).drop("dt", axis=1)

    # TS Features
    df_ts = pd.DataFrame(df["dt"])
    df_ts["Year"] = df_ts["dt"].dt.year
    df_ts["Month"] = df_ts["dt"].dt.month
    # df_ts['WeekByYear'] = df_ts['dt'].dt.isocalendar().week  # for Weekly
    # df_ts['WeekByMonth'] = df_ts['dt'].apply(lambda x: (x.day - 1) // 7 + 1)  # for Weekly

    # Lag Features
    df_lag = making_lag_features(lag_number, df_y, target)

    # Concat
    df_processed = pd.concat([df_ts, df_lag, df_x], axis=1)
    df_processed = df_processed.iloc[lag_number:].reset_index().drop(["index"], axis=1)
    return df_processed, df_lag


def target_split(df_input, target):
    df_input = df_input.drop(["dt"], axis=1)
    df_input_x = df_input.drop([target], axis=1)
    df_input_y = df_input[target].to_frame()
    return df_input_x, df_input_y


def data_split(df_input, target, valid_length, test_length):
    # Train Set
    df_train = df_input.iloc[: -(valid_length + test_length)]
    x_train, y_train = target_split(df_train, target)
    # Valid Set
    df_valid = df_input.iloc[-(valid_length + test_length) : -test_length]
    x_valid, y_valid = target_split(df_valid, target)
    # Test Set
    df_test = df_input.iloc[-test_length:]
    x_test, y_test = target_split(df_test, target)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def making_lag_features(lag_num, df_target, target_name):
    lag_columns = [f"lag_{i}" for i in range(1, lag_num + 1)]
    diff_columns = [f"delta_lag{i}_lag{i + 1}" for i in range(1, lag_num)]
    mean_columns = [f"mean_window_{window_size}" for window_size in [3, 6]]
    std_columns = [f"std_window_{window_size}" for window_size in [3, 6]]
    all_columns = lag_columns + diff_columns + mean_columns + std_columns
    df_lag_result = pd.DataFrame(columns=all_columns)
    df_lag_result = pd.concat([df_target, df_lag_result])
    # fill "lag"
    for lag in range(1, lag_num + 1):
        df_lag_result[f"lag_{lag}"] = df_lag_result[target_name].shift(lag)
    # fill "delta_lag"
    for i in range(1, lag_num):
        df_lag_result[f"delta_lag{i}_lag{i + 1}"] = (df_lag_result[f"lag_{i + 1}"] - df_lag_result[f"lag_{i}"]) / df_lag_result[f"lag_{i + 1}"]
    # fill "mean_window"
    df_lag_result["mean_window_3"] = df_lag_result[["lag_1", "lag_2", "lag_3"]].mean(axis=1)
    df_lag_result["mean_window_6"] = df_lag_result[lag_columns].mean(axis=1)
    # fill "std_window"
    df_lag_result["std_window_3"] = df_lag_result[["lag_1", "lag_2", "lag_3"]].std(axis=1)
    df_lag_result["std_window_6"] = df_lag_result[lag_columns].std(axis=1)
    return df_lag_result


def prediction(model, df_input):
    pred = model.predict(df_input)
    df_pred = pd.DataFrame(pred)
    df_pred[df_pred < 0] = 0
    df_pred = df_pred.rename(columns={0: "Prediction"})
    return df_pred


def pred_actual(df_actual, df_pred, target_name):
    # Actual
    df_actual = df_actual.rename(columns={target_name: "Actual"})
    df_actual = df_actual.reset_index().drop(["index"], axis=1)
    # Concat
    df_output = pd.concat([df_actual, df_pred], axis=1)
    return df_output


def metric(df_metric, idx_name):
    df_result = pd.DataFrame(
        columns=[
            "RMSE",
            "nRMSE(Mean)",
            "nRMSE(Max-Min)",
            "MAE",
            "nMAE(Mean)",
            "nMAE(Max-Min)",
            "MAPE",
            "sMAPE",
            "R2",
        ]
    )
    # RMSE
    rmse = np.sqrt(mean_squared_error(df_metric["Actual"], df_metric["Prediction"]))
    df_result.loc[idx_name, "RMSE"] = round(rmse, 2)
    # nRMSE
    nrmse_mean = np.sqrt(mean_squared_error(df_metric["Actual"], df_metric["Prediction"])) / np.mean(df_metric["Actual"])
    df_result.loc[idx_name, "nRMSE(Mean)"] = round(nrmse_mean, 4)
    nrmse_minmax = np.sqrt(mean_squared_error(df_metric["Actual"], df_metric["Prediction"])) / (df_metric["Actual"].max() - df_metric["Actual"].min())
    df_result.loc[idx_name, "nRMSE(Max-Min)"] = round(nrmse_minmax, 4)
    # MAE
    mae = mean_absolute_error(df_metric["Actual"], df_metric["Prediction"])
    df_result.loc[idx_name, "MAE"] = round(mae, 2)
    # nMAE
    nmae_mean = mean_absolute_error(df_metric["Actual"], df_metric["Prediction"]) / np.mean(df_metric["Actual"])
    df_result.loc[idx_name, "nMAE(Mean)"] = round(nmae_mean, 4)
    nmae_minmax = mean_absolute_error(df_metric["Actual"], df_metric["Prediction"]) / (df_metric["Actual"].max() - df_metric["Actual"].min())
    df_result.loc[idx_name, "nMAE(Max-Min)"] = round(nmae_minmax, 4)
    # MAPE
    mape = (mean_absolute_percentage_error(df_metric["Actual"], df_metric["Prediction"])* 100)
    df_result.loc[idx_name, "MAPE"] = round(mape, 2)
    # sMAPE
    absolute_errors = np.abs(df_metric["Actual"] - df_metric["Prediction"])
    sum_values = np.abs(df_metric["Actual"] + df_metric["Prediction"])
    smape = 100 * np.mean(2 * absolute_errors / sum_values)
    df_result.loc[idx_name, "sMAPE"] = round(smape, 2)
    # R-squared
    r_squared = r2_score(df_metric["Actual"], df_metric["Prediction"])
    df_result.loc[idx_name, "R2"] = round(r_squared, 2)
    return df_result


def optimizing_parameters(
    target_name,
    data_bundle,
    df_processed,
    y_test_updated,
    lag_number,
    test_length,
    total_iter,
    study_trials,
):
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_bundle

    # DF for Result
    df_best_params = pd.DataFrame()
    df_metric_valid_result = pd.DataFrame()
    df_metric_test_result = pd.DataFrame()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 200, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 0, 1),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 10),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
            "random_state": trial.suggest_int("random_state", 42, 42),
            "verbose": -1,
        }
        # Model
        model_optuna = LGBMRegressor(**params)
        model_optuna.fit(x_train, y_train)
        # Prediction
        pred_optuna = model_optuna.predict(x_valid)
        # Loss Function
        mse = mean_squared_error(y_valid, pred_optuna)
        r2 = r2_score(y_valid, pred_optuna)
        return r2 / mse
    
    for iter_idx in tqdm(range(0, total_iter)):
        # Optuna Tuning
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=study_trials)
        # Best Parameter
        best_params = study.best_params
        df_best_params = pd.concat([df_best_params, pd.DataFrame(best_params, index=[iter_idx])])

        # Model w/ Optuna
        model = LGBMRegressor(**best_params, verbose=-1)
        model.fit(x_train, y_train)

        # Prediction - ValidSet
        df_pred_valid = prediction(model, x_valid)
        df_pred_actual_valid = pred_actual(y_valid, df_pred_valid, target_name)
        df_metric_valid = metric(df_pred_actual_valid, target_name)
        df_metric_valid_result = pd.concat([df_metric_valid_result, df_metric_valid])

        # Prediction(Recursive) - Test
        for iter_idx_pred in range(0, test_length):
            what_to_predict = pd.DataFrame(x_test.iloc[iter_idx_pred,]).T
            # Prediction - TestSet
            df_pred_test = prediction(model, what_to_predict)
            y_test_updated.iloc[iter_idx_pred] = df_pred_test  # y Update
            # Lag Making
            df_tmp = pd.concat([y_train, y_valid])
            df_tmp = pd.concat([df_processed["dt"], df_tmp], axis=1)
            df_tmp = df_tmp.iloc[-(test_length * 2) :]
            df_tmp.update(y_test_updated)
            df_tmp = making_lag_features(lag_number, df_tmp, target_name)
            # Lag Update
            x_test.update(df_tmp)
        y_test_updated = (y_test_updated.rename(columns={target_name: "Prediction"}).reset_index().drop(["index"], axis=1))
        df_pred_actual_test = pred_actual(y_test, y_test_updated, target_name)
        df_metric_test = metric(df_pred_actual_test, target_name)
        df_metric_test_result = pd.concat([df_metric_test_result, df_metric_test])

    # Result
    df_metric_valid_result = df_metric_valid_result.reset_index().drop(["index"], axis=1)
    df_metric_test_result = df_metric_test_result.reset_index().drop(["index"], axis=1)

    # Sorting - TestSet
    metric_test_set = df_metric_test_result.sort_values(by="nRMSE(Max-Min)").reset_index()
    # Sorting - ValidSet
    index_order = metric_test_set["index"].values.tolist()
    metric_valid_set = df_metric_valid_result.reset_index()
    metric_valid_set = (metric_valid_set.set_index("index").reindex(index_order).reset_index())
    # Sorting - Parameters
    df_best_params = df_best_params.reset_index()
    df_best_params = (df_best_params.set_index("index").reindex(index_order).reset_index())
    return metric_valid_set, metric_test_set, df_best_params


def save_result(
    target_name,
    result_bundle,
    output_memo,
):
    metric_valid_set, metric_test_set, df_best_params = result_bundle

    # Saving Path
    save_path = f"./output/{target_name}/Metric_Parameters"
    if not os.path.exists(save_path):  # 폴더가 없는 경우 생성
        os.makedirs(save_path)
    # Date Saved
    today = datetime.now().date()
    saving_date = today.strftime("%y%m%d")
    # Time Saved
    current_time = datetime.now().time()
    saving_time = current_time.strftime("%H%M%S")

    # Saving - Best Parameters
    df_best_params.to_excel(
        save_path + f"/Parameters_{saving_date}_{saving_time}_{output_memo}.xlsx",
        sheet_name="Sheet1",
        index=True,
    )
    # Saving - Metric(Valid Set)
    metric_valid_set.to_excel(
        save_path + f"/Metric_ValidSet_{saving_date}_{saving_time}_{output_memo}.xlsx",
        sheet_name="Sheet1",
        index=True,
    )
    # Saving - Metric(Test Set)
    metric_test_set.to_excel(
        save_path + f"/Metric_TestSet_{saving_date}_{saving_time}_{output_memo}.xlsx",
        sheet_name="Sheet1",
        index=True,
    )
    print(f"{saving_date}_{saving_time}_result_saved")
