import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample
import shap
import os

from preprocessing import pre_processing, data_split, making_lag_features
from optimization import predicting


def converting_params(model_name, df_input):
    # Recall
    recall_best_params = df_input.loc[0].to_dict()

    # Type Conversion
    if model_name == "LGBM":
        recall_best_params["n_estimators"] = int(recall_best_params["n_estimators"])
        recall_best_params["max_depth"] = int(recall_best_params["max_depth"])
        recall_best_params["min_child_samples"] = int(
            recall_best_params["min_child_samples"]
        )
        recall_best_params["num_leaves"] = int(recall_best_params["num_leaves"])
        recall_best_params["subsample_freq"] = int(recall_best_params["subsample_freq"])
        recall_best_params["verbose"] = int(recall_best_params["verbose"])
        recall_best_params["random_state"] = int(recall_best_params["random_state"])
    elif model_name == "XGB":
        recall_best_params["n_estimators"] = int(recall_best_params["n_estimators"])
        recall_best_params["max_depth"] = int(recall_best_params["max_depth"])
        recall_best_params["random_state"] = int(recall_best_params["random_state"])
    elif model_name == "Qboost":
        recall_best_params["n_estimators"] = int(recall_best_params["n_estimators"])
        recall_best_params["max_depth"] = int(recall_best_params["max_depth"])
        recall_best_params["min_samples_split"] = int(
            recall_best_params["min_samples_split"]
        )
        recall_best_params["min_samples_leaf"] = int(
            recall_best_params["min_samples_leaf"]
        )
        recall_best_params["random_state"] = int(recall_best_params["random_state"])
    recall_best_params.pop("index", None)

    return recall_best_params


def recalling_best(
    df_input,
    model_name,
    target,
    target_list,
    df_best_params,
    valid_set_length,
    test_set_length,
):
    # Pre-processing
    df_input["dt"] = pd.to_datetime(df_input["dt"])
    df_processed, df_lag_result = pre_processing(df_input, target, target_list)
    df_for_train = df_processed.iloc[:-test_set_length]
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(
        df_for_train, target, valid_set_length, test_set_length
    )

    # Recursive Setting
    y_test_updated = y_test.copy()
    y_test_updated[target] = np.nan  # y Reset
    x_test[[col for col in df_lag_result.columns if col in x_test.columns]] = np.nan
    df_tmp = pd.concat([y_train, y_valid])
    df_tmp = pd.concat([df_for_train["dt"], df_tmp], axis=1)
    df_tmp = df_tmp.iloc[-(test_set_length * 2) :]
    df_tmp = making_lag_features(df_tmp, target)  # Lag Making
    x_test.update(df_tmp)  # Lag Update

    # Recall Best Parameter
    recall_best_params = converting_params(model_name, df_best_params)

    # Model w/ Best Parameter
    if model_name == "LGBM":
        model = LGBMRegressor(**recall_best_params)
        model.fit(x_train, y_train)
    elif model_name == "XGB":
        model = XGBRegressor(**recall_best_params)
        model.fit(x_train, y_train)
    elif model_name == "Qboost":
        model = GradientBoostingRegressor(**recall_best_params)
        model.fit(x_train, y_train)

    # Prediction(Recursive) - Test
    for iter_idx_pred in range(0, test_set_length):
        what_to_predict = pd.DataFrame(x_test.iloc[iter_idx_pred,]).T

        # Prediction - TestSet
        df_pred_test = predicting(model, what_to_predict)
        y_test_updated.iloc[iter_idx_pred] = df_pred_test

        # Lag Making
        df_tmp = pd.concat([y_train, y_valid])
        df_tmp = pd.concat([df_for_train["dt"], df_tmp], axis=1)
        df_tmp = df_tmp.iloc[-(test_set_length * 2) :]
        df_tmp.update(y_test_updated)
        df_tmp = making_lag_features(df_tmp, target)

        # Lag Update
        x_test.update(df_tmp)

    df_pred_test = pd.DataFrame(y_test_updated)
    df_pred_test = df_pred_test.rename(columns={target: "Prediction"})

    return df_pred_test, y_test


def _making_shap_values(df_shap_result, df_input, test_set_length, target, target_list):
    df_shap_result.reset_index(drop=True, inplace=True)
    df_shap_result.rename(
        columns={
            "Year": "dt_year",
            "Month": "dt_month",
            "WeekByYear": "dt_week_num",
            "WeekByMonth": "dt_week_month",
            # Lag
            "lag_1": "v_lag_1",
            "lag_2": "v_lag_2",
            "lag_3": "v_lag_3",
            "lag_4": "v_lag_4",
            "lag_5": "v_lag_5",
            "lag_6": "v_lag_6",
            "delta_lag1_lag2": "v_chgrate_1_2",
            "delta_lag2_lag3": "v_chgrate_2_3",
            "delta_lag3_lag4": "v_chgrate_3_4",
            "delta_lag4_lag5": "v_chgrate_4_5",
            "delta_lag5_lag6": "v_chgrate_5_6",
            "mean_window_3": "v_wndavg_3",
            "mean_window_6": "v_wndavg_6",
            "std_window_3": "v_wndstd_3",
            "std_window_6": "v_wndstd_6",
            # EX
            "USD_CNY": "EX_USD_CNY",
            "USD_KRW": "EX_USD_KRW",
            "USD_JPY": "EX_USD_JPY",
            "USD_AUD": "EX_AUD_USD",
            "USD_BRL": "EX_USD_BRL",
            "USD_INR": "EX_INR_USD",
            "USD_DXY": "Idx_DxyUSD",
            "Stocks_US500": "Idx_SnP500",
            "Stocks_USVIX": "Idx_SnPVIX",
            "Stocks_CH50": "Idx_CH50",
            "Stocks_CSI300": "Idx_CSI300",
            "Stocks_SHANGHAI50": "Idx_Shanghai50",
            "Stocks_SHANGHAI": "Idx_Shanghai",
            "Stocks_HK50": "Idx_HangSeng",
            "Stocks_GEI": "Idx_SnPGlobal1200",
        },
        inplace=True,
    )

    df_shap_result_col_name = df_shap_result.columns
    df_shap_result.reset_index(inplace=True)
    df_shap_result.rename(columns={"index": "h"}, inplace=True)
    snapshot_dt = df_input[["dt"]].iloc[-test_set_length]
    snapshot_dt = pd.to_datetime(snapshot_dt[0])

    df_shap_result_deepflow = pd.DataFrame()

    for col_idx in range(len(df_shap_result_col_name)):
        df_shap_result_tmp = df_shap_result[["h", df_shap_result_col_name[col_idx]]]
        df_shap_result_tmp.rename(
            columns={df_shap_result_col_name[col_idx]: "shap_value"}, inplace=True
        )
        df_shap_result_tmp["col_name"] = df_shap_result_col_name[col_idx]
        df_shap_result_tmp["snapshot_dt"] = snapshot_dt
        df_shap_result_tmp["grain_id"] = target_list[target]
        df_shap_result_tmp = df_shap_result_tmp[
            ["snapshot_dt", "grain_id", "h", "col_name", "shap_value"]
        ]
        df_shap_result_deepflow = pd.concat(
            [df_shap_result_deepflow, df_shap_result_tmp]
        )

    # Saving Path
    save_path = f"./output/{target}/_Result/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # SHAP Saving
    df_shap_result_deepflow.to_csv(
        save_path + "shap_result.csv",
        index=False,
    )

    return df_shap_result_deepflow


def forecasting_future(
    df_input,
    model_name,
    target,
    target_list,
    df_best_params,
    valid_set_length,
    test_set_length,
):
    print("Future Forecasting - Start")

    # Pre-processing
    df_input["dt"] = pd.to_datetime(df_input["dt"])
    df_processed, df_lag_result = pre_processing(df_input, target, target_list)
    df_for_train = df_processed.iloc[:-test_set_length]
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(
        df_for_train, target, valid_set_length, test_set_length
    )
    df_for_future = df_processed.iloc[-test_set_length:]
    df_future_x = df_for_future.drop([target], axis=1)
    df_future_x = df_future_x.drop(["dt"], axis=1)
    df_future_y = df_for_future[target].to_frame()

    # Recall Best Parameter
    recall_best_params = converting_params(model_name, df_best_params)

    # Model w/ Best Parameter
    if model_name == "LGBM":
        model = LGBMRegressor(**recall_best_params)
        model.fit(x_train, y_train)
    elif model_name == "XGB":
        model = XGBRegressor(**recall_best_params)
        model.fit(x_train, y_train)
    elif model_name == "Qboost":
        model = GradientBoostingRegressor(**recall_best_params)
        model.fit(x_train, y_train)

    # SHAP
    explainer = shap.TreeExplainer(model)
    x_col_name = df_future_x.columns
    df_shap_result = pd.DataFrame(columns=x_col_name)

    # Bootstraps
    df_ci_result = pd.DataFrame()
    n_bootstraps = 100  # 100  # here

    # Prediction(Recursive)
    for iter_idx_pred in range(0, test_set_length):
        what_to_predict = pd.DataFrame(df_future_x.iloc[iter_idx_pred,]).T
        bootstrap_preds = []

        # SHAP
        shap_values = explainer.shap_values(what_to_predict)
        shap_values_update = pd.DataFrame(shap_values, columns=x_col_name)
        df_shap_result = pd.concat([df_shap_result, shap_values_update])

        # Prediction - future
        df_pred_future = predicting(model, what_to_predict)
        df_future_y.iloc[iter_idx_pred] = df_pred_future  # y Update

        # Lag Making
        df_tmp = y_test
        df_tmp = pd.concat([df_tmp, df_future_y], axis=0)
        df_tmp = making_lag_features(df_tmp, target)

        # Lag Update
        df_future_x.update(df_tmp)

        # Confidence Interval
        for _ in range(n_bootstraps):
            # Bootstrap
            x_resampled, y_resampled = resample(x_train, y_train)
            model.fit(x_resampled, y_resampled)
            bootstrap_preds.append(model.predict(what_to_predict)[0])

        bootstrap_preds = np.array(bootstrap_preds)
        lower_bound = np.percentile(bootstrap_preds, 2.5, axis=0)
        upper_bound = np.percentile(bootstrap_preds, 97.5, axis=0)
        df_confidence_interval = pd.DataFrame(
            {
                "Lower_CI": [lower_bound],
                "Upper_CI": [upper_bound],
            }
        )
        df_ci_result = pd.concat([df_ci_result, df_confidence_interval])
        df_ci_result["Lower_CI"] = round(df_ci_result["Lower_CI"], 3)
        df_ci_result["Upper_CI"] = round(df_ci_result["Upper_CI"], 3)
        df_ci_result = df_ci_result.reset_index(drop=True)

    df_future_y = pd.DataFrame(df_future_y)
    df_future_y = df_future_y.rename(columns={target: "Prediction"})

    # SHAP
    _making_shap_values(df_shap_result, df_input, test_set_length, target, target_list)
    print("SHAP Result Saved")

    print("Future Forecasting - End")

    return df_future_y, df_ci_result
