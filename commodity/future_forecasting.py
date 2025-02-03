import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

from preprocessing import pre_processing, data_split, making_lag_features
from optimization import predicting


def loading_best_params(target, model_name, file_version):
    path = f"./output/{target}/{model_name}/Metric_Parameters/"
    # Parameter
    df_best_params = pd.read_excel(path + "Parameters_" + file_version + ".xlsx")
    df_best_params = df_best_params.drop(["Unnamed: 0"], axis=1)
    # Metric - Valid
    metric_valid_set = pd.read_excel(path + "Metric_ValidSet_" + file_version + ".xlsx")
    metric_valid_set = metric_valid_set.drop(["Unnamed: 0"], axis=1)
    # Metric - Test
    metric_test_set = pd.read_excel(path + "Metric_TestSet_" + file_version + ".xlsx")
    metric_test_set = metric_test_set.drop(["Unnamed: 0"], axis=1)

    result_packing = metric_valid_set, metric_test_set, df_best_params
    return result_packing


def converting_params(model_name, df_input, best_index):
    # Recall
    recall_best_params = df_input.loc[best_index].to_dict()
    # Type Conversion
    if model_name == "LGBM":
        recall_best_params["n_estimators"] = int(recall_best_params["n_estimators"])
        recall_best_params["max_depth"] = int(recall_best_params["max_depth"])
        recall_best_params["min_child_samples"] = int(recall_best_params["min_child_samples"])
        recall_best_params["num_leaves"] = int(recall_best_params["num_leaves"])
        recall_best_params["subsample_freq"] = int(recall_best_params["subsample_freq"])
        recall_best_params["random_state"] = int(recall_best_params["random_state"])
    elif model_name == "XGB":
        recall_best_params["n_estimators"] = int(recall_best_params["n_estimators"])
        recall_best_params["max_depth"] = int(recall_best_params["max_depth"])
        recall_best_params["random_state"] = int(recall_best_params["random_state"])
    elif model_name == "Qboost":
        recall_best_params["n_estimators"] = int(recall_best_params["n_estimators"])
        recall_best_params["max_depth"] = int(recall_best_params["max_depth"])
        recall_best_params["min_samples_split"] = int(recall_best_params["min_samples_split"])
        recall_best_params["min_samples_leaf"] = int(recall_best_params["min_samples_leaf"])
        recall_best_params["random_state"] = int(recall_best_params["random_state"])
    recall_best_params.pop("index", None)
    return recall_best_params


def recalling_best(
    df_input,
    model_name,
    target,
    target_list,
    result_prediction,
    valid_set_length,
    test_set_length,
):
    metric_valid_set, metric_test_set, df_best_params, best_index = result_prediction

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
    recall_best_params = converting_params(model_name, df_best_params, best_index)

    # Model w/ Best Parameter
    if model_name == "LGBM":
        model = LGBMRegressor(**recall_best_params, verbose=-1)
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


def forecasting_future(
    df_input,
    model_name,
    target,
    target_list,
    result_prediction,
    valid_set_length,
    test_set_length,
):
    print("Future Forecasting - Start")
    metric_valid_set, metric_test_set, df_best_params, best_index = result_prediction

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
    recall_best_params = converting_params(model_name, df_best_params, best_index)

    # Model w/ Best Parameter
    if model_name == "LGBM":
        model = LGBMRegressor(**recall_best_params, verbose=-1)
        model.fit(x_train, y_train)
    elif model_name == "XGB":
        model = XGBRegressor(**recall_best_params)
        model.fit(x_train, y_train)
    elif model_name == "Qboost":
        model = GradientBoostingRegressor(**recall_best_params)
        model.fit(x_train, y_train)

    # Prediction(Recursive)
    for iter_idx_pred in range(0, test_set_length):
        what_to_predict = pd.DataFrame(df_future_x.iloc[iter_idx_pred,]).T
        # Prediction - future
        df_pred_future = predicting(model, what_to_predict)
        df_future_y.iloc[iter_idx_pred] = df_pred_future  # y Update
        # Lag Making
        df_tmp = y_test
        df_tmp = pd.concat([df_tmp, df_future_y], axis=0)
        df_tmp = making_lag_features(df_tmp, target)
        # Lag Update
        df_future_x.update(df_tmp)
    df_future_y = pd.DataFrame(df_future_y)
    df_future_y = df_future_y.rename(columns={target: "Prediction"})
    print("Future Forecasting - End")
    return df_future_y
