import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm
import os
import glob
import optuna
from sklearn.metrics import mean_squared_error, r2_score

from preprocessing import making_lag_features
from result import calculating_metric

optuna.logging.set_verbosity(optuna.logging.ERROR)


def predicting(model, df_input):
    pred = model.predict(df_input)
    df_pred = pd.DataFrame(pred)
    df_pred[df_pred < 0] = 0
    df_pred = df_pred.rename(columns={0: "Prediction"})

    return df_pred


def merging_pred_actual(df_actual, df_pred, target_name):
    df_actual = df_actual.rename(columns={target_name: "Actual"})
    df_actual = df_actual.reset_index(drop=True)
    df_output = pd.concat([df_actual, df_pred], axis=1)

    return df_output


def optimizing_parameters(
    model_name,
    target_name,
    data_packing,
    df_processed,
    y_test_updated,
    test_length,
    total_iter,
    study_trials,
):
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_packing

    # Preserving
    y_test_updated_saving = y_test_updated.copy()
    x_test_saving = x_test.copy()

    if model_name == "LGBM":

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
                # "objective": "quantile",  # (Default: "mse" / Optional: "mae", "quantile")
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

    elif model_name == "XGB":

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 1, 200, step=10),
                "learning_rate": trial.suggest_float("learning_rate", 0, 1),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
                "gamma": trial.suggest_float("gamma", 0, 1),
                "random_state": trial.suggest_int("random_state", 42, 42),
                # "objective": "reg:quantileerror",  # (Default: "reg:squarederror" / Optional: "reg:absoluteerror", "reg:quantileerror")
                # "quantile_alpha": trial.suggest_float("quantile_alpha", 0.05, 0.95),  # "quantileerror"일 때, 활성화
            }
            # Model
            model_optuna = XGBRegressor(**params)
            model_optuna.fit(x_train, y_train)

            # Prediction
            pred_optuna = model_optuna.predict(x_valid)

            # Loss Function
            mse = mean_squared_error(y_valid, pred_optuna)
            r2 = r2_score(y_valid, pred_optuna)

            return r2 / mse

    elif model_name == "Qboost":

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 1, 200, step=10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 1),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": trial.suggest_int("random_state", 42, 42),
                "loss": "quantile",  # (Default: "quantile" / Optional: "squared_error", "absolute_error")
            }
            # Model
            model_optuna = GradientBoostingRegressor(**params)
            model_optuna.fit(x_train, y_train)

            # Prediction
            pred_optuna = model_optuna.predict(x_valid)

            # Loss Function
            mse = mean_squared_error(y_valid, pred_optuna)
            r2 = r2_score(y_valid, pred_optuna)

            return r2 / mse

    # Result DF
    df_best_params = pd.DataFrame()
    df_metric_valid_result = pd.DataFrame()
    df_metric_test_result = pd.DataFrame()

    for iter_idx in tqdm(range(0, total_iter)):
        # Reset
        x_test = x_test_saving.copy()
        y_test_updated = y_test_updated_saving.copy()

        # Optuna Tuning
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=study_trials)

        # Best Parameter
        best_params = study.best_params
        df_best_params = pd.concat(
            [df_best_params, pd.DataFrame(best_params, index=[iter_idx])]
        )

        # Model w/ Optuna
        if model_name == "LGBM":
            model = LGBMRegressor(**best_params, verbose=-1)
            model.fit(x_train, y_train)
        elif model_name == "XGB":
            model = XGBRegressor(**best_params)
            model.fit(x_train, y_train)
        elif model_name == "Qboost":
            model = GradientBoostingRegressor(**best_params)
            model.fit(x_train, y_train)

        # Prediction - ValidSet
        df_pred_valid = predicting(model, x_valid)
        df_pred_actual_valid = merging_pred_actual(y_valid, df_pred_valid, target_name)
        df_metric_valid = calculating_metric(df_pred_actual_valid, target_name)
        df_metric_valid_result = pd.concat([df_metric_valid_result, df_metric_valid])

        # Prediction(Recursive) - Test
        for iter_idx_pred in range(0, test_length):
            what_to_predict = pd.DataFrame(x_test.iloc[iter_idx_pred,]).T

            # Prediction - TestSet
            df_pred_test = predicting(model, what_to_predict)
            y_test_updated.iloc[iter_idx_pred] = df_pred_test  # y Update

            # Lag Making
            df_tmp = pd.concat([y_train, y_valid])
            df_tmp = pd.concat([df_processed["dt"], df_tmp], axis=1)
            df_tmp = df_tmp.iloc[-(test_length * 2) :]
            df_tmp.update(y_test_updated)
            df_tmp = making_lag_features(df_tmp, target_name)

            # Lag Update
            x_test.update(df_tmp)

        y_test_updated = y_test_updated.rename(columns={target_name: "Prediction"})
        y_test_updated = y_test_updated.reset_index(drop=True)
        df_pred_actual_test = merging_pred_actual(y_test, y_test_updated, target_name)
        df_metric_test = calculating_metric(df_pred_actual_test, target_name)
        df_metric_test_result = pd.concat([df_metric_test_result, df_metric_test])

    # Result
    df_metric_valid_result = df_metric_valid_result.reset_index(drop=True)
    df_metric_test_result = df_metric_test_result.reset_index(drop=True)

    # Sorting - TestSet
    metric_test_set = df_metric_test_result.sort_values(by="nRMSE(Max-Min)")
    metric_test_set = metric_test_set.reset_index()

    # Sorting - ValidSet
    index_order = metric_test_set["index"].values.tolist()
    metric_valid_set = df_metric_valid_result.reset_index()
    metric_valid_set = metric_valid_set.set_index("index")
    metric_valid_set = metric_valid_set.reindex(index_order)
    metric_valid_set = metric_valid_set.reset_index()

    # Sorting - Parameters
    df_best_params = df_best_params.reset_index()
    df_best_params = df_best_params.set_index("index")
    df_best_params = df_best_params.reindex(index_order)
    df_best_params = df_best_params.reset_index()

    return metric_valid_set, metric_test_set, df_best_params


def choosing_best_model(target, model_list):
    df_best_metric = pd.DataFrame()

    for model_name in model_list:
        # Metric Load
        path = f"./output/{target}/{model_name}/Metric_Parameters/"
        metric_file_list = glob.glob(os.path.join(path, "Metric_TestSet*.xlsx"))
        latest_file = max(metric_file_list)
        metric_test_set = pd.read_excel(latest_file)

        # Saving Best Model
        what_to_save = pd.DataFrame(metric_test_set.iloc[0]).T
        what_to_save["Model"] = model_name
        df_best_metric = pd.concat([df_best_metric, what_to_save])
        df_best_metric = df_best_metric.reset_index(drop=True)

    best_index = df_best_metric["nRMSE(Max-Min)"].idxmin()
    best_model = df_best_metric.iloc[best_index]["Model"]

    # Parameter Load
    path = f"./output/{target}/{best_model}/Metric_Parameters/"
    params_file_list = glob.glob(os.path.join(path, "Parameters*.xlsx"))
    latest_file = max(params_file_list)
    df_best_params = pd.read_excel(latest_file)
    df_best_params = df_best_params.drop(["Unnamed: 0"], axis=1)
    df_best_params = pd.DataFrame(df_best_params.iloc[0]).T

    return best_model, best_index, df_best_params
