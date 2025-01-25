import pandas as pd
from lightgbm import LGBMRegressor
from tqdm import tqdm
import optuna
from sklearn.metrics import mean_squared_error, r2_score

from preprocessing import making_lag_features
from result import calculating_metric

optuna.logging.set_verbosity(optuna.logging.ERROR)


def _predicting(model, df_input):
    pred = model.predict(df_input)
    df_pred = pd.DataFrame(pred)
    df_pred[df_pred < 0] = 0
    df_pred = df_pred.rename(columns={0: "Prediction"})
    return df_pred


def _merging_pred_actual(df_actual, df_pred, target_name):
    df_actual = df_actual.rename(columns={target_name: "Actual"})
    df_actual = df_actual.reset_index().drop(["index"], axis=1)
    df_output = pd.concat([df_actual, df_pred], axis=1)
    return df_output


def optimizing_parameters(fixed_parameters_packing, total_iter, study_trials):
    target_name, data_packing, df_processed, y_test_updated, test_length = (fixed_parameters_packing)
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_packing

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
        df_pred_valid = _predicting(model, x_valid)
        df_pred_actual_valid = _merging_pred_actual(y_valid, df_pred_valid, target_name)
        df_metric_valid = calculating_metric(df_pred_actual_valid, target_name)
        df_metric_valid_result = pd.concat([df_metric_valid_result, df_metric_valid])

        # Prediction(Recursive) - Test
        for iter_idx_pred in range(0, test_length):
            what_to_predict = pd.DataFrame(x_test.iloc[iter_idx_pred,]).T
            # Prediction - TestSet
            df_pred_test = _predicting(model, what_to_predict)
            y_test_updated.iloc[iter_idx_pred] = df_pred_test  # y Update
            # Lag Making
            df_tmp = pd.concat([y_train, y_valid])
            df_tmp = pd.concat([df_processed["dt"], df_tmp], axis=1)
            df_tmp = df_tmp.iloc[-(test_length * 2) :]
            df_tmp.update(y_test_updated)
            df_tmp = making_lag_features(df_tmp, target_name)
            # Lag Update
            x_test.update(df_tmp)
        y_test_updated = (y_test_updated.rename(columns={target_name: "Prediction"}).reset_index().drop(["index"], axis=1))
        df_pred_actual_test = _merging_pred_actual(y_test, y_test_updated, target_name)
        df_metric_test = calculating_metric(df_pred_actual_test, target_name)
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
