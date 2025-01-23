import pandas as pd
import numpy as np
import optuna
import warnings

from run_cli import PredictionParameter
from extra import (
    pre_processing,
    data_split,
    making_lag_features,
    optimizing_parameters,
    save_result,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)


target = "IronOre"  # fix this
valid_set_length = 3
test_set_length = 7


def train_and_predict(
    param: PredictionParameter, target_name, valid_length, test_length
):
    # TODO impl
    ref_date = param.ref_date
    input_path = param.input_path
    ext_path_list = param.ext_path_list
    output_path = param.output_path
    print(ref_date, input_path, ext_path_list, output_path)

    # Load Data
    df = pd.read_parquet(input_path)

    # Pre-processing
    lag_number = 6
    df_processed, df_lag_result = pre_processing(df, target_name, lag_number)
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(
        df_processed, target_name, valid_length, test_length
    )

    # Recursive Setting
    y_test_updated = y_test.copy()
    y_test_updated[target_name] = np.nan  # y Reset
    x_test[[col for col in df_lag_result.columns if col in x_test.columns]] = np.nan  # Lag Reset
    df_tmp = pd.concat([y_train, y_valid])
    df_tmp = pd.concat([df_processed["dt"], df_tmp], axis=1)
    df_tmp = df_tmp.iloc[-(test_length * 2) :]
    df_tmp = making_lag_features(lag_number, df_tmp, target_name)  # Lag Making
    x_test.update(df_tmp)  # Lag Update
    del df_tmp

    # HyperParameters Optimization
    data_bundle = x_train, y_train, x_valid, y_valid, x_test, y_test
    metric_valid_set, metric_test_set, df_best_params = optimizing_parameters(
        target_name,
        data_bundle,
        df_processed,
        y_test_updated,
        lag_number,
        test_length,
        300,
        50,
    )
    result_bundle = metric_valid_set, metric_test_set, df_best_params
    return result_bundle


def main():
    from run_cli import run_cli
    param = run_cli()
    result = train_and_predict(param, target, valid_set_length, test_set_length)
    save_result(target, result, "")


if __name__ == "__main__":
    main()