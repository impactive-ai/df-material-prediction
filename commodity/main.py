import pandas as pd
import numpy as np
import warnings

from run_cli import PredictionParameter
from preprocessing import pre_processing, data_split, making_lag_features
from features_forecasting import forecasting_features
from optimization import optimizing_parameters, merging_pred_actual
from result import saving_result, calculating_metric, saving_plot_n_result
from future_forecasting import loading_best_params, recalling_best, forecasting_future

warnings.filterwarnings("ignore")


target = "IronOre"  # fix this
target_list = [
    "IronOre",
    "Nickel",
    "Coal",
    "CokingCoal",
    "Steel",
    "Copper",
    "Aluminum",
]
valid_set_length = 3
test_set_length = 7


def loading_data(param: PredictionParameter):
    ref_date = param.ref_date
    input_path = param.input_path
    ext_path_list = param.ext_path_list
    output_path = param.output_path
    print(ref_date, input_path, ext_path_list, output_path)

    # Load Data
    df = pd.read_parquet(input_path)
    # 추후 삭제 - "date"를 월의 첫 날짜로 변경
    df["Date"] = (pd.to_datetime(df["Date"]) - pd.offsets.MonthEnd() + pd.offsets.MonthBegin(1))
    # 추후 삭제 - Rename
    df = df.rename(columns={"Date": "dt"})
    return df


def train_and_predict(df, target_name):
    # Pre-processing
    df_processed, df_lag_result = pre_processing(df, target_name, target_list)
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(
        df_processed, target_name, valid_set_length, test_set_length
    )

    # Recursive Setting
    y_test_updated = y_test.copy()
    y_test_updated[target_name] = np.nan  # y Reset
    x_test[[col for col in df_lag_result.columns if col in x_test.columns]] = (np.nan)  # Lag Reset
    df_tmp = pd.concat([y_train, y_valid])
    df_tmp = pd.concat([df_processed["dt"], df_tmp], axis=1)
    df_tmp = df_tmp.iloc[-(test_set_length * 2) :]
    df_tmp = making_lag_features(df_tmp, target_name)  # Lag Making
    x_test.update(df_tmp)  # Lag Update

    # HyperParameters Optimization
    print("Hyperparameter Optimization - Start")
    data_packing = x_train, y_train, x_valid, y_valid, x_test, y_test
    fixed_parameters_packing = (
        target_name,
        data_packing,
        df_processed,
        y_test_updated,
        test_set_length,
    )
    metric_valid_set, metric_test_set, df_best_params = optimizing_parameters(fixed_parameters_packing, 300, 50)
    result_packing = metric_valid_set, metric_test_set, df_best_params

    print("Hyperparameter Optimization - End")
    return result_packing


def main():
    from run_cli import run_cli

    # Data Load
    param = run_cli()
    df = loading_data(param)

    # Feature Forecasting
    df_expanded = forecasting_features(df, target_list)

    # Optimization
    optimization_flag = False  # fix this
    if optimization_flag == True:  # 최적화가 필요할 때
        result_prediction = train_and_predict(df, target)
        saving_result(target, result_prediction, "")
    elif optimization_flag == False:  # 최적화 결과가 이미 존재할 때
        print("Hyperparameter Optimization - Skip")
        result_prediction = loading_best_params(target, "250131_210436_")  # fix this
        print("Optimized Hyperparameter - Loaded")

    # Recall Best Model
    df_pred_test, df_actual_test = recalling_best(
        df_expanded,
        target,
        target_list,
        result_prediction,
        valid_set_length,
        test_set_length,
    )

    # Future Forecasting
    df_pred_future = forecasting_future(
        df_expanded,
        target,
        target_list,
        result_prediction,
        valid_set_length,
        test_set_length,
    )

    # Result Saving
    saving_plot_n_result(target, df_pred_test, df_pred_future, df_expanded)
    print("All Process - Done")


if __name__ == "__main__":
    main()
