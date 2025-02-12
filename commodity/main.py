import pandas as pd
import numpy as np
import warnings

from run_cli import PredictionParameter
from preprocessing import pre_processing, data_split, making_lag_features
from features_forecasting import forecasting_features
from optimization import optimizing_parameters, choosing_best_model
from future_forecasting import recalling_best, forecasting_future
from result import (
    saving_essential,
    saving_result,
    saving_plot_n_result,
    merging_results,
    clearing_all_files,
)

warnings.filterwarnings("ignore")

# ----------------- Model Parameters -----------------
forecasting_units = "Monthly"  # ("Monthly", "Weekly")
data_setting = "Fixed"  # ("Fixed", "Manually")
result_saving_for_tracking = "False"  # Default: "False"
# ----------------------------------------------------

model_list = ["LGBM", "XGB", "Qboost"]
target_dict = {
    "IronOre": "SCO:COM",
    "Nickel": "LN1:COM",
    "Coal": "XAL1:COM",
    "CokingCoal": "_DJMc1:COM",
    "Steel": "JBP:COM",
    "Copper": "HG1:COM",
    "Aluminum": "LMAHDS03:COM",
}

if data_setting == "Fixed":
    if forecasting_units == "Monthly":
        valid_set_length = 3
        test_set_length = 7
        future_length = 7
    else:
        valid_set_length = 4
        test_set_length = 4
        future_length = 8
else:
    valid_set_length = 3  # fill this as you want
    test_set_length = 7  # fill this as you want
    future_length = 7  # fill this as you want


def loading_data(param: PredictionParameter):
    ref_date = param.ref_date
    input_path = param.input_path
    ext_path_list = param.ext_path_list
    output_path = param.output_path
    print(ref_date, input_path, ext_path_list, output_path)

    # Load Data
    df = pd.read_parquet(input_path)

    # 추후 삭제 - from here
    df["Date"] = (
        pd.to_datetime(df["Date"]) - pd.offsets.MonthEnd() + pd.offsets.MonthBegin(1)
    )
    df = df.rename(columns={"Date": "dt"})
    # 추후 삭제 - to here

    return df


def train_and_predict(df, target_name, model_name):
    # Pre-processing
    df_processed, df_lag_result = pre_processing(df, target_name, target_dict)
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(
        df_processed, target_name, valid_set_length, test_set_length
    )

    # Recursive Setting
    y_test_updated = y_test.copy()
    y_test_updated[target_name] = np.nan  # y Reset
    x_test[[col for col in df_lag_result.columns if col in x_test.columns]] = (
        np.nan
    )  # Lag Reset
    df_tmp = pd.concat([y_train, y_valid])
    df_tmp = pd.concat([df_processed["dt"], df_tmp], axis=1)
    df_tmp = df_tmp.iloc[-(test_set_length * 2) :]
    df_tmp = making_lag_features(df_tmp, target_name)  # Lag Making
    x_test.update(df_tmp)  # Lag Update

    # HyperParameters Optimization
    print(f"Hyperparameter Optimization - Start ({model_name})")
    data_packing = x_train, y_train, x_valid, y_valid, x_test, y_test
    metric_valid_set, metric_test_set, df_best_params = optimizing_parameters(
        model_name,
        target_name,
        data_packing,
        df_processed,
        y_test_updated,
        test_set_length,
        300,  # 300
        50,  # 50
    )
    return metric_valid_set, metric_test_set, df_best_params


def main():
    from run_cli import run_cli

    # Data Load
    param = run_cli()
    df = loading_data(param)

    # Feature Forecasting
    df_expanded = forecasting_features(
        df, target_dict, valid_set_length, test_set_length, future_length
    )

    # Forecasting
    for target in target_dict:
        print(f"-----------------{target}-----------------")

        for model_name in model_list:
            # Optimization
            metric_valid_set, metric_test_set, df_best_params = train_and_predict(
                df, target, model_name
            )
            # Result
            if result_saving_for_tracking == "False":
                saving_essential(
                    target,
                    model_name,
                    metric_test_set,
                    df_best_params,
                )
            else:
                saving_result(
                    target,
                    model_name,
                    metric_valid_set,
                    metric_test_set,
                    df_best_params,
                    "",
                )

        # Best Model & Params
        best_model, best_index, df_best_params = choosing_best_model(target, model_list)

        # Recall Best Model
        df_pred_test, df_actual_test = recalling_best(
            df_expanded,
            best_model,
            target,
            target_dict,
            df_best_params,
            valid_set_length,
            test_set_length,
        )

        # Future Forecasting
        df_pred_future, df_ci_result = forecasting_future(
            df_expanded,
            best_model,
            target,
            target_dict,
            df_best_params,
            valid_set_length,
            test_set_length,
        )

        # Result Saving
        saving_plot_n_result(
            df,
            target,
            best_model,
            df_pred_test,
            df_pred_future,
            df_ci_result,
            df_expanded,
            result_saving_for_tracking,
        )

    # Result Merge
    merging_results(target_dict)
    clearing_all_files(result_saving_for_tracking)

    print("All Process - Done")

    return


if __name__ == "__main__":
    main()
