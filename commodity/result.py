import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)


def calculating_metric(df_metric, idx_name):
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


def saving_result(
    target_name,
    result_packed,
    output_memo,
):
    metric_valid_set, metric_test_set, df_best_params = result_packed

    # Saving Path
    save_path = f"./output/{target_name}/Metric_Parameters"
    if not os.path.exists(save_path):
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
    print(f"Result Saved {saving_date}_{saving_time}")
