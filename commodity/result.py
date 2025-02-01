import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
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


def saving_plot_n_result(target, df_pred_test, df_pred_future, df_expanded):
    plot_num = (len(df_pred_future) * 2) + 4
    df_plot = df_expanded[["dt", target]].iloc[-plot_num:]
    df_plot = df_plot.rename(columns={target: "Actual"})
    df_plot = df_plot.reset_index().drop(["index"], axis=1)
    # df_pred_test
    df_tmp = df_plot.iloc[-(len(df_pred_future) * 2) : -len(df_pred_future)]
    df_tmp = df_tmp.reset_index().drop(["index"], axis=1)
    df_pred_test = df_pred_test.reset_index().drop(["index"], axis=1)
    df_pred_test = pd.concat([df_tmp, df_pred_test], axis=1)
    df_pred_test["Target"] = target
    df_pred_test["Tag"] = "TestSet"
    # df_pred_future
    df_tmp = df_plot.iloc[-len(df_pred_future) :]
    df_tmp = df_tmp.reset_index().drop(["index"], axis=1)
    df_pred_future = df_pred_future.reset_index().drop(["index"], axis=1)
    df_pred_future = pd.concat([df_tmp, df_pred_future], axis=1)
    df_pred_future["Target"] = target
    df_pred_future["Tag"] = "FutureForecasting"
    # df_plot
    df_plot = df_plot.set_index("dt")
    df_plot["Prediction"] = np.nan
    df_plot["Target"] = target
    df_plot["Tag"] = ""
    df_pred_test = df_pred_test.set_index("dt")
    df_pred_future = df_pred_future.set_index("dt")
    df_plot.update(df_pred_test[["Prediction", "Target", "Tag"]])
    df_plot.update(df_pred_future[["Prediction", "Target", "Tag"]])
    df_plot = df_plot.reset_index()

    # Plot
    plt.figure(figsize=(16, 7))
    plt.plot(df_plot["dt"], df_plot["Actual"], label="Actual", marker="o")
    plt.plot(df_plot["dt"], df_plot["Prediction"], label="Prediction", marker="o")
    # Vertical Line - TestSet
    testset_starting_date = df_plot.iloc[-((len(df_pred_future) * 2))]["dt"]
    vertical_line_test = testset_starting_date.strftime("%Y-%m-%d")
    vertical_line_test_dt = datetime.strptime(vertical_line_test, "%Y-%m-%d")
    plt.axvline(x=date2num(vertical_line_test_dt), color="black", linestyle="--", alpha=0.5)
    plt.text(
        date2num(vertical_line_test_dt),
        df_plot["Actual"].min(),
        " TestSet",
        color="black",
        va="top",
        ha="left",
    )
    # Vertical Line - Future Forecasting
    forecasting_starting_date = df_plot.iloc[-(len(df_pred_future) + 1)]["dt"]
    vertical_line_forecasting = forecasting_starting_date.strftime("%Y-%m-%d")
    vertical_line_forecasting_dt = datetime.strptime(vertical_line_forecasting, "%Y-%m-%d")
    plt.axvline(x=date2num(vertical_line_forecasting_dt), color="red", linestyle="--", alpha=0.5)
    plt.text(
        date2num(vertical_line_forecasting_dt),
        df_plot["Actual"].min(),
        " FutureForecasting",
        color="red",
        va="top",
        ha="left",
    )
    # Label & Title
    plt.xlabel("dt")
    plt.ylabel("Price")
    plt.title(target)
    plt.legend()
    plt.grid(True)

    # Plot Saving
    save_path = f"./output/{target}/Result/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_file_name = "Plot_PredActual.png"
    saving_dir = save_path + plot_file_name
    plt.savefig(saving_dir)

    # DF Saving
    df_export = df_plot.iloc[-(len(df_pred_future) * 2) :]
    df_export = df_export.reset_index().drop(["index"], axis=1)
    df_export["Actual"] = round(df_export["Actual"], 3)
    df_export["Prediction"] = round(df_export["Prediction"], 3)
    df_export.to_excel(
        save_path + "Values_PredActual.xlsx",
        sheet_name="Sheet1",
        index=False,
    )
    print("All Results are saved")
    return df_export
