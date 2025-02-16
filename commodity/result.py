import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import os
import shutil
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
    nrmse_mean = np.sqrt(
        mean_squared_error(df_metric["Actual"], df_metric["Prediction"])
    ) / np.mean(df_metric["Actual"])
    df_result.loc[idx_name, "nRMSE(Mean)"] = round(nrmse_mean, 4)
    nrmse_minmax = np.sqrt(
        mean_squared_error(df_metric["Actual"], df_metric["Prediction"])
    ) / (df_metric["Actual"].max() - df_metric["Actual"].min())
    df_result.loc[idx_name, "nRMSE(Max-Min)"] = round(nrmse_minmax, 4)

    # MAE
    mae = mean_absolute_error(df_metric["Actual"], df_metric["Prediction"])
    df_result.loc[idx_name, "MAE"] = round(mae, 2)

    # nMAE
    nmae_mean = mean_absolute_error(
        df_metric["Actual"], df_metric["Prediction"]
    ) / np.mean(df_metric["Actual"])
    df_result.loc[idx_name, "nMAE(Mean)"] = round(nmae_mean, 4)
    nmae_minmax = mean_absolute_error(df_metric["Actual"], df_metric["Prediction"]) / (
        df_metric["Actual"].max() - df_metric["Actual"].min()
    )
    df_result.loc[idx_name, "nMAE(Max-Min)"] = round(nmae_minmax, 4)

    # MAPE
    mape = (
        mean_absolute_percentage_error(df_metric["Actual"], df_metric["Prediction"])
        * 100
    )
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


def saving_essential(
    target_name,
    model_name,
    metric_test_set,
    df_best_params,
):
    # Saving Path
    save_path = f"./output/{target_name}/{model_name}/Metric_Parameters"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Saving - Best Parameters
    df_best_params.to_excel(
        save_path + f"/Parameters.xlsx",
        sheet_name="Sheet1",
        index=True,
    )
    # Saving - Metric(Test Set)
    metric_test_set.to_excel(
        save_path + f"/Metric_TestSet.xlsx",
        sheet_name="Sheet1",
        index=True,
    )

    return


def saving_result(
    target_name,
    model_name,
    metric_valid_set,
    metric_test_set,
    df_best_params,
    output_memo,
):
    # Saving Path
    save_path = f"./output/{target_name}/{model_name}/Metric_Parameters"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    today = datetime.now().date()
    saving_date = today.strftime("%y%m%d")  # Date Saved
    current_time = datetime.now().time()
    saving_time = current_time.strftime("%H%M%S")  # Time Saved

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

    return


def saving_plot_n_result(
    df,
    target,
    model_name,
    df_pred_test,
    df_pred_future,
    df_ci_result,
    df_expanded,
    result_saving_for_tracking,
):
    plot_num = (len(df_pred_future) * 2) + 4
    df_plot = df_expanded[["dt", target]].iloc[-plot_num:]
    df_plot = df_plot.rename(columns={target: "Actual"})
    df_plot = df_plot.reset_index(drop=True)

    # df_pred_test
    df_tmp = df_plot.iloc[-(len(df_pred_future) * 2) : -len(df_pred_future)]
    df_tmp = df_tmp.reset_index(drop=True)
    df_pred_test = df_pred_test.reset_index(drop=True)
    df_pred_test = pd.concat([df_tmp, df_pred_test], axis=1)
    df_pred_test["Target"] = target
    df_pred_test["Tag"] = "TestSet"

    # df_pred_future
    df_tmp = df_plot.iloc[-len(df_pred_future) :]
    df_tmp = df_tmp.reset_index(drop=True)
    df_pred_future = df_pred_future.reset_index(drop=True)
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
    plt.axvline(
        x=date2num(vertical_line_test_dt), color="black", linestyle="--", alpha=0.5
    )
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
    vertical_line_forecasting_dt = datetime.strptime(
        vertical_line_forecasting, "%Y-%m-%d"
    )
    plt.axvline(
        x=date2num(vertical_line_forecasting_dt), color="red", linestyle="--", alpha=0.5
    )
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

    # Saving Path
    save_path = f"./output/{target}/_Result/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Plot Saving
    if not result_saving_for_tracking == "False":
        plot_file_name = "Plot_PredActual.png"
        saving_dir = save_path + plot_file_name
        plt.savefig(saving_dir)

    # DF Saving
    df_export = df_plot.iloc[-(len(df_pred_future) * 2) :]
    df_export = df_export.reset_index(drop=True)
    df_export["Model"] = model_name
    df_export["Actual"] = round(df_export["Actual"], 3)
    df_export["Prediction"] = round(df_export["Prediction"], 3)
    if not result_saving_for_tracking == "False":
        df_export.to_excel(
            save_path + "Values_PredActual.xlsx",
            sheet_name="Sheet1",
            index=False,
        )

    # DF Saving for Deepflow - "prediction.csv"
    df_deepflow_pred = df_export[
        [
            "dt",
            "Target",
            "Prediction",
        ]
    ]
    df_deepflow_pred.rename(
        columns={
            "dt": "snapshot_dt",
            "Target": "grain_id",
            "Prediction": "pred",
        },
        inplace=True,
    )
    df_deepflow_pred = df_deepflow_pred.iloc[-len(df_pred_future) :]
    df_deepflow_pred = df_deepflow_pred.reset_index(drop=True)
    df_deepflow_pred["snapshot_dt"] = df_deepflow_pred["snapshot_dt"][0]
    df_deepflow_pred = df_deepflow_pred.reset_index()
    df_deepflow_pred.rename(columns={"index": "h"}, inplace=True)
    df_deepflow_pred = df_deepflow_pred[["snapshot_dt", "grain_id", "h", "pred"]]
    df_deepflow_pred = pd.concat([df_deepflow_pred, df_ci_result], axis=1)
    df_deepflow_pred.rename(
        columns={"Lower_CI": "pred_min", "Upper_CI": "pred_max"}, inplace=True
    )
    df_deepflow_pred.to_csv(save_path + f"prediction.csv", index=False)

    # DF Saving for Deepflow - "price_history.csv"
    df_actual = df[["dt", target]]
    df_actual["grain_id"] = target
    df_actual[target] = round(df_actual[target], 3)
    df_actual.rename(columns={target: "v"}, inplace=True)
    df_actual = df_actual[["grain_id", "dt", "v"]]
    df_actual.to_csv(save_path + f"price_history.csv", index=False)

    # Metric Saving
    if not result_saving_for_tracking == "False":
        df_metric_testset = calculating_metric(df_pred_test, "TestSet")
        df_metric_testset.to_excel(
            save_path + "Metric_TestSet.xlsx",
            sheet_name="Sheet1",
            index=True,
        )

    return df_export


def merging_results(target_dict):
    df_prediction_result = pd.DataFrame()
    df_price_history_result = pd.DataFrame()
    df_shap_result = pd.DataFrame()

    save_path = "./final_output/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    current_date = datetime.now().strftime("%Y%m%d")

    for target in target_dict:
        # prediction.csv
        path = f"./output/{target}/_Result/"
        df_prediction = pd.read_csv(path + "prediction.csv")
        df_prediction["grain_id"] = target_dict[target]
        df_prediction_result = pd.concat([df_prediction_result, df_prediction], axis=0)
        df_prediction_result.to_csv(
            save_path + f"prediction_{current_date}.csv", index=False
        )

        # price_history.csv
        df_price_history = pd.read_csv(path + "price_history.csv")
        df_price_history["grain_id"] = target_dict[target]
        df_price_history_result = pd.concat(
            [df_price_history_result, df_price_history], axis=0
        )
        df_price_history_result.to_csv(
            save_path + f"price_history_{current_date}.csv", index=False
        )

        # shap.csv
        df_shap = pd.read_csv(path + "shap_result.csv")
        df_shap_result = pd.concat([df_shap_result, df_shap], axis=0)
        df_shap_result.to_csv(save_path + f"shap_{current_date}.csv", index=False)

    # Type Conversion
    df_prediction_result["snapshot_dt"] = pd.to_datetime(
        df_prediction_result["snapshot_dt"]
    )
    df_price_history_result["dt"] = pd.to_datetime(df_price_history_result["dt"])
    df_shap_result["snapshot_dt"] = pd.to_datetime(df_shap_result["snapshot_dt"])

    return


def clearing_all_files(result_saving_for_tracking):
    if result_saving_for_tracking == "False":
        delete_path = "./output"
        if os.path.exists(delete_path):
            shutil.rmtree(delete_path)
            print("Directory Cleared")

    return
