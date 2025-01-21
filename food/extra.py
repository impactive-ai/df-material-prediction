"""
서버에서 동작할 때 불필요해보이는 함수들을 별도로 분리함
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta


def interpret_shap(model, X_test, show_plot=False):
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "shap_value_mean_abs": np.mean(np.abs(shap_values), axis=0),
        }
    )  # .sort_values(by='shap_value_mean_abs', ascending=False)
    print(shap_importance)

    if show_plot:
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        plt.show()


def print_result_plot(
    train,
    test,
    y_train,
    y_test,
    y_hat,
    lower_ci_90,
    upper_ci_90,
    lower_ci_95,
    upper_ci_95,
    save_path=None,
):
    plt.figure(figsize=(18, 4))
    plt.plot(
        list(train.dt) + list(test.dt), list(y_train) + list(y_test), label="Actual"
    )
    plt.plot(test.dt.tolist(), y_hat, label="Prediction", color="red")

    plt.axvline(test["dt"].iloc[0], color="grey", linestyle="--")
    plt.legend()
    plt.grid(True)
    if save_path:
        entire_period_path = f"{save_path}_entire_period.png"
        plt.savefig(entire_period_path)
    plt.show()

    combined_dates = list(train.dt[-8:]) + list(test.dt)
    combined_values = list(y_train[-8:]) + list(y_test)

    plt.figure(figsize=(11, 5.5))
    plt.plot(combined_dates, combined_values, label="Actual", marker="o")
    plt.plot(test.dt, y_hat, label="Prediction", color="red", marker="o")

    plt.fill_between(
        test.dt,
        lower_ci_90,
        upper_ci_90,
        color="orange",
        alpha=0.3,
        label="90% Confidence Interval",
    )
    plt.fill_between(
        test.dt,
        lower_ci_95,
        upper_ci_95,
        color="orange",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    plt.axvline(test["dt"].iloc[0], color="grey", linestyle="--")
    plt.legend()
    plt.grid(True)
    if save_path:
        zoomed_in_path = f"{save_path}_zoomed.png"
        plt.savefig(zoomed_in_path)
    plt.show()


def calc_metrics(y_true, y_pred):
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_percentage_error,
        mean_absolute_error,
        r2_score,
    )

    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 결과 출력
    print(f"MAPE: {mape:.2f}%")
    print(f"nRMSE: {nrmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
