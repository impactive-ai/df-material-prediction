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


def plot_predictions(
    df,
    predictions,
    lower_ci_90,
    upper_ci_90,
    lower_ci_95,
    upper_ci_95,
    anchor=None,
    save_path=None,
):
    # 전체 기간 플롯
    df = df[df["date"] <= anchor]
    plt.figure(figsize=(18, 4))
    plt.plot(df.date, df.price, label="Actual")

    # 예측 날짜 생성 (마지막 날짜 이후의 1~6개월)
    last_date = df.date.iloc[-1]
    last_price = df.price.iloc[-1]
    if anchor is None:
        anchor = last_date

    prediction_dates = [
        pd.to_datetime(anchor) + relativedelta(months=i)
        for i in range(1, len(predictions) + 1)
    ]
    prediction_dates = [date.strftime("%Y-%m") for date in prediction_dates]

    plt.plot(
        [last_date] + prediction_dates,
        [last_price] + predictions,
        label="Prediction",
        color="red",
    )

    total_dates = df.date.tolist() + prediction_dates
    xtick_positions = np.linspace(0, len(total_dates) - 1, 10, dtype=int)
    xtick_labels = [total_dates[i] for i in xtick_positions]
    plt.xticks(xtick_positions, xtick_labels)

    plt.axvline(last_date, color="grey", linestyle="--")
    plt.legend()
    plt.grid(True)
    if save_path:
        entire_period_path = f"{save_path}_entire_period.png"
        plt.savefig(entire_period_path)
    plt.show()

    # 신뢰 구간 플롯 (최근 데이터 + 예측값)
    recent_actual_dates = df.date.iloc[-8:]
    recent_actual_values = df.price.iloc[-8:]

    plt.figure(figsize=(11, 5.5))
    plt.plot(
        recent_actual_dates,
        recent_actual_values,
        label="Actual",
        marker="o",
        color="blue",
    )
    plt.plot(
        [last_date] + [prediction_dates[0]],
        [last_price] + [predictions[0]],
        color="red",
    )
    plt.plot(prediction_dates, predictions, label="Prediction", color="red", marker="o")

    plt.fill_between(
        [last_date] + prediction_dates,
        [last_price] + lower_ci_90,
        [last_price] + upper_ci_90,
        color="orange",
        alpha=0.3,
        label="90% Confidence Interval",
    )
    plt.fill_between(
        [last_date] + prediction_dates,
        [last_price] + lower_ci_95,
        [last_price] + upper_ci_95,
        color="orange",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    combined_dates = list(recent_actual_dates) + prediction_dates
    plt.xticks(range(0, len(combined_dates), 2), combined_dates[::2], rotation=45)
    plt.axvline(last_date, color="grey", linestyle="--")
    # plt.ylim([35000, 45000])
    plt.legend()
    plt.grid(True)
    if save_path:
        zoomed_in_path = f"{save_path}_zoomed_prediction.png"
        plt.savefig(zoomed_in_path)
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
    df,
    save_path=None,
):
    plt.figure(figsize=(18, 4))
    plt.plot(
        list(train.date) + list(test.date), list(y_train) + list(y_test), label="Actual"
    )
    plt.plot(test.date, y_hat, label="Prediction", color="red")

    total_dates = train.date.tolist() + test.date.tolist()
    xtick_positions = np.linspace(0, len(total_dates) - 1, 8, dtype=int)
    xtick_labels = [total_dates[i] for i in xtick_positions]
    plt.xticks(xtick_positions, xtick_labels)

    plt.axvline(df[df["year"] == 2024]["date"].iloc[0], color="grey", linestyle="--")
    plt.legend()
    plt.grid(True)
    if save_path:
        entire_period_path = f"{save_path}_entire_period.png"
        plt.savefig(entire_period_path)
    plt.show()

    combined_dates = list(train.date[-8:]) + list(test.date)
    combined_values = list(y_train[-8:]) + list(y_test)

    plt.figure(figsize=(11, 5.5))
    plt.plot(combined_dates, combined_values, label="Actual", marker="o")
    plt.plot(test.date, y_hat, label="Prediction", color="red", marker="o")

    plt.fill_between(
        test.date,
        lower_ci_90,
        upper_ci_90,
        color="orange",
        alpha=0.3,
        label="90% Confidence Interval",
    )
    plt.fill_between(
        test.date,
        lower_ci_95,
        upper_ci_95,
        color="orange",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    plt.xticks(range(0, len(combined_dates), 2), combined_dates[::2])
    plt.axvline(df[df["year"] == 2024]["date"].iloc[0], color="grey", linestyle="--")
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
