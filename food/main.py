from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import pandas as pd
import warnings

from run_cli import PredictionParameter

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def run_pipeline(
    df: pd.DataFrame,
    model,
    *,
    test_window: int,
    param_grid=None,
    best_params=None,
    scale="standard",
    exclude_col: list[str] | None = None,
    show_result=False,
    show_shap=False,
):
    from featurization import prepare_data, scale_data
    from sklearn.metrics import mean_absolute_percentage_error

    exclude_col = exclude_col or []

    train_df = df.iloc[:-test_window]
    test_df = df.iloc[-test_window:]

    X_train, y_train = prepare_data(train_df, exclude_col)
    X_test, y_test = prepare_data(test_df, exclude_col)
    X_train, X_test_scaled, scaler = scale_data(X_train, X_test, scale)

    if best_params is None:
        from util import run_model

        best_model, best_params = run_model(model, param_grid, X_train, y_train)
        if show_result:
            print("Best Params:", best_params)
    else:
        model.set_params(**best_params)
        best_model = model.fit(X_train, y_train)

    y_hat = best_model.predict(X_test_scaled)
    mape = mean_absolute_percentage_error(y_test, y_hat)

    # plot_metrics(y_test, y_hat)

    if show_shap:
        from extra import interpret_shap

        # 기본적으로 shap_importance 출력, PRINT시 summary plot 출력
        interpret_shap(best_model, X_test, show_result)

    if show_result:
        from extra import print_result_plot
        from util import calculate_confidence_intervals

        if isinstance(best_model, RandomForestRegressor):
            _, lower_ci_90, upper_ci_90, lower_ci_95, upper_ci_95 = (
                calculate_confidence_intervals(
                    best_model, X_train, y_train, X_test_scaled, method="tree"
                )
            )
        else:
            _, lower_ci_90, upper_ci_90, lower_ci_95, upper_ci_95 = (
                calculate_confidence_intervals(
                    best_model, X_train, y_train, X_test_scaled, method="bootstrap"
                )
            )

        # plot_path = f"test_result/test_{product}_{month}_months"
        # os.makedirs("test_result", exist_ok=True)

        # print_result_plot(
        #     train_df,
        #     test_df,
        #     y_train,
        #     y_test,
        #     y_hat,
        #     lower_ci_90,
        #     upper_ci_90,
        #     lower_ci_95,
        #     upper_ci_95,
        #     save_path=plot_path,
        # )

    return round(mape, 4), best_model, scaler, test_df.dt, y_test.values, y_hat


def train(
    param,
    product,
    config,
    months_list: list[int],
    params_mapping,
    ref_month,
    exclude_col: list[str],
):
    from featurization import preprocess_data

    model_dict = {}
    scaler_dict = {}

    results = []

    model_class = config["model"]
    # param_grid = config["param_grid"]

    for month in months_list:
        params_month = params_mapping[month]

        print(f"Processing {product}, {month} months ahead...")
        df = preprocess_data(
            param.input_path,
            param.ext_path_list,
            product,
            month,
            ref_month,
            n_list=params_month["n_list"],
            window=params_month["window"],
        )

        mape, model, scaler, dates, test, pred = run_pipeline(
            df,
            model_class(random_state=42),
            best_params=params_month[
                "best_params"
            ],  # 최적화된 best_params 사용 또는 param_grid=param_grid로 파라미터 탐색
            scale=config["scaler"],
            show_result=True,
            test_window=6,
            exclude_col=exclude_col,
        )  # 최근 6개월 test

        for date_, price_test, price_pred in zip(dates, test, pred):
            results.append(
                {
                    "product": product,
                    "month": month,
                    "date": date_,
                    "price_test": price_test,
                    "price_pred": price_pred,
                }
            )

        model_dict[month] = model
        scaler_dict[month] = scaler

    return model_dict, scaler_dict, results


def predict(
    param,
    models,
    scalers,
    months_params,
    product,
    this_month,
    months_list: list[int],
    exclude_col: list[str] = None,
    show_shap=False,
):
    from featurization import prepare_data, preprocess_data
    from util import calculate_confidence_intervals

    future_predictions = []
    shap_values = None if not show_shap else {}
    lower_ci_90_list, upper_ci_90_list = [], []
    exclude_col = exclude_col or []

    for month in months_list:
        model = models[month]
        scaler = scalers[month]
        params = months_params[month]
        
        df = preprocess_data(
            param.input_path,
            param.ext_path_list,
            product,
            month,
            this_month,
            n_list=params["n_list"],
            window=params["window"],
            is_test=True,
        )

        X, _ = prepare_data(df, exclude_col)
        X_pred = pd.DataFrame(X.iloc[-1, :].values.reshape(1, -1), columns=X.columns)
        X_scaled = scaler.transform(X_pred)
        y_hat = model.predict(X_scaled)
        future_predictions.extend(y_hat)

        print(
            f'Predicting for {product}, {month} months ahead: {df["dt"].iloc[-1]}, price: {y_hat[0]:.2f}'
        )

        # 신뢰구간 산출
        std_pred, _, _, _, _ = calculate_confidence_intervals(
            model,
            X,
            df["v_mean"],
            X_scaled,
            method="bootstrap" if isinstance(model, XGBRegressor) else "tree",
        )
        # FIXME
        lower_ci_90_list.extend(y_hat - 1.645 * std_pred)
        upper_ci_90_list.extend(y_hat + 1.645 * std_pred)

        if show_shap:
            from extra import interpret_shap

            shap_importance = interpret_shap(model, X_pred)
            shap_values[f"{month}_months"] = shap_importance

    return (
        future_predictions,
        shap_values,
        lower_ci_90_list,
        upper_ci_90_list,
    )


def run(param: PredictionParameter):
    # from util import load_models_and_scalers
    from config import products_mapping, params_mapping, grain_id_mapping
    from datetime import date

    horizon = 7
    horizon_list = list(range(1, horizon + 1))  # n개월 예측

    this_month = param.ref_date.date()  # 예측 시점 날짜 (년-월)
    today_fn = date.today().strftime("%Y%m%d")

    combined_shap = []
    combined_test_results = []
    combined_predictions = []
    exclude_col = []

    for product, config in products_mapping.items():
        print(f"Training and saving models for {product}...")
        models, scalers, results = train(
            param,
            product,
            config,
            horizon_list,
            params_mapping[product],
            this_month,
            exclude_col,
        )

        combined_test_results.append(results)

        # 예측 실행
        (
            predictions,
            shap_values,
            lower_ci_90_list,
            upper_ci_90_list,
        ) = predict(
            param,
            models,
            scalers,
            params_mapping[product],
            product,
            this_month,
            horizon_list,
            exclude_col,
        )

        # 예측 결과 정리
        pred = pd.DataFrame(
            {
                "snapshot_dt": [this_month] * horizon,
                "grain_id": [grain_id_mapping[product]] * horizon,
                "h": list(range(horizon)),
                "pred": predictions,
                "pred_min": [max(x, 0.0) for x in lower_ci_90_list],
                "pred_max": upper_ci_90_list,
            }
        )

        combined_predictions.append(pred)

        # SHAP 결과 처리 (True 인 경우)
        if shap_values:
            shap_df = pd.concat(
                [df.assign(month=month) for month, df in shap_values.items()],
                ignore_index=True,
            )
            shap_df["product"] = product
            combined_shap.append(shap_df)

    # 예측 결과 csv 저장
    if combined_predictions:
        all_predictions = pd.concat(combined_predictions, ignore_index=True)
        all_predictions.to_csv(param.output_path, index=False, float_format="%.2f")
        print(f"Predictions saved successfully.")

    # SHAP 결과 csv 저장 (True 인 경우)
    if combined_shap:
        final_combined_df = pd.concat(combined_shap, ignore_index=True)
        final_combined_df.to_csv(f"shap_results_{today_fn}.csv", index=False)
        print("SHAP results saved successfully.")


def main():
    from run_cli import run_cli

    param = run_cli()
    run(param)


if __name__ == "__main__":
    main()
