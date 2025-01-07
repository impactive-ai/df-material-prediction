import numpy as np


def calculate_confidence_intervals(
    model, X_train, y_train, X_test, method="bootstrap", n_samples=100
):
    from sklearn.ensemble import RandomForestRegressor

    if isinstance(model, RandomForestRegressor):
        # RandomForest의 경우, 트리별 예측을 이용하여 신뢰 구간 계산
        predictions = np.array([tree.predict(X_test) for tree in model.estimators_])
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)

    elif (
        method == "bootstrap"
    ):  # XGBoost의 경우 부트스트랩 샘플링을 통해 신뢰 구간 계산
        from xgboost import XGBRegressor
        from sklearn.utils import resample

        predictions = []
        for _ in range(n_samples):
            X_resampled, y_resampled = resample(X_train, y_train)

            model_boot = XGBRegressor(**model.get_params())
            model_boot.fit(X_resampled, y_resampled)

            preds = model_boot.predict(X_test)
            predictions.append(preds)

        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
    else:
        raise ValueError(f"mean_prediction, std_prediction must be initialized.")

    # 90% 신뢰 구간 (z = 1.645)
    lower_ci_90 = mean_prediction - 1.645 * std_prediction
    upper_ci_90 = mean_prediction + 1.645 * std_prediction

    # 95% 신뢰 구간 (z = 1.96)
    lower_ci_95 = mean_prediction - 1.96 * std_prediction
    upper_ci_95 = mean_prediction + 1.96 * std_prediction

    return std_prediction, lower_ci_90, upper_ci_90, lower_ci_95, upper_ci_95


def load_models_and_scalers(directory, product, months_list: list[int]):
    import os
    import joblib

    loaded_models = {}
    loaded_scalers = {}

    for months in months_list:
        model_path = os.path.join(directory, f"model_{product}_{months}_months.pkl")
        scaler_path = os.path.join(directory, f"scaler_{product}_{months}_months.pkl")
        loaded_models[months] = joblib.load(model_path)
        loaded_scalers[months] = joblib.load(scaler_path)

    return loaded_models, loaded_scalers


def run_model(model, param_grid, X_train, y_train):
    from sklearn.model_selection import GridSearchCV

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def valid_date(s):
    try:
        from datetime import datetime

        return datetime.strptime(s, "%Y-%m-%d")  # 날짜 포맷: YYYY-MM-DD
    except ValueError:
        import argparse

        raise argparse.ArgumentTypeError(
            f"Invalid date: '{s}'. Expected format: YYYY-MM-DD"
        )
