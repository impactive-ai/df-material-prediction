import pandas as pd

lag_num = 6


def pre_processing(df, target):
    # "date"를 월의 첫 날짜로 변경 - 추후 삭제
    df["Date"] = (pd.to_datetime(df["Date"]) - pd.offsets.MonthEnd() + pd.offsets.MonthBegin(1))
    # Rename - 추후 삭제
    df = df.rename(columns={"Date": "dt"})

    # Target(y)
    target_list = [
        "IronOre",
        "Nickel",
        "Coal",
        "CokingCoal",
        "Steel",
        "Copper",
        "Aluminum",
    ]
    df_y = pd.DataFrame(df[target])

    # Features(X)
    df_x = df.drop(target_list, axis=1).drop("dt", axis=1)

    # TS Features
    df_ts = pd.DataFrame(df["dt"])
    df_ts["Year"] = df_ts["dt"].dt.year
    df_ts["Month"] = df_ts["dt"].dt.month
    # df_ts['WeekByYear'] = df_ts['dt'].dt.isocalendar().week  # for Weekly
    # df_ts['WeekByMonth'] = df_ts['dt'].apply(lambda x: (x.day - 1) // 7 + 1)  # for Weekly

    # Lag Features
    df_lag = making_lag_features(df_y, target)

    # Concat
    df_processed = pd.concat([df_ts, df_lag, df_x], axis=1)
    df_processed = df_processed.iloc[lag_num:].reset_index().drop(["index"], axis=1)
    return df_processed, df_lag


def _target_split(df_input, target):
    df_input = df_input.drop(["dt"], axis=1)
    df_input_x = df_input.drop([target], axis=1)
    df_input_y = df_input[target].to_frame()
    return df_input_x, df_input_y


def data_split(df_input, target, valid_length, test_length):
    # Train Set
    df_train = df_input.iloc[: -(valid_length + test_length)]
    x_train, y_train = _target_split(df_train, target)
    # Valid Set
    df_valid = df_input.iloc[-(valid_length + test_length) : -test_length]
    x_valid, y_valid = _target_split(df_valid, target)
    # Test Set
    df_test = df_input.iloc[-test_length:]
    x_test, y_test = _target_split(df_test, target)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def making_lag_features(df_target, target_name):
    lag_columns = [f"lag_{i}" for i in range(1, lag_num + 1)]
    diff_columns = [f"delta_lag{i}_lag{i + 1}" for i in range(1, lag_num)]
    mean_columns = [f"mean_window_{window_size}" for window_size in [3, 6]]
    std_columns = [f"std_window_{window_size}" for window_size in [3, 6]]
    all_columns = lag_columns + diff_columns + mean_columns + std_columns
    df_lag_result = pd.DataFrame(columns=all_columns)
    df_lag_result = pd.concat([df_target, df_lag_result])
    # fill "lag"
    for lag in range(1, lag_num + 1):
        df_lag_result[f"lag_{lag}"] = df_lag_result[target_name].shift(lag)
    # fill "delta_lag"
    for i in range(1, lag_num):
        df_lag_result[f"delta_lag{i}_lag{i + 1}"] = (df_lag_result[f"lag_{i + 1}"] - df_lag_result[f"lag_{i}"]) / df_lag_result[f"lag_{i + 1}"]
    # fill "mean_window"
    df_lag_result["mean_window_3"] = df_lag_result[["lag_1", "lag_2", "lag_3"]].mean(axis=1)
    df_lag_result["mean_window_6"] = df_lag_result[lag_columns].mean(axis=1)
    # fill "std_window"
    df_lag_result["std_window_3"] = df_lag_result[["lag_1", "lag_2", "lag_3"]].std(axis=1)
    df_lag_result["std_window_6"] = df_lag_result[lag_columns].std(axis=1)
    return df_lag_result
