# --------------------------------
# 파일 생성일: 2024/10/25
# 최종 수정일: 2024/12/31
# 파일 작성자: 한동대학교 강지은
# --------------------------------


import warnings
import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _add_features(df, months_ahead, n_list=None, window=None):
    if n_list is not None:
        for n in n_list:
            df[f"price_prev_{n}"] = df["price"].shift(n)

        if n_list != [1]:
            n_list_max = max(n_list)

            df[f"price_mw_mean_{n_list_max}"] = (
                df["price"].shift(months_ahead).rolling(window=n_list_max).mean()
            )
            df[f"price_mw_std_{n_list_max}"] = (
                df["price"].shift(months_ahead).rolling(window=n_list_max).std()
            )
            df[f"price_mw_min_{n_list_max}"] = (
                df["price"].shift(months_ahead).rolling(window=n_list_max).min()
            )
            df[f"price_mw_max_{n_list_max}"] = (
                df["price"].shift(months_ahead).rolling(window=n_list_max).max()
            )

    if window is not None:
        df[f"price_mw_mean_{window}"] = (
            df["price"].shift(months_ahead).rolling(window=window).mean()
        )
        df[f"price_mw_std_{window}"] = (
            df["price"].shift(months_ahead).rolling(window=window).std()
        )

    df.dropna(inplace=True)

    return df


def _data_to_monthly(df: pd.DataFrame, suffix="", lag=0):
    df = (
        df.groupby(["date"])
        .agg(
            price=("price", "mean"),
            price_min=("price", "min"),
            price_max=("price", "max"),
            price_std=("price", "std"),
            year=("year", "first"),
            month=("month", "first"),
        )
        .fillna(0)
        .reset_index()
    )

    df["price"] = df["price"].astype(int)
    df[f"price_min_{lag}{suffix}"] = df["price_min"].shift(lag)
    df[f"price_max_{lag}{suffix}"] = df["price_max"].shift(lag)
    df[f"price_std_{lag}{suffix}"] = df["price_std"].shift(lag)

    df.drop(columns=["price_min", "price_max", "price_std"], inplace=True)
    df.dropna(inplace=True)

    return df


def _fill_missing_months(df, end_year, end_month, value=0, daily=False):
    start_year = df.iloc[0]["year"]
    start_month = df.iloc[0]["month"]
    # 날짜 범위 생성
    start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
    end_date = pd.Timestamp(year=end_year, month=end_month, day=1)
    date_range = pd.date_range(
        start=start_date, end=end_date, freq="MS"
    )  # 월별 시작일로 범위 생성

    # DataFrame의 기존 날짜를 기준으로 새 DataFrame 생성
    all_dates = pd.DataFrame({"year": date_range.year, "month": date_range.month})

    merged = pd.merge(all_dates, df, on=["year", "month"], how="left")
    merged["price"] = merged["price"].fillna(value)

    if daily:
        merged["prev_price"] = merged["prev_price"].fillna(value)
        merged["prev_change"] = merged["prev_change"].fillna(value)
        merged["month_price"] = merged["month_price"].fillna(value)
        merged["month_change"] = merged["month_change"].fillna(value)
        merged["year_price"] = merged["year_price"].fillna(value)
        merged["year_change"] = merged["year_change"].fillna(value)
        merged["avg_year_price"] = merged["avg_year_price"].fillna(value)

    # 가격이 없는 행을 0으로 채움
    merged["date"] = merged.apply(
        lambda row: f"{row['year']}-{row['month']:02d}", axis=1
    )

    return merged


def _add_predict_row(df, month):
    last_date = pd.to_datetime(df.date.iloc[-1])

    for m in range(1, month + 1):
        predict_date = last_date + relativedelta(months=m)
        predict_row = pd.DataFrame(
            {
                "date": [predict_date.strftime("%Y-%m")],
                "year": [predict_date.year],
                "month": [predict_date.month],
                "price": [0.0],
            }
        )
        df = pd.concat([df, predict_row], ignore_index=True)
    print("predicted_date:", df["date"].iloc[-1])
    return df


def _add_daily_strawberry(df, date, lag=0):
    df_daily = pd.read_csv(f"strawberry_retail_1_daily.csv")
    df_daily.drop(columns=["product", "unit"], inplace=True)
    df_daily = df_daily[::-1].reset_index(drop=True)
    df_daily = _fill_missing_months(df_daily, int(date[:4]), int(date[-2:]), daily=True)
    df_daily = _data_to_monthly(df_daily, suffix="_daily", lag=lag)
    df_daily.drop(columns=["price"], inplace=True)
    df = pd.merge(df, df_daily, on=["date", "year", "month"], how="inner")
    # df.tail()

    return df


def _add_month_onehot(df):
    one_hot = pd.get_dummies(df["month"], prefix="month")
    df = pd.concat([df, one_hot], axis=1)
    return df


def _add_external_data(df, lag=1, exo_file="external_data_202501.csv"):
    # TODO 두개의 외부데이터 파일을 참조하는데 하나로 통합할수 없는지? 저 파일은 어떻게 가공해서 나온건지 궁금합니다.
    df_exo = pd.read_csv(exo_file)
    external_data_list = pd.read_csv("external_data_used_20241129.csv")
    external_data_list = external_data_list.values.flatten()  # projction
    df_exo = df_exo[external_data_list]

    df_exo["date"] = pd.to_datetime(df_exo["dt"]).apply(
        lambda x: (x + relativedelta(months=lag)).strftime("%Y-%m")
    )
    df_exo.drop(columns=["dt"], inplace=True)
    df = pd.merge(df, df_exo, on="date", how="inner")
    # df.tail()
    return df


def _count_weekends(year, month):
    dates = pd.date_range(
        f"{int(year)}-{month:02d}-01",
        periods=pd.Period(year=year, month=month, freq="M").days_in_month,
    )
    weekends = dates[dates.weekday >= 5]  # 5는 토요일, 6은 일요일

    return len(weekends)


def _count_holidays(year, month):
    import holidays

    kr_holidays = holidays.KR()
    dates = pd.date_range(
        f"{int(year)}-{month:02d}-01",
        periods=pd.Period(year=year, month=month, freq="M").days_in_month,
    )
    holidays_in_month = [date for date in dates if date in kr_holidays]
    return len(holidays_in_month)


def _add_holiday(df, weekend=False, holiday=False):
    if weekend:
        df["weekends"] = df.apply(
            lambda row: _count_weekends(row["year"], row["month"]), axis=1
        )
    if holiday:
        df["holidays"] = df.apply(
            lambda row: _count_holidays(row["year"], row["month"]), axis=1
        )
    return df


def preprocess_data(
    file_path: str,
    product: str,
    month,
    ref_month,
    n_list: list[int] = None,
    window: int = None,
    is_test: bool = False,
):
    """
    데이터 전처리
    :param file_path: 입력 파일 경로
    :param product: 품목 종류
    :param month:
    :param ref_month:
    :param n_list: (from config)
    :param window: (from config)
    :param is_test:
    :return: 전처리된 데이터
    """
    raw_df = pd.read_csv(file_path)
    raw_df = raw_df[raw_df.price != 0]
    raw_df = raw_df[raw_df["date"] < ref_month]

    if product == "strawberry":
        raw_df = _fill_missing_months(
            raw_df.copy(), int(ref_month[:4]), int(ref_month[-2:])
        )
        # TODO 매년 11월 가격 2000원 지정. 어떤 의도인가요? 설명 부탁
        raw_df.loc[raw_df["month"] == 11, "price"] = 2000
        raw_df = raw_df[raw_df.price != 0]
        df = raw_df.copy()
    else:
        df = raw_df.copy()

    if is_test:
        df = _add_predict_row(df, month)

    # 제품별 전처리
    if product == "cabbage":
        df = _data_to_monthly(df, lag=month)
        df = _add_holiday(df, holiday=True)
    elif product == "strawberry":
        df = _add_daily_strawberry(df, ref_month, lag=month)
        df = _add_month_onehot(df)
    elif product == "tomato":
        df = _add_holiday(df, weekend=True)
    elif product == "SMP":
        df = _data_to_monthly(df, lag=month)
        df = _add_holiday(df, weekend=True, holiday=True)
    elif product == "rice":
        pass  # 그대로 사용

    df = _add_features(df, n_list[0], n_list, window)
    df = _add_external_data(df, lag=n_list[0])

    return df


def prepare_data(df, exclude_col: list[str] = None, month=False):
    exclude_col = exclude_col or []

    if month:
        drop_cols = ["price", "year", "date"] + exclude_col
    else:
        drop_cols = ["price", "year", "month", "date"] + exclude_col

    X = df.drop(columns=drop_cols)
    y = df["price"]

    return X, y


def scale_data(X_train, X_test, scale="standard"):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    scaler = StandardScaler() if scale == "standard" else MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler
