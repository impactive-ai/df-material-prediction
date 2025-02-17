# --------------------------------
# 파일 생성일: 2024/10/25
# 최종 수정일: 2024/12/31
# 파일 작성자: 한동대학교 강지은
# --------------------------------


import warnings
import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _add_features(df, lag, n_list=None, window=None):
    if n_list is not None:
        for n in n_list:
            df[f"v_prev_{n}"] = df["v_mean"].shift(n)

        if n_list != [1]:
            n_list_max = max(n_list)

            df[f"v_mw_mean_{n_list_max}"] = (
                df["v_mean"].shift(lag).rolling(window=n_list_max).mean()
            )
            df[f"v_mw_std_{n_list_max}"] = (
                df["v_mean"].shift(lag).rolling(window=n_list_max).std()
            )
            df[f"v_mw_min_{n_list_max}"] = (
                df["v_mean"].shift(lag).rolling(window=n_list_max).min()
            )
            df[f"v_mw_max_{n_list_max}"] = (
                df["v_mean"].shift(lag).rolling(window=n_list_max).max()
            )

    if window is not None:
        df[f"v_mw_mean_{window}"] = (
            df["v_mean"].shift(lag).rolling(window=window).mean()
        )
        df[f"v_mw_std_{window}"] = df["v_mean"].shift(lag).rolling(window=window).std()

    df.dropna(inplace=True)

    return df


def _monthly_shift(df: pd.DataFrame, suffix="", lag=0):
    df["v_mean"] = df["v_mean"].astype(int)
    df[f"v_low_{lag}{suffix}"] = df["v_low"].shift(lag)
    df[f"v_high_{lag}{suffix}"] = df["v_high"].shift(lag)
    df[f"v_std_{lag}{suffix}"] = df["v_std"].shift(lag)

    df.drop(columns=["v_low", "v_high", "v_std"], inplace=True)
    df.dropna(inplace=True)

    return df


def _add_november_rows(df):
    from datetime import datetime

    start_year = df["dt"].min().year
    end_year = df["dt"].max().year

    november_dates = [datetime(year, 11, 1) for year in range(start_year, end_year + 1)]

    last_date = df["dt"].max()
    if last_date < datetime(end_year, 11, 1):
        november_dates = november_dates[:-1]

    november_df = pd.DataFrame({"dt": november_dates})
    november_df["v_mean"] = 2000
    november_df["v_low"] = 0
    november_df["v_high"] = 0
    november_df["v_std"] = 0

    final_df = (
        pd.concat([df, november_df], ignore_index=True)
        .sort_values(by="dt")
        .reset_index(drop=True)
    )

    return final_df


def _add_predict_row(df, lag):
    last_date = df.dt.iloc[-1]

    for m in range(1, lag + 1):
        predict_date = last_date + relativedelta(months=m)
        predict_row = pd.DataFrame(
            {
                "dt": [predict_date],
                "v_low": [0.0],
                "v_high": [0.0],
                "v_mean": [0.0],
                "v_std": [0.0],
            }
        )
        df = pd.concat([df, predict_row], ignore_index=True)

    return df


def _add_month_onehot(df, valid_months=[1,2,3,4,5,11,12], prefix="month"):
    month = df["dt"].apply(lambda x: x.month)
    one_hot = pd.get_dummies(month, prefix=prefix)
    valid_months = [f"{prefix}_{x}" for x in valid_months]
    one_hot = one_hot[[col for col in one_hot.columns if col in valid_months]]
    df = pd.concat([df, one_hot], axis=1)
    return df


def _load_source(filepath: str):
    import pandas as pd

    pdf = pd.read_parquet(filepath)
    if "dt" not in pdf.columns:
        raise ValueError("dt column not found")

    pdf["dt"] = pd.to_datetime(pdf["dt"])
    # pdf = pdf.set_index("dt")

    return pdf


def _merge_external(*, extdata_path_list: list[str], dt_from, dt_to, freq="MS"):
    import pandas as pd

    out_pdf = pd.DataFrame({"dt": pd.date_range(dt_from, dt_to, freq=freq)})

    for filepath in extdata_path_list:
        pdf = _load_source(filepath)
        out_pdf = pd.merge(out_pdf, pdf, on="dt", how="left")

    return out_pdf


EXT_COLUMNS = [
    "dt",
    "KOSIS022",
    "KOSIS023",
    "KOSIS024",
    "KOSIS025",
    "KOSIS026",
    "KOSIS027",
    "KOSIS028",
    "KOSIS029",
    "KOSIS030",
    "KOSIS031",
    "KOSIS032",
    "KOSIS033",
    "KOSIS034",
    "KOSIS056",
    "KOSIS077",
    "KOSIS087",
    "KOSIS088",
    "KOSIS089",
    "KOSIS090",
    "KOSIS091",
    "KOSIS092",
    "KOSIS093",
    "KOSIS094",
    "KOSIS095",
    "KOSIS096",
    "KOSIS097",
    "KOSIS098",
    "KOSIS099",
    "KOSIS100",
    "KOSIS101",
    "KOSIS102",
    "KOSIS103",
    "KOSIS104",
    "KOSIS105",
    "KOSIS106",
    "KOSIS107",
    "KOSIS108",
    "KOSIS109",
    "KOSIS110",
    "평균기온(°C)",
    "평균최고기온(°C)",
    "평균최저기온(°C)",
    "최고기온(°C)",
    "최저기온(°C)",
    "평균해면기압(hPa)",
    "평균수증기압(hPa)",
    "평균상대습도(%)",
    "월합강수량(00~24h만)(mm)",
    "평균풍속(m/s)",
    "최대풍속(m/s)",
    "평균운량(1/10)",
    "일조율(%)",
    "합계 일사량(MJ/m2)",
    "평균 최저초상온도(°C)",
    "평균지면온도(°C)",
]


def _add_external_data(df, ext_list, ref_date, lag=1):
    df_exo = _merge_external(
        extdata_path_list=ext_list,
        dt_from="2000-01-01",
        dt_to=ref_date,
    )

    df_exo = df_exo[EXT_COLUMNS]

    # df_exo["dt"] = df_exo["dt"].apply(lambda x: x + relativedelta(months=lag))
    df_exo["dt"] = df_exo["dt"] + pd.DateOffset(months=lag)
    df = pd.merge(df, df_exo, on="dt", how="inner")

    return df


def _count_weekends(dt):
    # 해당 월의 첫 번째 날짜와 마지막 날짜 계산
    first_day = dt.replace(day=1)
    last_day = (first_day + pd.offsets.MonthEnd(0)).date()
    dates = pd.date_range(first_day, last_day)
    weekends = dates[dates.weekday >= 5]  # 5는 토요일, 6은 일요일

    return len(weekends)


def _count_holidays(dt):
    import holidays

    kr_holidays = holidays.KR()
    first_day = dt.replace(day=1)
    last_day = (first_day + pd.offsets.MonthEnd(0)).date()
    dates = pd.date_range(first_day, last_day)
    holidays_in_month = [date for date in dates if date in kr_holidays]

    return len(holidays_in_month)


def preprocess_data(
    input_path: str,
    ext_list: list[str],
    product: str,
    lag,
    ref_month,
    n_list: list[int] = None,
    window: int = None,
    is_test: bool = False,
):
    """
    데이터 전처리
    :param input_path: 입력 데이터 경로
    :param ext_list: 외부 데이터 경로 목록
    :param product: 품목 종류
    :param lag:
    :param ref_month:
    :param n_list: (from config)
    :param window: (from config)
    :param is_test:
    :return: 전처리된 데이터
    """

    from config import grain_id_mapping

    raw_df = pd.read_parquet(input_path)
    raw_df = raw_df[raw_df["grain_id"] == grain_id_mapping[product]].drop_duplicates()

    if product == "SMP":
        raw_df.loc[raw_df["v_count"] == 1, "v_std"] = 0
        
    raw_df = raw_df[['dt', 'v_low', 'v_high', 'v_mean', 'v_std']]
    
    raw_df["dt"] = pd.to_datetime(raw_df["dt"])
    df = raw_df[raw_df["dt"].dt.date < ref_month].copy()
    
    if is_test:
        df = _add_predict_row(df, lag)

    # 제품별 전처리
    if product == "cabbage":
        df = _monthly_shift(df, lag=lag)
        df["holidays"] = df["dt"].apply(_count_holidays)
    elif product == "strawberry":
        df = _add_november_rows(df)
        df = _monthly_shift(df, lag=lag)
        df = _add_month_onehot(df)
    elif product == "tomato":
        df.drop(columns=["v_low", "v_high", "v_std"], inplace=True)
        df["weekends"] = df["dt"].apply(_count_weekends)
    elif product == "SMP":
        df = _monthly_shift(df, lag=lag)
        df["holidays"] = df["dt"].apply(_count_holidays)
        df["weekends"] = df["dt"].apply(_count_weekends)
    elif product == "rice":
        df.drop(columns=["v_low", "v_high", "v_std"], inplace=True)

    df = _add_features(df, n_list[0], n_list, window)
    df = _add_external_data(df, ext_list, ref_month, lag=n_list[0])

    return df


def prepare_data(df, exclude_col: list[str] = None):
    exclude_col = exclude_col or []
    drop_cols = ["v_mean", "dt"] + exclude_col

    X = df.drop(columns=drop_cols)
    y = df["v_mean"]

    return X, y


def scale_data(X_train, X_test, scale="standard"):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    scaler = StandardScaler() if scale == "standard" else MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler
