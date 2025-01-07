from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import pandas as pd
import warnings

from run_cli import PredictionParameter


def train_and_predict(param: PredictionParameter):
    # TODO impl
    ref_date = param.ref_date
    input_path = param.input_path
    ext_path_list = param.ext_path_list
    output_path = param.output_path

    print(ref_date, input_path, ext_path_list, output_path)


def main():
    from run_cli import run_cli

    param = run_cli()
    train_and_predict(param)


if __name__ == "__main__":
    main()
