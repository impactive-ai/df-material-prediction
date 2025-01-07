from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

xgb_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "subsample": [0.8, 0.9, 1],
}

rf_param_grid = {"n_estimators": [100, 200, 300, 350], "max_depth": [5, 10, 15, 20, 25]}

# 제품별 설정
products_mapping = {
    "cabbage": {
        "file_path": "data/cabbage_garak.csv",
        "model": XGBRegressor(random_state=42),
        "scaler": "standard",
        "param_grid": xgb_param_grid,
    },
    "strawberry": {
        "file_path": "data/strawberry_retail_1.csv",
        "model": RandomForestRegressor(random_state=42),
        "scaler": "mm",
        "param_grid": rf_param_grid,
    },
    "tomato": {
        "file_path": "data/tomato_wholesale_1.csv",
        "model": XGBRegressor(random_state=42),
        "scaler": "standard",
        "param_grid": xgb_param_grid,
    },
    "SMP": {
        "file_path": "data/SMP_NL.csv",
        "model": XGBRegressor(random_state=42),
        "scaler": "standard",
        "param_grid": xgb_param_grid,
    },
    "rice": {
        "file_path": "data/rice_retail_1.csv",
        "model": RandomForestRegressor(random_state=42),
        "scaler": "standard",
        "param_grid": rf_param_grid,
    },
}

# Parameters for each product
params_mapping = {
    "cabbage": {
        1: {
            "n_list": [1, 2, 3, 6, 7],
            "window": 12,
            "best_params": {
                "learning_rate": 0.2,
                "max_depth": 3,
                "n_estimators": 300,
                "subsample": 1,
            },
        },
        2: {
            "n_list": [2, 3, 6],
            "window": 8,
            "best_params": {
                "learning_rate": 0.2,
                "max_depth": 3,
                "n_estimators": 100,
                "subsample": 0.9,
            },
        },
        3: {
            "n_list": [3, 6, 7, 9, 10],
            "window": 11,
            "best_params": {
                "learning_rate": 0.3,
                "max_depth": 3,
                "n_estimators": 300,
                "subsample": 0.8,
            },
        },
        4: {
            "n_list": [4, 6, 8, 9, 11],
            "window": 10,
            "best_params": {
                "learning_rate": 0.2,
                "max_depth": 3,
                "n_estimators": 300,
                "subsample": 1,
            },
        },
        5: {
            "n_list": [5, 6, 7],
            "window": 12,
            "best_params": {
                "learning_rate": 0.01,
                "max_depth": 3,
                "n_estimators": 300,
                "subsample": 1,
            },
        },
        6: {
            "n_list": [6, 7, 8, 9, 11, 12, 13],
            "window": 17,
            "best_params": {
                "learning_rate": 0.2,
                "max_depth": 3,
                "n_estimators": 200,
                "subsample": 0.9,
            },
        },
    },
    "strawberry": {
        1: {
            "n_list": [1, 2, 3, 6, 7],
            "window": 9,
            "best_params": {"max_depth": 10, "n_estimators": 100},
        },
        2: {
            "n_list": [2, 4, 5, 6, 8],
            "window": 8,
            "best_params": {"max_depth": 10, "n_estimators": 200},
        },
        3: {
            "n_list": [3, 6, 7, 9, 10],
            "window": 11,
            "best_params": {"max_depth": 5, "n_estimators": 200},
        },
        4: {
            "n_list": [4, 7, 9, 10],
            "window": 16,
            "best_params": {"max_depth": 5, "n_estimators": 200},
        },
        5: {
            "n_list": [5, 6, 8],
            "window": 12,
            "best_params": {"max_depth": 5, "n_estimators": 100},
        },
        6: {
            "n_list": [6, 9, 13],
            "window": 18,
            "best_params": {"max_depth": 5, "n_estimators": 200},
        },
    },
    "tomato": {
        1: {
            "n_list": [1, 3, 7],
            "window": 8,
            "best_params": {
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 300,
                "subsample": 1,
            },
        },
        2: {
            "n_list": [2, 3],
            "window": 8,
            "best_params": {
                "learning_rate": 0.3,
                "max_depth": 3,
                "n_estimators": 200,
                "subsample": 1,
            },
        },
        3: {
            "n_list": [3, 4, 5, 8],
            "window": 11,
            "best_params": {
                "learning_rate": 0.3,
                "max_depth": 3,
                "n_estimators": 200,
                "subsample": 0.8,
            },
        },
        4: {
            "n_list": [4, 6, 8, 10, 11],
            "window": 14,
            "best_params": {
                "learning_rate": 0.2,
                "max_depth": 3,
                "n_estimators": 100,
                "subsample": 0.8,
            },
        },
        5: {
            "n_list": [5, 8, 9, 12],
            "window": 16,
            "best_params": {
                "learning_rate": 0.3,
                "max_depth": 3,
                "n_estimators": 100,
                "subsample": 0.8,
            },
        },
        6: {
            "n_list": [6, 7, 8, 9, 11, 12, 13],
            "window": 17,
            "best_params": {
                "learning_rate": 0.2,
                "max_depth": 3,
                "n_estimators": 300,
                "subsample": 0.9,
            },
        },
    },
    "SMP": {
        1: {
            "n_list": [1, 2],
            "window": 9,
            "best_params": {
                "learning_rate": 0.1,
                "max_depth": 10,
                "n_estimators": 200,
                "subsample": 0.8,
            },
        },
        2: {
            "n_list": [2, 3, 4],
            "window": 11,
            "best_params": {
                "learning_rate": 0.2,
                "max_depth": 5,
                "n_estimators": 100,
                "subsample": 0.8,
            },
        },
        3: {
            "n_list": [3, 4, 5, 6],
            "window": 13,
            "best_params": {
                "learning_rate": 0.2,
                "max_depth": 10,
                "n_estimators": 100,
                "subsample": 1,
            },
        },
        4: {
            "n_list": [4, 5],
            "window": 16,
            "best_params": {
                "learning_rate": 0.05,
                "max_depth": 5,
                "n_estimators": 300,
                "subsample": 0.8,
            },
        },
        5: {
            "n_list": [5, 6, 7],
            "window": 17,
            "best_params": {
                "learning_rate": 0.01,
                "max_depth": 10,
                "n_estimators": 200,
                "subsample": 0.9,
            },
        },
        6: {
            "n_list": [6, 7, 11, 13],
            "window": 14,
            "best_params": {
                "learning_rate": 0.1,
                "max_depth": 7,
                "n_estimators": 200,
                "subsample": 0.8,
            },
        },
    },
    "rice": {
        1: {
            "n_list": [1, 6, 7, 8, 9],
            "window": 11,
            "best_params": {"max_depth": 5, "n_estimators": 300},
        },
        2: {
            "n_list": [2],
            "window": 12,
            "best_params": {"max_depth": 5, "n_estimators": 200},
        },
        3: {
            "n_list": [3],
            "window": 12,
            "best_params": {"max_depth": 5, "n_estimators": 300},
        },
        4: {
            "n_list": [4],
            "window": 16,
            "best_params": {"max_depth": 5, "n_estimators": 350},
        },
        5: {
            "n_list": [5],
            "window": 17,
            "best_params": {"max_depth": 5, "n_estimators": 350},
        },
        6: {
            "n_list": [6],
            "window": 18,
            "best_params": {"max_depth": 5, "n_estimators": 300},
        },
    },
}

grain_id_mapping = {
    "cabbage": "DF00002",
    "strawberry": "DF00001",
    "tomato": "DF00003",
    "SMP": "DF00004",
    "rice": "DF00012",
}
