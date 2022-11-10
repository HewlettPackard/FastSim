"""
Explore feature importance for application power usage via decision tress
"""

import argparse, os, sys

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from funcs import parse_cache, timelimit_str_to_timedelta, hour_to_timeofday


POWER_COLS = ["JobID", "Start", "End", "ConsumedEnergyRaw", "AllocNodes"]
SUBMIT_COLS = ["ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem", "Timelimit", "Submit"]
# SUBMIT_COLS = ["ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem",
#                "Submit", "JobName", "Timelimit", "SubmitLine"]
FINISH_COLS = [] # TODO

def clean_df(df, queue_only=False):
    df = df.loc[
        (
            (df.Group.notna()) & (df.ReqMem.notna()) & (~df.ReqMem.str.contains("\?")) &
            (df.QOS.notna()) & (df.Timelimit.notna()) &
            (df.Timelimit != "Partition_Limit") & (df.Timelimit != "UNLIMITED")
            # & (df.SubmitLine.notna())
        )
    ]

    cols = ["ReqCPUS", "ReqNodes", "ReqMem", "AllocNodes"]
    df[cols] = df[cols].replace(
        { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    ).astype(float).astype(int)

    df.Submit = pd.to_datetime(df.Submit, format="%Y-%m-%dT%H:%M:%S")
    df.Submit = df.Submit.apply(lambda row: hour_to_timeofday(row.hour))

    df.Timelimit = df.Timelimit.apply(
        lambda row: round(timelimit_str_to_timedelta(row).total_seconds() / 60)
    )

    df["PowerPerNode"] = df.apply(lambda row: float(row.Power) / float(row.AllocNodes), axis=1)

    if not queue_only:
        pass # TODO loc + replace for new cols

    return df

def main(args):
    df_power = parse_cache(
        args.data,
        args.cache,
        ".".join(os.path.basename(args.data).split(".")[:-1]),
        "decision_tree_queue_df",
        cols=POWER_COLS+SUBMIT_COLS if args.queue_data else POWER_COLS+SUBMIT_COLS+FINISH_COLS
    )

    df_power = clean_df(df_power, queue_only=args.queue_data)

    # print(np.sort(df_power.ReqCPUS.unique()))
    # print(np.sort(df_power.ReqNodes.unique()))
    # print(np.sort(df_power.Group.unique()))
    # print(np.sort(df_power.ReqMem.unique()))
    # print(np.sort(df_power.QOS.unique()))
    # print(np.sort(df_power.JobName.unique()))
    # print(len(df_power.JobName.unique()), len(df_power.JobName))
    # print(df_power.JobName.value_counts()[:10].index.tolist())
    # print(df_power.Submit.unique())
    # print(df_power.Submit)
    # print(df_power.SubmitLine)
    # print(np.sort(df_power.SubmitLine.unique()))
    # print(np.sort(df_power.Timelimit.unique()))


    target = df_power.PowerPerNode
    data = df_power.drop(
        [
            "JobID", "Start", "End", "ConsumedEnergyRaw", "AllocNodes", "DeltaT", "Power",
            "PowerPerNode"
        ],
        axis=1
    )

    numerical_columns = ["ReqCPUS", "ReqNodes", "ReqMem", "Timelimit"]
    categorical_columns = ["Group", "QOS", "Submit"]

    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()

    encoder = ColumnTransformer([
        ('one-hot-encoder', categorical_preprocessor, categorical_columns),
        ('standard_scaler', numerical_preprocessor, numerical_columns)
    ])

    print(data)
    print(encoder.fit_transform(data))
    print(type(encoder.fit_transform(data)))
    print(encoder.fit_transform(data).shape)
    print(len(data.Group.unique()) + len(data.QOS.unique()) + len(data.Submit.unique()))
    sys.exit()

    model = DecisionTreeRegressor()

    model = make_pipeline(preprocessor, DecisionTreeRegressor())

    data_train, data_test, target_train, target_test = train_test_split(
        data, target, random_state=1
    )

    model.fit(data_train, target_train)

    features = model['preprocessor'].transformers_[1][1]['one-hot-encoder'].get_feature_names(categorical_columns)
    print(features)
    # print(model["decisiontreeregressor"].feature_importances_)
    # for feature, importance in zip(model["decisiontreeregressor"].feature_names_in_, model["decisiontreeregressor"].feature_importances_):
    #     print(feature, importance)

    train_mean_power = target_train.mean()
    print("Test Set:")
    data_test["PowerPrediction"] = model.predict(data_test)
    data_test["TruePower"] = target_test
    data_test["FractionalError"] = data_test.apply(
        lambda row: (row.PowerPrediction - row.TruePower) / row.TruePower, axis=1
    )
    data_test["FractionalErrorBaseline"] = data_test.apply(
        lambda row: (train_mean_power - row.TruePower) / row.TruePower, axis=1
    )
    print("Estimator MAE: {}\n".format(data_test.FractionalError.abs().mean()))
    print("Baseline MAE: {}\n".format(data_test.FractionalErrorBaseline.abs().mean()))
    print("Training Set:")
    data_train["PowerPrediction"] = model.predict(data_train)
    data_train["TruePower"] = target_train
    data_train["FractionalError"] = data_train.apply(
        lambda row: (row.PowerPrediction - row.TruePower) / row.TruePower, axis=1
    )
    data_test["FractionalErrorBaseline"] = data_test.apply(
        lambda row: (train_mean_power - row.TruePower) / row.TruePower, axis=1
    )
    print("Estimator MAE: {}\n".format(data_train.FractionalError.abs().mean()))
    print("Baseline MAE: {}\n".format(data_train.FractionalErrorBaseline.abs().mean()))







def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=str)

    datacols_group = parser.add_mutually_exclusive_group()
    datacols_group.add_argument(
        "--queue_data", action="store_true",
        help="Use only job submission data"
    )
    datacols_group.add_argument(
        "--queue_and_run_data", action="store_true",
        help="Use job submission and completion data"
    )

    parser.add_argument(
        "--cache", type=str, default="",
        help="How to use the cache (save|load)"
    )

    return parser.parse_args()

if __name__ == "__main__":
    pd.options.display.float_format = "{:.3f}".format

    main(parse_arguments())

