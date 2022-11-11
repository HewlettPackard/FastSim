"""
Explore feature importance for application power usage via decision tress
"""

import argparse, os, sys, joblib

import pandas as pd
import numpy as np
import yaml
from matplotlib import pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance

import xgboost as xgb
from xgboost import plot_tree

from funcs import parse_cache, timelimit_str_to_timedelta, hour_to_timeofday


POWER_COLS = ["JobID", "Start", "End", "ConsumedEnergyRaw", "AllocNodes"]
# Not going to try categorising JobName and SubmitLine
SUBMIT_COLS = ["ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem", "Timelimit", "Submit"]
FINISH_COLS = [] # TODO

HPARAM_DIR = "/work/y02/y02/awilkins/archer2_jobdata/hparams"
MODELS_DIR = "/work/y02/y02/awilkins/archer2_jobdata/models"

def clean_df(df, queue_only=False):
    pd.options.mode.chained_assignment = None

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

    pd.options.mode.chained_assignment = "warn"

    return df


def hyperparam_search(data, target, encoder, params, save_prefix):
    model = xgb.XGBRegressor(verbosity=0, n_jobs=4)

    # Hyperparam search on subset of training data (default test_size=0.25)
    data_train, data_test, target_train, target_test = train_test_split(
        encoder.fit_transform(data), target.to_numpy(), random_state=1, test_size=0.75
    )

    print("Searching hyperparams...")

    search = GridSearchCV(
        model, params, scoring='neg_mean_absolute_percentage_error', verbose=2, n_jobs=1
    )

    search.fit(data_train, target_train)

    print("Search results:")
    print(search.cv_results_)

    print("Best params:\n{}\nwith score {}".format(search.best_params_, search.best_score_))

    with open(os.path.join(HPARAM_DIR, "{}_bestparams.yml".format(save_prefix)), "w") as f:
        yaml.dump(search.best_params_, f)
    with open(os.path.join(HPARAM_DIR, "{}_searchresults.yml".format(save_prefix)), "w") as f:
        yaml.dump(search.cv_results_, f)

    return search.best_params_


def print_predictions(model, data_test, target_test, data_train=None, target_train=None):
    data_test = data_test.copy()
    print("Test Set:")
    data_test["PowerPrediction"] = model.predict(data_test)
    data_test["TruePower"] = target_test
    data_test["FractionalError"] = data_test.apply(
        lambda row: (row.PowerPrediction - row.TruePower) / row.TruePower, axis=1
    )
    print(data_test)
    print("Estimator MAE fractional = {}".format(data_test.FractionalError.abs().mean()))
    print("Estimator Coeff of Determination = {}\n".format(model.score(data_test, target_test)))

    if data_train is None:
        return

    data_train = data_train.copy()
    print("Training Set:")
    data_train["PowerPrediction"] = model.predict(data_train)
    data_train["TruePower"] = target_train
    data_train["FractionalError"] = data_train.apply(
        lambda row: (row.PowerPrediction - row.TruePower) / row.TruePower, axis=1
    )
    print(data_train)
    print("Estimator MAE fracional = {}".format(data_train.FractionalError.abs().mean()))
    print("Estimator Coeff of Determination = {}\n".format(model.score(data_train, target_train)))


def main(args):
    df_power = parse_cache(
        args.data,
        args.cache,
        ".".join(os.path.basename(args.data).split(".")[:-1]),
        "decision_tree_queue_df",
        cols=POWER_COLS+SUBMIT_COLS if args.queue_data else POWER_COLS+SUBMIT_COLS+FINISH_COLS
    )

    df_power = clean_df(df_power, queue_only=args.queue_data)

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

    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore", sparse=False)
    numerical_preprocessor = StandardScaler()

    encoder = ColumnTransformer([
        ('one-hot-encoder', categorical_preprocessor, categorical_columns),
        ('standard_scaler', numerical_preprocessor, numerical_columns)
    ])

    data_train, data_test, target_train, target_test = train_test_split(
        data, target, random_state=1
    )

    # data_onehot = encoder.fit_transform(data)
    # data_onehot_colnames = np.array(
    #     [ name.split("__")[1] for name in encoder.get_feature_names_out() ]
    # )
    # target = target.to_numpy()

    save_prefix ="queue" if args.queue_data else "queue-run"
    if args.model_pkl == "load":
        xgboost_model = joblib.load(
            os.path.join(MODELS_DIR, "{}_xgboostmodel.joblib".format(save_prefix))
        )

    else:
        if args.hyperparams == "search":
            # Semi-manual grid search history for queue-only data
            # params = {
            #     'n_estimators' : [50, 75, 100, 125, 150], 'max_depth' : [4, 6, 8, 10, 12],
            #     'learning_rate' : [0.2, 0.3, 0.4]
            # }
            # params = {
            #     'n_estimators' : [150, 175, 200], 'max_depth' : [12, 16, 20],
            #     'learning_rate' : [0.3]
            # }
            # params = {
            #     'n_estimators' : [200, 250, 300], 'max_depth' : [12], 'learning_rate' : [0.3]
            # }
            # params = {
            #     'n_estimators' : [200] , 'max_depth' : [12], 'learning_rate' : [0.3],
            #     'reg_alpha' : [0.0, 0.1], 'reg_lambda' : [0.9, 1.0, 1.1], 'gamma' : [0.0, 0.5],
            #     'min_child_weight' : [0.5, 1.0, 1.5]
            # }
            # params = {
            #     'n_estimators' : [200] , 'max_depth' : [12], 'learning_rate' : [0.3],
            #     'reg_alpha' : [0.0, 0.05], 'reg_lambda' : [0.7, 0.8, 0.9],
            #     'gamma' : [0.25, 0.5, 0.75], 'min_child_weight' : [0.25, 0.5, 0.75]
            # }
            params = {
                'n_estimators' : [200] , 'max_depth' : [12], 'learning_rate' : [0.3],
                'reg_alpha' : [0.0], 'reg_lambda' : [0.1, 0.4, 0.5, 0.6, 0.7],
                'gamma' : [0.75, 0.85], 'min_child_weight' : [0.1, 0.25, 0.4]
            }

            best_params = hyperparam_search(data, target, encoder, params, save_prefix)

        elif args.hyperparams == "load":
            with open(os.path.join(HPARAM_DIR, "{}_bestparams.yml".format(save_prefix)), "r") as f:
                best_params = yaml.safe_load(f)

        else:
            best_params = {}

        xgboost_model = xgb.XGBRegressor(verbosity=2, n_jobs=4, **best_params)

        model = make_pipeline(encoder, xgboost_model)

        model.fit(data_train, target_train)

    if args.model_pkl == "save":
        joblib.dump(
            xgboost_model, os.path.join( MODELS_DIR, "{}_xgboostmodel.joblib".format(save_prefix))
        )

    print_predictions(
        model, data_test, target_test, data_train=data_train, target_train=target_train
    )

    # Requires graphviz python package and exe to work - copy model to laptop and do this
    # if args.visualise_tree:
    #     print("Trying to plot a tree from xgboost...")
    #     plot_tree(model.named_steps['xgbregressor'], num_trees=1)
    #     plt.gcf().set_size_inches(18.5, 10.5)
    #     plt.show()

    if args.feature_importance:
        imp_multi = permutation_importance(
            model, data_test, target_test, n_repeats=30, random_state=1,
            scoring=["r2", "neg_mean_absolute_percentage_error"]
        )

        print(imp_multi)
        for metric in imp_multi:
            imp = imp_multi[metric]
            print("{}".format(metric))
            print("{}\n".format(imp))
            print("{}\n".format(type(imp)))


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

    parser.add_argument(
        "--model_pkl", type=str, default="",
        help="What to do regarding a pickled model (save|load)"
    )

    parser.add_argument(
        "--hyperparams", type=str, default="",
        help="Search for (change params in code to specify search) or load hyperparams from file (search|load)"
    )

    parser.add_argument(
        "--feature_importance", action="store_true",
        help="Compute and plot feature importance"
    )

    args = parser.parse_args()

    if not args.hyperparams:
        print("NOTE: using default hyperparams")

    return parser.parse_args()

if __name__ == "__main__":
    pd.options.display.float_format = "{:.3f}".format

    main(parse_arguments())

