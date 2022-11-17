"""
Explore feature importance for application power usage via decision tress
"""

import argparse, os, sys, joblib

import pandas as pd
import numpy as np
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance, PartialDependenceDisplay, partial_dependence

import xgboost as xgb
from xgboost import plot_tree

from funcs import parse_cache, timelimit_str_to_timedelta, hour_to_timeofday


POWER_COLS = ["JobID", "Start", "End", "ConsumedEnergyRaw", "AllocNodes"]
# Not going to try categorising JobName and SubmitLine
SUBMIT_COLS = ["ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem", "Timelimit", "Submit"]
FINISH_COLS = ["Elapsed", "ExitCode", "NTasks", "TotalCPU", "CPUTime", "MaxRSS", "MaxVMSize",
               "AvePages", "AveDiskRead", "AveDiskWrite"]

HPARAM_DIR = "/work/y02/y02/awilkins/archer2_jobdata/hparams"
MODELS_DIR = "/work/y02/y02/awilkins/archer2_jobdata/models"
CACHE_DIR = "/work/y02/y02/awilkins/pandas_cache"
PLOT_DIR = "/work/y02/y02/awilkins/archer2_jobdata/plots"

def clean_df(df, queue_only=False):
    cols_to_str = ["JobID", "ReqMem", "ReqCPUS", "ReqNodes", "AllocNodes"]
    df[cols_to_str] = df[cols_to_str].astype(str)

    df_jobs = df.loc[(~df.JobID.str.contains("\."))]

    # Remove nans first to prevent illegal operations in next slicing
    df_jobs = df_jobs.loc[
        ((df.Group.notna()) & (df.ReqMem.notna()) & (df.QOS.notna()) & (df.Timelimit.notna()))
    ]

    df_jobs = df_jobs.loc[(
        (~df.ReqMem.str.contains("\?")) & (df.Timelimit != "Partition_Limit") &
        (df.Timelimit != "UNLIMITED")
    )]

    cols = ["ReqCPUS", "ReqNodes", "ReqMem", "AllocNodes"]
    df_jobs[cols] = df_jobs[cols].replace(
        { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    ).astype(float).astype(int)

    df_jobs.Submit = pd.to_datetime(df_jobs.Submit, format="%Y-%m-%dT%H:%M:%S")
    df_jobs.Submit = df_jobs.Submit.apply(lambda row: hour_to_timeofday(row.hour))

    df_jobs.Timelimit = df_jobs.Timelimit.apply(
        lambda row: round(timelimit_str_to_timedelta(row).total_seconds() / 60)
    )

    df_jobs["PowerPerNode"] = df_jobs.apply(lambda row: float(row.Power) / float(row.AllocNodes), axis=1)

    # NOTE there are a few jobs with >2kW power per node (36 in the case I checked),
    # this seems unrealistic for a 128 core node. Just going to leave them for now, the model
    # is learning to ignore them anyway due to how infrequent they are
    # print(len(df_jobs.loc[(df_jobs.PowerPerNode >= 2000)]))
    # print(df_jobs.loc[(data_test.PowerPerNode >= 2000)])
    # sys.exit()

    if queue_only:
        return df_jobs

    cols_to_str = ["MaxRSS", "MaxVMSize", "AvePages", "NTasks", "AveDiskWrite", "AveDiskRead"]
    df[cols_to_str] = df[cols_to_str].astype(str)

    df_jobs = df_jobs.loc[(df_jobs.TotalCPU.notna())]

    df_jobs.Elapsed = df_jobs.Elapsed.apply(
        lambda row: round(timelimit_str_to_timedelta(row).total_seconds())
    )
    df_jobs.TotalCPU = df_jobs.TotalCPU.apply(
        lambda row: round(timelimit_str_to_timedelta(row).total_seconds())
    )
    df_jobs.CPUTime = df_jobs.CPUTime.apply(
        lambda row: round(timelimit_str_to_timedelta(row).total_seconds())
    )
    df_jobs["CPUUtil"] = df_jobs.apply(lambda row: row.TotalCPU/row.CPUTime, axis=1)

    df_jobs.ExitCode = df_jobs.ExitCode.apply(lambda row: 0 if row == "0:0" else 1)

    df_steps = df.loc[(df.JobID.str.contains("\."))]
    df_steps["ParentJobID"] = df_steps.JobID.apply(lambda row: row.split(".")[0])

    jobids = df_jobs.JobID.unique()
    l0 = len(df_steps)
    df_steps = df_steps.loc[(df_steps.ParentJobID.isin(jobids))]
    print("{} job steps missing parent jobs removed".format(l0 - len(df_steps)))

    cols = ["MaxRSS", "MaxVMSize", "AvePages", "NTasks", "AveDiskWrite", "AveDiskRead",
            "AllocNodes"]
    df_steps[cols] = df_steps[cols].replace(
        { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    ).astype(float).astype(int)

    df_jobs["MaxRSSPerNode"] = np.nan
    df_jobs["MaxVMSizePerNode"] = np.nan
    df_jobs["DiskReadRatePerNode"] = np.nan
    df_jobs["DiskWriteRatePerNode"] = np.nan
    df_jobs["PageFaultRatePerNode"] = np.nan
    df_jobs["NumStepsPerNode"] = np.nan
    df_jobs["NumTasksPerNode"] = np.nan

    df_steps["MaxRSSPerNode"] = df_steps.apply(
        lambda row: float(row.MaxRSS * row.NTasks) / float(row.AllocNodes), axis=1
    )
    df_steps["MaxVMSizePerNode"] = df_steps.apply(
        lambda row: float(row.MaxVMSize * row.NTasks) / float(row.AllocNodes), axis=1
    )
    df_steps["TotAveDiskRead"] = df_steps.apply(lambda row: row.AveDiskRead * row.NTasks, axis=1)
    df_steps["TotAveDiskWrite"] = df_steps.apply(lambda row: row.AveDiskWrite * row.NTasks, axis=1)
    df_steps["TotAvePageFault"] = df_steps.apply(lambda row: row.AvePages * row.NTasks, axis=1)

    for jobid in tqdm(jobids, desc="Aggregating data from job steps..."):
        df_steps_slice = df_steps.loc[(df_steps.ParentJobID == jobid)]
        max_rss_pernode = df_steps_slice.MaxRSSPerNode.max()
        max_vmsize_pernode = df_steps_slice.MaxVMSizePerNode.max()

        df_jobs_slice = df_jobs.loc[(df_jobs.JobID == jobid)]
        elapsed = float(df_jobs_slice.Elapsed.iloc[0])
        allocnodes = float(df_jobs_slice.AllocNodes.iloc[0])
        diskread_rate_pernode = float(df_steps_slice.TotAveDiskRead.sum()) / elapsed / allocnodes
        diskwrite_rate_pernode = float(df_steps_slice.TotAveDiskWrite.sum()) / elapsed / allocnodes
        pagefault_rate_pernode = float(df_steps_slice.TotAvePageFault.sum()) / elapsed / allocnodes

        numtasks_pernode = float(df_steps_slice.NTasks.sum()) / allocnodes

        df_jobs.loc[(df_jobs.JobID == jobid), "NumStepsPerNode"] = (float(len(df_steps_slice)) /
                                                                    allocnodes)

        df_jobs.loc[(df_jobs.JobID == jobid), "NumTasksPerNode"] = numtasks_pernode
        df_jobs.loc[(df_jobs.JobID == jobid), "MaxRSSPerNode"] = max_rss_pernode / 1e+9
        df_jobs.loc[(df_jobs.JobID == jobid), "MaxVMSizePerNode"] = max_vmsize_pernode / 1e+9
        df_jobs.loc[(df_jobs.JobID == jobid), "DiskReadRatePerNode"] = (diskread_rate_pernode /
                                                                        1e+6)
        df_jobs.loc[(df_jobs.JobID == jobid), "DiskWriteRatePerNode"] = (diskwrite_rate_pernode /
                                                                         1e+6)
        df_jobs.loc[(df_jobs.JobID == jobid), "PageFaultRatePerNode"] = pagefault_rate_pernode

    df_jobs.Elapsed = df_jobs.Elapsed.apply(lambda row: round(row / 60))

    return df_jobs


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


def print_predictions(
    model, queue_only, data_test, target_test, data_train=None, target_train=None
    ):
    data_test = data_test.copy()
    print("Test Set:")
    data_test["PowerPrediction"] = model.predict(data_test)
    data_test["TruePower"] = target_test
    data_test["FractionalError"] = data_test.apply(
        lambda row: (row.PowerPrediction - row.TruePower) / row.TruePower, axis=1
    )
    data_test["Within20W"] = data_test.apply(
        lambda row: 1 if abs(row.PowerPrediction - row.TruePower) < 20 else 0, axis=1
    )
    data_test["Within50W"] = data_test.apply(
        lambda row: 1 if abs(row.PowerPrediction - row.TruePower) < 50 else 0, axis=1
    )
    data_test["Within100W"] = data_test.apply(
        lambda row: 1 if abs(row.PowerPrediction - row.TruePower) < 100 else 0, axis=1
    )
    print(data_test)
    # with pd.option_context('display.max_rows', 500, 'display.max_columns', None):
    #     print(data_test[["ReqCPUS", "ReqNodes", "TruePower", "PowerPrediction"]][100:600])
    print("Estimator MAE fractional = {}".format(data_test.FractionalError.abs().mean()))
    print("Estimator within 20W accuracy = {}".format(data_test.Within20W.mean()))
    print("Estimator within 50W accuracy = {}".format(data_test.Within50W.mean()))
    print("Estimator within 100W accuracy = {}".format(data_test.Within100W.mean()))
    print("Estimator Coeff of Determination = {}\n".format(model.score(data_test, target_test)))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    data_test.hist(
        column="FractionalError", ax=ax, grid=False, bins=np.arange(-1.0, 1.025, 0.025),
        histtype="step", linewidth=2
    )
    ax.set_title("Fractional Error of Job Power per Node Prediction", fontsize=18)
    ax.set_ylabel("# Jobs")
    ax.set_xlabel("(PPred - PTrue) / PTrue", fontsize=16)
    ax.set_xlim(left=-1.0, right=1.0)
    ax.tick_params(axis='x', which='major', labelsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "predpower_fractionalerr_{}.pdf".format(
        "queue" if queue_only else "queue_run"
    )))
    plt.show()

    # print(data_test.TruePower.min(), data_test.TruePower.max())
    # print(data_test.PowerPrediction.min(), data_test.PowerPrediction.max())
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    data_test.hist(
        column="TruePower", ax=ax, grid=False, bins=np.arange(0, 710, 10), histtype="step",
        figsize=(12, 8), label="True", linewidth=2
    )
    data_test.hist(
        column="PowerPrediction", ax=ax, grid=False, bins=np.arange(0, 710, 10), histtype="step",
        figsize=(12, 8), label="Predicted", linewidth=2
    )
    ax.set_title("Predicted Distribution of Job Power per Nodes", fontsize=18)
    ax.set_xlabel("Power per Node", fontsize=16)
    ax.set_ylabel("# Jobs")
    ax.set_xlim(left=50, right=700)
    ax.tick_params(axis='x', which='major', labelsize=14)
    plt.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "predpower_distribution_{}.pdf".format(
        "queue" if queue_only else "queue_run"
    )))
    plt.show()

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
    if args.cache == "load_cleaned":
        df_power = pd.read_pickle(os.path.join(CACHE_DIR, "{}/{}.pkl".format(
            ".".join(os.path.basename(args.data).split(".")[:-1]),
            "decision_tree_queue_df_cleaned" if args.queue_data
                                             else "decision_tree_queue_run_df_cleaned"
        )))
    else:
        df_power = parse_cache(
            args.data,
            args.cache,
            ".".join(os.path.basename(args.data).split(".")[:-1]),
            "decision_tree_queue_df" if args.queue_data else "decision_tree_queue_run_df",
            cols=POWER_COLS+SUBMIT_COLS if args.queue_data else POWER_COLS+SUBMIT_COLS+FINISH_COLS,
            remove_steps=args.queue_data
        )

        pd.options.mode.chained_assignment = None
        df_power = clean_df(df_power, queue_only=args.queue_data)
        pd.options.mode.chained_assignment = None

        df_power = df_power.drop(
            ["JobID", "Start", "End", "ConsumedEnergyRaw", "DeltaT", "Power"], axis=1
        )
        if args.queue_data:
            df_power = df_power.drop(["AllocNodes"], axis=1)
        else:
            df_power = df_power.drop(
                ["NTasks", "TotalCPU", "CPUTime", "MaxRSS", "MaxVMSize", "AvePages", "AveDiskRead",
                 "AveDiskWrite"],
                axis=1
            )

    if args.cache == "save_cleaned":
        df_power.to_pickle(os.path.join(CACHE_DIR, "{}/{}.pkl".format(
            ".".join(os.path.basename(args.data).split(".")[:-1]),
            "decision_tree_queue_df_cleaned" if args.queue_data
                                             else "decision_tree_queue_run_df_cleaned"
        )))

    target = df_power.PowerPerNode
    data = df_power.drop(["PowerPerNode"], axis=1)

    numerical_columns = ["ReqCPUS", "ReqNodes", "ReqMem", "Timelimit"]
    categorical_columns = ["Group", "QOS", "Submit"]
    if args.queue_and_run_data:
        numerical_columns += [
            "AllocNodes", "CPUUtil", "ExitCode", "Elapsed", "MaxRSSPerNode", "MaxVMSizePerNode",
            "DiskReadRatePerNode", "DiskWriteRatePerNode", "PageFaultRatePerNode",
            "NumStepsPerNode", "NumTasksPerNode"
        ]

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
        model = joblib.load(
            os.path.join(MODELS_DIR, "{}_xgboostpipeline.joblib".format(save_prefix))
        )
        xgboost_model = joblib.load(
            os.path.join(MODELS_DIR, "{}_xgboostmodel.joblib".format(save_prefix))
        )

    else:
        if args.hyperparams == "search":
            if args.queue_data:
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
                #     'reg_alpha' : [0.0, 0.1], 'reg_lambda' : [0.9, 1.0, 1.1],
                #     'gamma' : [0.0, 0.5], 'min_child_weight' : [0.5, 1.0, 1.5]
                # }
                # params = {
                #     'n_estimators' : [200] , 'max_depth' : [12], 'learning_rate' : [0.3],
                #     'reg_alpha' : [0.0, 0.05], 'reg_lambda' : [0.7, 0.8, 0.9],
                #     'gamma' : [0.25, 0.5, 0.75], 'min_child_weight' : [0.25, 0.5, 0.75]
                # }
                # params = {
                #     'n_estimators' : [200] , 'max_depth' : [12], 'learning_rate' : [0.3],
                #     'reg_alpha' : [0.0], 'reg_lambda' : [0.1, 0.4, 0.5, 0.6, 0.7],
                #     'gamma' : [0.75, 0.85], 'min_child_weight' : [0.1, 0.25, 0.4]
                # }
                # params = {
                #     'n_estimators' : [200] , 'max_depth' : [12], 'learning_rate' : [0.3],
                #     'reg_alpha' : [0.0], 'reg_lambda' : [0.7], 'gamma' : [0.85, 0.9, 0.95],
                #     'min_child_weight' : [0.25, 0.5, 0.75, 0.1]
                # }
                # params = {
                #     'n_estimators' : [200] , 'max_depth' : [12], 'learning_rate' : [0.3],
                #     'reg_alpha' : [0.0], 'reg_lambda' : [0.7], 'gamma' : [0.9],
                #     'min_child_weight' : [0.0, 0.0005, 0.001]
                # }
                # params = {
                #     'n_estimators' : [150, 200, 250] , 'max_depth' : [10, 12, 14],
                #     'learning_rate' : [0.2, 0.3, 0.4], 'reg_alpha' : [0.0], 'reg_lambda' : [0.7],
                #     'gamma' : [0.9], 'min_child_weight' : [0.0]
                # }
                params = {
                    'n_estimators' : [200] , 'max_depth' : [12], 'learning_rate' : [0.3],
                    'reg_alpha' : [0.0, 0.05], 'reg_lambda' : [0.6, 0.7, 0.8],
                    'gamma' : [0.85, 0.9, 0.95], 'min_child_weight' : [0.0, 0.05]
                }

            else:
                # Semi-manual grid search history for queue and run data
                # params = {
                #     'n_estimators' : [150, 200, 250] , 'max_depth' : [8, 12, 14],
                #     'learning_rate' : [0.2, 0.30, 0.4], 'reg_alpha' : [0.0, 0.1],
                #     'reg_lambda' : [0.6, 0.7, 0.8], 'gamma' : [0.8, 0.9, 1.0],
                #     'min_child_weight' : [0.0, 0.001]
                # }
                # params = {
                #     'n_estimators' : [125, 150, 175] , 'max_depth' : [13, 14, 16],
                #     'learning_rate' : [0.15, 0.2, 0.25], 'reg_alpha' : [0.0],
                #     'reg_lambda' : [0.5, 0.6], 'gamma' : [0.7, 0.8], 'min_child_weight' : [0.0]
                # }
                # params = {
                #     'n_estimators' : [75, 100, 125] , 'max_depth' : [9, 11, 13],
                #     'learning_rate' : [0.05, 0.1, 0.15], 'reg_alpha' : [0.0],
                #     'reg_lambda' : [0.6], 'gamma' : [0.8], 'min_child_weight' : [0.0]
                # }
                params = {
                    'n_estimators' : [115, 125] , 'max_depth' : [12, 13],
                    'learning_rate' : [0.025, 0.05, 0.075], 'reg_alpha' : [0.0],
                    'reg_lambda' : [0.6], 'gamma' : [0.8], 'min_child_weight' : [0.0]
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
            model, os.path.join(MODELS_DIR, "{}_xgboostsipeline.joblib".format(save_prefix))
        )
        joblib.dump(
            xgboost_model, os.path.join(MODELS_DIR, "{}_xgboostmodel.joblib".format(save_prefix))
        )

    if args.print_predictions:
        print_predictions(model, args.queue_data, data_test, target_test)

    # Requires graphviz python package and exe to work - copy model to laptop and do this
    # if args.visualise_tree:
    #     print("Trying to plot a tree from xgboost...")
    #     plot_tree(model.named_steps['xgbregressor'], num_trees=1)
    #     plt.gcf().set_size_inches(18.5, 10.5)
    #     plt.show()

    if args.feature_importance:
        imp_multi = permutation_importance(
            model, data_test, target_test, n_repeats=30, random_state=1,
            scoring=["r2", "neg_mean_absolute_percentage_error", "neg_mean_squared_error"]
        )

        for metric in imp_multi:
            imp = imp_multi[metric]

            imp_sorted_idx = imp.importances_mean.argsort()
            imp_sorted_idx, idx_truncated = ((imp_sorted_idx, False) if len(imp_sorted_idx) <= 10
                                             else (imp_sorted_idx[-10:], True))

            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            ax.boxplot(
                imp.importances[imp_sorted_idx].T, vert=False,
                labels=data_test.columns[imp_sorted_idx]
            )
            ax.set_xlabel("Feature Importance (loss={})".format(metric), fontsize=18)
            ax.tick_params(axis='y', which='major', labelsize=16)
            plt.title("Permutation Importance{}".format(" (top 10)" if idx_truncated else ""))
            fig.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "feature_importance_{}_{}.pdf".format(
                "queue" if args.queue_data else "queue_run", metric
            )))
            plt.show()

    if args.partial_dependence:
        if args.queue_data:
            # Need to plot manually for categorical features. Doesn't look like there is an elegant
            # way to make PDP for one-hot encoded categorical varialbes
            categorical_preprocessor = OneHotEncoder(handle_unknown="ignore", sparse=False)
            numerical_preprocessor = StandardScaler()
            encoder= ColumnTransformer([
                ('one-hot-encoder', categorical_preprocessor, categorical_columns),
                ('standard_scaler', numerical_preprocessor, numerical_columns)
            ])
            group_to_num = { group : num for num, group in enumerate(data_train.Group.unique()) }
            num_to_group = { num : group for group, num in group_to_num.items() }
            data_train_groupnums = data_train.copy()
            data_train_groupnums.Group = data_train_groupnums.Group.apply(
                lambda row: group_to_num[row]
            )
            encoder.fit(data_train_groupnums)
            model_groupnums = make_pipeline(encoder, xgboost_model)
            partialdep_data = partial_dependence(
                model_groupnums, data_train_groupnums, "Group", kind="both",
                percentiles=(0.0, 1.0), grid_resolution=len(data_train_groupnums.Group.unique())
            )


            x = np.array(list(range(max(num_to_group) + 1)))
            height = partialdep_data['average'][0]
            yerr = np.std(partialdep_data["individual"][0], axis=0)
            sorted_idx = height.argsort()

            print(sorted_idx)
            print(num_to_group)

            # Copied from stackoverflow on how to broken axis (pain)
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 8))
            ax.bar(x, height[sorted_idx], yerr=yerr[sorted_idx])
            ax2.bar(x, height[sorted_idx], yerr=yerr[sorted_idx])
            ax.set_xticks(x)
            ax.set_xticklabels(
                np.array(
                    [ num_to_group[num] for num in range(max(num_to_group) + 1) ]
                )[sorted_idx],
                rotation=45
            )
            ax2.set_xticks(x)
            ax2.set_xticklabels(
                np.array(
                    [ num_to_group[num] for num in range(max(num_to_group) + 1) ]
                )[sorted_idx],
                rotation=45
            )
            ax.set_xlim(-0.5, 5.6)
            ax2.set_xlim(max(x) - 5.6, max(x) + 0.5)
            ax.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax.set_yticklabels([])
            ax.yaxis.set_label_position("left")
            ax.yaxis.tick_left()
            ax.tick_params(labelright='off')
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
            ax.grid(axis='y')
            ax2.grid(axis='y')
            d = 0.015
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
            ax.plot((1-d,1+d), (-d,+d), **kwargs)
            ax.plot((1-d,1+d),(1-d,1+d), **kwargs)
            kwargs.update(transform=ax2.transAxes)
            ax2.plot((-d,+d), (1-d,1+d), **kwargs)
            ax2.plot((-d,+d), (-d,+d), **kwargs)
            ax2.set_ylabel("Power (W)", fontsize=16)
            fig.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "partialdependence_queue_Group_brokenax.pdf"))
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            plt.bar(x, height[sorted_idx], yerr=yerr[sorted_idx])
            ax.set_xticks(x)
            ax.set_xticklabels(np.array(list(range(max(num_to_group) + 1))), fontsize=6)
            ax.set_xlim(left=-1, right=max(x) + 1)
            ax.set_xlabel("Group", fontsize=16)
            ax.set_ylabel("Power (W)", fontsize=16)
            plt.title("Partial Dependence of Power per Node on Group", fontsize=16)
            fig.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "partialdependence_queue_Group_all.pdf"))
            plt.show()

            print("max Timelimit={}, max Timelimit={}".format(
                data_train.Timelimit.min(), data_train.Timelimit.max()
            ))
            data_train_boundtimelimit = data_train.loc[(data_train.Timelimit <= 2000)]
            max_timelimit = data_train_boundtimelimit.Timelimit.max()
            print("Fraction of data with Timelimit<={} is {}".format(
                max_timelimit, len(data_train_boundtimelimit) / len(data_train)
            ))
            PartialDependenceDisplay.from_estimator(
                model, data_train_boundtimelimit, ["Timelimit"], kind="both", n_jobs=4,
                grid_resolution=max_timelimit, percentiles=(0.0, 1.0), verbose=1
            )
            for x, label in (
                { 360 : "6hrs", 720 : "12hrs", 1080 : "16hrs", 1440 : "1 day" }.items()
            ):
                plt.axvline(x, linestyle='dashed', c='r', linewidth=0.5)
                plt.text(x + 0.1, 750, label, rotation=90, c='r')
            fig = plt.gcf()
            fig.set_size_inches(14, 8)
            plt.ylabel("Partial Dependence (Power W)", fontsize=16)
            plt.xlabel("TimeLimit (minutes)", fontsize=16)
            plt.xlim(left=-0.01*max_timelimit, right=1.01*max_timelimit)
            plt.title("Partial Dependence of Power Per Node on Timelimit", fontsize=16)
            fig.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "partialdependence_queue_Timelimit.pdf"))
            plt.show()

            print("max ReqCPUS={}, max ReqCPUS={}".format(
                data_train.ReqCPUS.min(), data_train.ReqCPUS.max()
            ))
            data_train_boundreqcpu = data_train.loc[(data_train.ReqCPUS <= 2048)]
            max_reqcpus = data_train_boundreqcpu.ReqCPUS.max()
            print("Fraction of data with ReqCPUS<={} is {}".format(
                max_reqcpus, len(data_train_boundreqcpu) / len(data_train)
            ))
            PartialDependenceDisplay.from_estimator(
                model, data_train_boundreqcpu, ["ReqCPUS"], kind="both", n_jobs=4,
                grid_resolution=max_reqcpus, percentiles=(0.0, 1.0), verbose=1
            )
            for x in range(128, max_reqcpus, 128):
                plt.axvline(x, linestyle='dashed', c='r', linewidth=0.5)
                plt.text(x + 0.1, 750, str(x), rotation=90, c='r')
            fig = plt.gcf()
            fig.set_size_inches(14, 8)
            plt.ylabel("Partial Dependence (Power W)", fontsize=16)
            plt.xlabel("ReqCPUS", fontsize=16)
            plt.xlim(left=-0.01*max_reqcpus, right=1.01*max_reqcpus)
            plt.title("Partial Dependence of Power Per Node on ReqCPUS", fontsize=16)
            fig.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "partialdependence_queue_ReqCPUSmax{}.pdf".format(
                max_reqcpus
            )))
            plt.show()

        else:
            # print("min CPUUtil={}, max CPUUtil={}".format(
            #     data_train.CPUUtil.min(), data_train.CPUUtil.max()
            # ))
            # max_cpuutil = data_train.CPUUtil.max()
            # PartialDependenceDisplay.from_estimator(
            #     model, data_train, ["CPUUtil"], kind="both", n_jobs=4, grid_resolution=1000,
            #     percentiles=(0.00, 1.00), verbose=1
            # )
            # fig = plt.gcf()
            # fig.set_size_inches(14, 8)
            # plt.ylabel("Partial Dependence (Power W)", fontsize=16)
            # plt.xlabel("CPUUtil ([0,1])", fontsize=16)
            # plt.xlim(left=-0.01*max_cpuutil, right=1.01*max_cpuutil)
            # plt.ylim(bottom=100, top=800)
            # plt.title("Partial Dependence of Power Per Node on CPUUtil", fontsize=16)
            # fig.tight_layout()
            # plt.savefig(os.path.join(PLOT_DIR, "partialdependence_queue_run_CPUUtil.pdf"))
            # plt.show()

            # print("min NumTasksPerNode={}, max NumTasksPerNode={}".format(
            #     data_train.NumTasksPerNode.min(), data_train.NumTasksPerNode.max()
            # ))
            # data_train_boundnumtaskspernode = data_train.loc[(data_train.NumTasksPerNode <= 800)]
            # max_numtaskspernode = data_train_boundnumtaskspernode.NumTasksPerNode.max()
            # print("Fraction of data with NumTasksPerNode<={} is {}".format(
            #     max_numtaskspernode, len(data_train_boundnumtaskspernode) / len(data_train)
            # ))
            # PartialDependenceDisplay.from_estimator(
            #     model, data_train_boundnumtaskspernode, ["NumTasksPerNode"], kind="both", n_jobs=4,
            #     grid_resolution=800, percentiles=(0.0, 1.0), verbose=1
            # )
            # fig = plt.gcf()
            # fig.set_size_inches(14, 8)
            # plt.ylabel("Partial Dependence (Power W)", fontsize=16)
            # plt.xlabel("NumTasksPerNode", fontsize=16)
            # plt.xlim(left=-0.01*max_numtaskspernode, right=1.01*max_numtaskspernode)
            # plt.ylim(bottom=100, top=800)
            # plt.title("Partial Dependence of Power Per Node on NumTasksPerNode", fontsize=16)
            # fig.tight_layout()
            # plt.savefig(os.path.join(PLOT_DIR, "partialdependence_queue_run_NumTasksPerNode.pdf"))
            # plt.show()

            # print("min Elapsed={}, max Elapsed={}".format(
            #     data_train.Elapsed.min(), data_train.Elapsed.max()
            # ))
            # data_train_boundelapsed = data_train.loc[(data_train.Elapsed <= 500)]
            # max_elapsed = data_train_boundelapsed.Elapsed.max()
            # print("Fraction of data with Elapsed<={} is {}".format(
            #     max_elapsed, len(data_train_boundelapsed) / len(data_train)
            # ))
            # PartialDependenceDisplay.from_estimator(
            #     model, data_train_boundelapsed, ["Elapsed"], kind="both", n_jobs=4,
            #     grid_resolution=500, percentiles=(0.0, 1.0), verbose=1
            # )
            # fig = plt.gcf()
            # fig.set_size_inches(14, 8)
            # plt.ylabel("Partial Dependence (Power W)", fontsize=16)
            # plt.xlabel("Elapsed (minutes)", fontsize=16)
            # plt.title("Partial Dependence of Power Per Node on Elapsed", fontsize=16)
            # plt.xlim(left=-0.01*max_elapsed, right=1.01*max_elapsed)
            # plt.ylim(bottom=100, top=800)
            # fig.tight_layout()
            # plt.savefig(os.path.join(PLOT_DIR, "partialdependence_queue_run_Elapsed.pdf"))
            # plt.show()

            print("min MaxVMSizePerNode={}, max MaxVMSizePerNode={}".format(
                data_train.MaxVMSizePerNode.min(), data_train.MaxVMSizePerNode.max()
            ))
            data_train_boundmaxvmsizepernode = data_train.loc[(data_train.MaxVMSizePerNode <= 800)]
            max_elapsed = data_train_boundmaxvmsizepernode.MaxVMSizePerNode.max()
            print("Fraction of data with MaxVMSizePerNode<={} is {}".format(
                max_elapsed, len(data_train_boundmaxvmsizepernode) / len(data_train)
            ))
            PartialDependenceDisplay.from_estimator(
                model, data_train_boundmaxvmsizepernode, ["MaxVMSizePerNode"], kind="both",
                n_jobs=4, grid_resolution=400, percentiles=(0.0, 1.0), verbose=1
            )
            fig = plt.gcf()
            fig.set_size_inches(14, 8)
            plt.ylabel("Partial Dependence (Power W)", fontsize=16)
            plt.xlabel("MaxVMSizePerNode (GB)", fontsize=16)
            plt.title("Partial Dependence of Power Per Node on MaxVMSizePerNode", fontsize=16)
            plt.xlim(left=-0.01*max_elapsed, right=1.01*max_elapsed)
            plt.ylim(bottom=100, top=700)
            fig.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "partialdependence_queue_run_MaxVMSizePerNode.pdf"))
            plt.show()

            print("min MaxRSSPerNode={}, max MaxRSSPerNode={}".format(
                data_train.MaxRSSPerNode.min(), data_train.MaxRSSPerNode.max()
            ))
            data_train_boundmaxrsspernode = data_train.loc[(data_train.MaxRSSPerNode <= 800)]
            max_elapsed = data_train_boundmaxrsspernode.MaxRSSPerNode.max()
            print("Fraction of data with MaxRSSPerNode<={} is {}".format(
                max_elapsed, len(data_train_boundmaxrsspernode) / len(data_train)
            ))
            PartialDependenceDisplay.from_estimator(
                model, data_train_boundmaxrsspernode, ["MaxRSSPerNode"], kind="both", n_jobs=4,
                grid_resolution=400, percentiles=(0.0, 1.0), verbose=1
            )
            fig = plt.gcf()
            fig.set_size_inches(14, 8)
            plt.ylabel("Partial Dependence (Power W)", fontsize=16)
            plt.xlabel("MaxRSSPerNode (GB)", fontsize=16)
            plt.title("Partial Dependence of Power Per Node on MaxRSSPerNode", fontsize=16)
            plt.xlim(left=-0.01*max_elapsed, right=1.01*max_elapsed)
            plt.ylim(bottom=100, top=800)
            fig.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, "partialdependence_queue_run_MaxRSSPerNode.pdf"))
            plt.show()


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
        help="How to use the cache (save|load|load_cleaned)"
    )

    parser.add_argument(
        "--model_pkl", type=str, default="",
        help="What to do regarding a pickled model (save|load)"
    )

    parser.add_argument(
        "--hyperparams", type=str, default="",
        help="Search for (change params in code to specify search) or load hyperparams from file" +
             "(search|load)"
    )

    parser.add_argument(
        "--feature_importance", action="store_true",
        help="Compute and plot feature importance"
    )

    parser.add_argument(
        "--partial_dependence", action="store_true",
        help="Compute and plot partial dependences of interest (hardcoded)"
    )

    parser.add_argument(
        "--print_predictions", action="store_true",
        help="Print and summarise power predictions for all entires in test + training set"
    )

    args = parser.parse_args()

    if args.model_pkl != "load" and not args.hyperparams:
        print("NOTE: using default hyperparams")

    return parser.parse_args()

if __name__ == "__main__":
    pd.options.display.float_format = "{:.3f}".format

    main(parse_arguments())

