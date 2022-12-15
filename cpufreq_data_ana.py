import os, joblib
from glob import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity

CPUFREQ_DIR = "/work/y02/y02/awilkins/archer2_jobdata/cpu_freq_tests"
MODELS_DIR = "/work/y02/y02/awilkins/archer2_jobdata/models"

for filepath in glob(os.path.join(CPUFREQ_DIR, "*perf_energy.csv")):
    df = pd.read_csv(filepath, delimiter=",", header=0)
    df["Application"] = os.path.basename(filepath).split("_")[0]

    try:
        df_cpufreq = pd.concat([df_cpufreq, df])
    except:
        df_cpufreq = df

print(df_cpufreq)

# Plotting separately for each application
if False:
    for app in df_cpufreq.Application.unique():
        df_slice = df_cpufreq.loc[(df_cpufreq.Application == app)]
        df_slice.plot.hist(
            column=["Time"], by=["Freq", "Application"], bins=round(len(df_slice) / 15)
        )
    plt.show()

    for app in df_cpufreq.Application.unique():
        df_slice = df_cpufreq.loc[(df_cpufreq.Application == app)]
        df_slice.plot.hist(
            column=["Energy"], by=["Freq", "Application"], bins=round(len(df_slice) / 15)
        )
    plt.show()

# Plotting for all applications
if False:
    df_cpufreq = df_cpufreq.reset_index(drop=True)
    df_cpufreq.plot.hist(
        column=["Energy"], by=["Freq"], bins=round(len(df_cpufreq) / 15) * 5,
        xlim=(min(df_cpufreq.Energy) - 0.1 * max(df_cpufreq.Energy), 1.1 * max(df_cpufreq.Energy))
    )
    plt.show()
    df_cpufreq.plot.hist(
        column=["Time"], by=["Freq"], bins=round(len(df_cpufreq) / 15) * 5,
        xlim=(min(df_cpufreq.Time) - 0.1 * max(df_cpufreq.Time), 1.1 * max(df_cpufreq.Time))
    )
    plt.show()

# Pairing with 2.25GHz measurements to get slowdowns and power reductions (I thinks its ok to pair since the measurements are independent)
df_cpufreq["Power"] = df_cpufreq.apply(lambda row: row.Energy / (row.Time * 60), axis=1)
cpuresponse = {
    2000000 : { "Application" : [], "PowerFactor" : [], "TimeFactor" : [] },
    1500000 : { "Application" : [], "PowerFactor" : [], "TimeFactor" : [] }
}
for app in df_cpufreq.Application.unique():
    df_cpufreq_app = df_cpufreq.loc[(df_cpufreq.Application == app)]
    for freq in [1500000, 2000000]:
        row_high_low_zip = zip(
            df_cpufreq_app.loc[(df_cpufreq_app.Freq == 2250000)].iterrows(),
            df_cpufreq_app.loc[(df_cpufreq_app.Freq == freq)].iterrows()
        )
        for (_, row_highfreq), (_, row_lowfreq) in row_high_low_zip:
            cpuresponse[freq]["Application"].append(app)
            cpuresponse[freq]["PowerFactor"].append(row_lowfreq.Power / row_highfreq.Power)
            cpuresponse[freq]["TimeFactor"].append(row_lowfreq.Time / row_highfreq.Time)

df_cpuresponse_2 = pd.DataFrame.from_dict(cpuresponse[2000000])
df_cpuresponse_15 = pd.DataFrame.from_dict(cpuresponse[1500000])
print(df_cpuresponse_2)
print(df_cpuresponse_15)

cpuresponse_2_data = df_cpuresponse_2[["TimeFactor", "PowerFactor"]].to_numpy()
cpuresponse_15_data = df_cpuresponse_15[["TimeFactor", "PowerFactor"]].to_numpy()

if False:
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    h = ax.hist2d(
        cpuresponse_2_data[:,0], cpuresponse_2_data[:,1],
        bins=[np.linspace(1.0, 1.2, 21), np.linspace(0.65, 1.0, 36)], cmap='jet', cmin=1
    )
    ax.plot([1.0, 1.2], [1.0, 1 / 1.2], c='r')
    ax.annotate("Constant Energy", c='r', xy=(1.1, 0.93))
    ax.set_xlabel("time factor")
    ax.set_ylabel("power factor")
    ax.set_title("2GHz")
    fig.colorbar(h[3])
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    h = ax.hist2d(
        cpuresponse_15_data[:, 0], cpuresponse_15_data[:, 1],
        bins=[np.linspace(1.0, 1.6, 61), np.linspace(0.60, 1.0, 41)], cmap='jet', cmin=1
    )
    ax.plot([1.0, 1.6], [1.0, 1 / 1.6], c='r')
    p1, p2 = ax.transData.transform_point((1.0, 1.0)), ax.transData.transform_point((1.6, 1 / 1.6))
    rise, run = p2[1] - p1[1], p2[0] - p1[0]
    ax.annotate("Constant Energy", c='r', xy=(1.3, 0.83))
    ax.set_xlabel("time factor")
    ax.set_ylabel("power factor")
    ax.set_title("1.5GHz")
    fig.colorbar(h[3])
    fig.tight_layout()
    plt.show()

x, y = cpuresponse_2_data[:, 0], cpuresponse_2_data[:, 1]
# xx, yy, = np.mgrid[1.0:1.3:500j, 0.65:1.0:500j]
xx, yy, = np.mgrid[0.5:1.8:500j, 0.0:1.5:500j]
xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
xy_train = np.vstack([y, x]).T

# for kernel in ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]:
kde = KernelDensity(bandwidth=0.03, kernel="gaussian")
kde.fit(xy_train)

# score_samples() returns the log-likelihood of the samples
z = np.exp(kde.score_samples(xy_sample))
zz = np.reshape(z, xx.shape)

plt.pcolormesh(xx, yy, np.ma.masked_where(zz < 1e-4, zz))
plt.scatter(x, y, s=1, facecolor='white')
plt.show()

joblib.dump(kde, os.path.join(MODELS_DIR, "cpufreq2ghz_kde.joblib"))

x, y = cpuresponse_15_data[:, 0], cpuresponse_15_data[:, 1]
xx, yy, = np.mgrid[1.0:1.6:500j, 0.60:1.0:500j]
xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
xy_train = np.vstack([y, x]).T

# for kernel in ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]:
kde = KernelDensity(bandwidth=0.03, kernel="gaussian")
kde.fit(xy_train)

# score_samples() returns the log-likelihood of the samples
z = np.exp(kde.score_samples(xy_sample))
zz = np.reshape(z, xx.shape)

plt.pcolormesh(xx, yy, zz)
plt.scatter(x, y, s=1, facecolor='white')
plt.show()

joblib.dump(kde, os.path.join(MODELS_DIR, "cpufreq15ghz_kde.joblib"))

