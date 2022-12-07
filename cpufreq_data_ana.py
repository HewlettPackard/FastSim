import os
from glob import glob

import pandas as pd
from matplotlib import pyplot as plt

CPUFREQ_DIR = "/work/y02/y02/awilkins/archer2_jobdata/cpu_freq_tests"

for filepath in glob(os.path.join(CPUFREQ_DIR, "*perf_energy.csv")):
    df = pd.read_csv(filepath, delimiter=",", header=0)
    df["Application"] = os.path.basename(filepath).split("_")[0]

    try:
        df_cpufreq = pd.concat([df_cpufreq, df])
    except:
        df_cpufreq = df

print(df_cpufreq)

# for app in df_cpufreq.Application.unique():
#     df_slice = df_cpufreq.loc[(df_cpufreq.Application == app)]
#     df_slice.plot.hist(column=["Time"], by=["Freq", "Application"], bins=round(len(df_slice) / 15))
# plt.show()

# for app in df_cpufreq.Application.unique():
#     df_slice = df_cpufreq.loc[(df_cpufreq.Application == app)]
#     df_slice.plot.hist(column=["Energy"], by=["Freq", "Application"], bins=round(len(df_slice) / 15))
# plt.show()

df_cpufreq = df_cpufreq.reset_index(drop=True)
df_cpufreq.plot.hist(column=["Energy"], by=["Freq"], bins=round(len(df_cpufreq) / 15) * 5, xlim=(min(df_cpufreq.Energy) - 0.1 * max(df_cpufreq.Energy), 1.1 * max(df_cpufreq.Energy)))
plt.show()
df_cpufreq.plot.hist(column=["Time"], by=["Freq"], bins=round(len(df_cpufreq) / 15) * 5, xlim=(min(df_cpufreq.Time) - 0.1 * max(df_cpufreq.Time), 1.1 * max(df_cpufreq.Time)))
plt.show()
