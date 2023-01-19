from datetime import timedelta

global CABS_DIR
global PLOT_DIR
# global BASELINE_POWER
# global SLURMTOCAB_FACTOR
global NODEDOWN_MEAN
global BD_THRESHOLD
global MIN_STEP
global BACKFILL_OPTS
global KDE_MODEL_2GHZ
global ASSOCS_FILE

CABS_DIR = "/home/y02/shared/power"
PLOT_DIR = "/work/y02/y02/awilkins/archer2_jobdata/plots"

# Factors are from linear fit (see powerusage_cabs_against_slurm_shifted_grouped.pdf)
# node_down_mean is just from assuming todays sinfo -R is typical (there were also big partial
# shutdowns at the start of the slurm data I am not accounting for). I am failry sure that these
# depend on node occupancy so I am just going to ignore these for now
# BASELINE_POWER = 0 # kW (previously 1789 then 1692)
# SLURMTOCAB_FACTOR = 1.0 # (previously 0.517 then 0.578)
NODEDOWN_MEAN = 100 # 291
BD_THRESHOLD = timedelta(hours=1)
MIN_STEP = timedelta(seconds=10)

DEFER = True
SCHED_INTERVAL = timedelta(seconds=60)
SCHED_MIN_INTERVAL=timedelta(seconds=2)
PRIORITYCALCPERIOD = timedelta(minutes=5)

BACKFILL_OPTS = {
    "min_block_width" : timedelta(minutes=10), "max_job_test" : 1000 ,
    "window" : timedelta(minutes=5760), "interval" : timedelta(seconds=30)
}

KDE_MODEL_2GHZ = "/work/y02/y02/awilkins/archer2_jobdata/models/cpufreq2ghz_kde.joblib"

ASSOCS_FILE = "/work/y02/y02/awilkins/sacct_archer2_assocs_050123.csv"

NODE_EVENTS_FILE = "/work/y02/y02/awilkins/sacctmgr_events_221019-230104.csv"

