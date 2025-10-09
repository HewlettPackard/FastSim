#!/bin/bash

STARTTIME="2025-01-01"
ENDTIME="2025-10-01"

mkdir slurm_dump || exit

sacctmgr -p show association withDeleted \
         format=User,Account,ParentName,Partition,Shares,MaxJobs,MaxSubmit > slurm_dump/sacctmgr_assocs.csv

sacct -ap --allocations --noconvert --starttime=$STARTTIME --endtime=$ENDTIME \
--format=User,Account,AllocNodes,ConsumedEnergyRaw,ExitCode,Flags,JobID,JobName,Partition,QOS,Reason,ReqNodes,Start,State,End,Elapsed,Submit,SubmitLine,Timelimit > slurm_dump/sacct_jobs.csv

sinfo --reservation | sed -r "s/ {1,}/|/g" | sed "s/$/|/" > slurm_dump/sinfo_resv.csv

sreport reservation Utilization -p -n start=$STARTTIME end=$ENDTIME format=Name,Nodes,Start,End > slurm_dump/sreport_resv.csv

sacctmgr -p show events Start=$STARTTIME End=$ENDTIME > slurm_dump/sacctmgr_events.csv

sacctmgr -p list qos \
         format=Name,Priority,GrpTRES,GrpJobs,GrpSubmit,GrpSubmit,MaxTRESPerUser,MaxJobsPU,MaxJobsPA,MaxSubmitPU,MaxSubmitPA,MaxSubmit,MaxJobs > \
slurm_dump/sacctmgr_qos.csv

cp /nopt/slurm/etc/slurm.conf slurm_dump/slurm.conf

tar -zcvf slurm_dump.tar.gz slurm_dump/