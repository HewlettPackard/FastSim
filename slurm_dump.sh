#!/bin/bash

STARTTIME="2022-10-19"
ENDTIME="2023-02-19"

mkdir slurm_dump || exit

declare -A usr_anom

counter=0
for usr_line in $(
  sacctmgr -p show association withDeleted format=User | tail -n +2 | sort | uniq | sed 's/.$//'
);
do
  usr_anom[$usr_line]="User${counter}"
  counter=$((counter+1))
done

declare -A acc_anom

counter=0
for acc_line in $(
  sacctmgr -p show association withDeleted format=Account | tail -n +2 | sort | uniq | sed 's/.$//'
);
do
  if [ "$acc_line" = "root" ];
  then
    acc_anom["root"]="root"
  else
    acc_anom[$acc_line]="Acc${counter}"
    counter=$((counter+1))
  fi
done

sacctmgr -p show association withDeleted \
         format=User,Account,ParentName,Partition,MaxJobs,MaxSubmit |
tail -n +2 |
while read assoc_line;
do
  usr=$(echo "${assoc_line}" | cut -d '|' -f 1)
  if [ ! -z $usr ]; # If field is not set, leave it unset
  then
    usr=${usr_anom[$usr]}
  fi
  acc=$(echo "${assoc_line}" | cut -d '|' -f 2)
  if [ ! -z $acc ];
  then
    acc=${acc_anom[$acc]}
  fi
  parent=$(echo "${assoc_line}" | cut -d '|' -f 3)
  if [ ! -z $parent ];
  then
    parent=${acc_anom[$parent]}
  fi

  echo $assoc_line |
  awk -v usr=$usr -v acc=$acc -v parent=$parent -v OFS='|' -F '|' \
      '{$1=usr; $2=acc; $3=parent; print}' >> \
  slurm_dump/sacctmgr_assocs.csv
done 

sacct -ap --allocations --noconvert --starttime=$STARTTIME --endtime=$ENDTIME \
      --format=User,Account,AllocNodes,ConsumedEnergyRaw,ExitCode,Flags,JobID,JobName,Partition,QOS,Reason,ReqNodes,Start,State,End,Elapsed,Submit,SubmitLine,Timelimit |
tail -n +2 |
while read job_line;
do
  usr=$(echo "${job_line}" | cut -d '|' -f 1)
  if [ ! -z $usr ];
  then
    usr=${usr_anom[$usr]}
  fi
  acc=$(echo "${job_line}" | cut -d '|' -f 2)
  if [ ! -z $acc ];
  then
    acc=${acc_anom[$acc]}
  fi

  echo $job_line |
  awk -v usr=$usr -v acc=$acc -v OFS='|' -F '|' '{$1=usr; $2=acc; print}' >> \
  slurm_dump/sacct_jobs.csv
done 

sinfo --reservation | sed -r "s/ {1,}/|/g" | sed "s/$/|/" > slurm_dump/sinfo_resv.csv

sacctmgr -p show events Start=2022-10-19 End=2023-02-19 > slurm_dump/sacctmgr_events.csv

sacctmgr -p list qos \
         format=Name,Priority,GrpTRES,GrpJobs,GrpSubmit,GrpSubmit,MaxTRESPerUser,MaxJobsPU,MaxJobsPA,MaxSubmitPU,MaxSubmitPA,MaxSubmit,MaxJobs > \
slurm_dump/sacctmgr_qos.csv

cp /etc/slurm/slurm.conf slurm_dump/slurm.conf

tar -zcvf slurm_dump.tar.gz slurm_dump/

