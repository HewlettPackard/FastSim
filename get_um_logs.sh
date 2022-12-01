#!/bin/bash

JOB_NAMES=("u-ch330.atmos_main" "u-ck777.atmos_main" "u-ck778.atmos_main")
OUTPUT_FORMAT=JobID,JobName,ReqNodes,ReqCPUS,ReqMem,Partition,QOS,TotalCPU,CPUTime,AveDiskRead,AveDiskWrite,AveRSS,AveVMSize,AveCPU,AvePages,MaxDiskRead,MaxDiskWrite,MaxRSS,MaxVMSize,MaxPages,NTasks,State,ExitCode,Flags,Elapsed,NodeList,AllocNodes,ConsumedEnergyRaw

out_file=$1

echo $OUTPUT_FORMAT | sed -e "s/\,/\|/g" -e "s/$/\|/" > $out_file

for job_name in ${JOB_NAMES[@]}; do
  for jobid in `sacct -ap -u n1280run -S 2022-01-01 --noconvert -o JobID,JobName \
                | grep "$job_name" | cut -d "|" -f 1`; do
    sacct -ap -j $jobid --noconvert --noheader -o $OUTPUT_FORMAT >> $out_file
  done
done

