#!/bin/bash
# Collect FastSim inputs using only NORMAL USER privileges (no operator/admin).
#
# Collects everything an unprivileged account can typically read:
#   - slurm.conf
#   - QOS (sacctmgr list qos)
#   - node events (sacctmgr show events)
#   - current reservations (sinfo --reservation)
#   - historic reservations (sreport reservation utilization)
#   - associations with REAL shares, reconstructed from sshare -aP
#     (sacctmgr show association is often restricted to your own rows,
#      but sshare -a exposes the full fairshare tree)
#
# NOT collected: the all-users job trace. sacct -a is typically restricted
# (PrivateData=jobs); obtain the trace from your site's job-history database
# instead. See README "Input Data" for the required columns.
#
# Run scripts/check_dump_access.sh first to confirm what your account can see.

STARTTIME=2025-10-01T00:00:00
ENDTIME=2026-01-01T00:00:00
SLURM_CONF_PATH=${SLURM_CONF:-/nopt/slurm/etc/slurm.conf}

mkdir -p slurm_dump

echo "Dumping QOS."
sacctmgr -p list qos format=Name,Priority,GrpTRES,GrpJobs,GrpSubmit,MaxTRESPU,MaxJobsPU,MaxJobs,MaxSubmitPU,MaxSubmit > slurm_dump/sacctmgr_qos.csv

echo "Dumping node events."
sacctmgr -p show events Start=$STARTTIME End=$ENDTIME > slurm_dump/sacctmgr_events.csv

echo "Dumping current reservations."
sinfo --reservation | sed 's/  */|/g' > slurm_dump/sinfo_resv.csv

echo "Dumping historic reservations."
{
    echo "Name|Nodes|Start|End|"
    sreport reservation Utilization -p -n start=$STARTTIME end=$ENDTIME format=Name,Nodes,Start,End
} > slurm_dump/sreport_resv.csv

echo "Reconstructing associations from sshare."
# sshare -aP: Account|User|RawShares|NormShares|RawUsage|EffectvUsage|FairShare
# The Account field is indented with one leading space per tree depth, which
# lets us recover ParentName for account rows. User rows reference their
# account by name, matching how FairTree reads sacctmgr_assocs.csv.
sshare -aP --format=Account,User,RawShares | awk -F'|' '
    BEGIN { print "Account|User|ParentName|Partition|Shares" }
    NR == 1 { next }                       # header
    {
        account = $1
        depth = 0
        while (substr(account, 1, 1) == " ") { depth++; account = substr(account, 2) }
        user = $2
        shares = ($3 ~ /^[0-9]+$/) ? $3 : 1   # "parent" or empty -> 1

        if (user == "") {                  # account (or root) row
            if (account == "root") { parent_at[0] = "root"; print "root||||1"; next }
            print account "||" parent_at[depth-1] "||" shares
            parent_at[depth] = account
        } else {                           # user association row
            print account "|" user "|||" shares
        }
    }
' > slurm_dump/sacctmgr_assocs.csv

echo "Copying slurm.conf from $SLURM_CONF_PATH."
cp "$SLURM_CONF_PATH" slurm_dump/slurm.conf

echo "Done. Place your job trace (from the job-history database) at"
echo "slurm_dump/sacct_jobs.csv, or point job_dump in the YAML config at it."
