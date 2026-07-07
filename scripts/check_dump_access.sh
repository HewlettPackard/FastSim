#!/bin/bash
# Probe which FastSim inputs your current (unprivileged) SLURM account can
# actually obtain, and therefore which config keys you can populate.
# Read-only and cheap: every query is limited to a tiny window / few rows.
#
# Run on a login node:  bash check_dump_access.sh

PROBE_START=$(date -d "yesterday" +%Y-%m-%dT00:00:00 2>/dev/null || date -v-1d +%Y-%m-%dT00:00:00)
PROBE_END=$(date +%Y-%m-%dT00:00:00)

report () { # name, config key, status, detail
    printf "%-22s %-22s %-8s %s\n" "$1" "$2" "$3" "$4"
}

printf "%-22s %-22s %-8s %s\n" "INPUT" "CONFIG KEY" "STATUS" "DETAIL"
printf "%s\n" "--------------------------------------------------------------------------------"

# --- slurm.conf (required) ---
found_conf=""
for conf in "$SLURM_CONF" /nopt/slurm/etc/slurm.conf /etc/slurm/slurm.conf; do
    if [ -n "$conf" ] && [ -r "$conf" ]; then found_conf="$conf"; break; fi
done
if [ -n "$found_conf" ]; then
    report "slurm.conf" "slurm_conf" "OK" "$found_conf"
else
    report "slurm.conf" "slurm_conf" "MISSING" "not readable at common paths; ask admins or use scontrol write config"
fi

# --- job trace (required): can you see other users' jobs? ---
own_jobs=$(sacct -X -p --starttime="$PROBE_START" --endtime="$PROBE_END" --format=User 2>/dev/null | tail -n +2 | wc -l)
all_jobs=$(sacct -aX -p --starttime="$PROBE_START" --endtime="$PROBE_END" --format=User 2>/dev/null | tail -n +2 | wc -l)
other_users=$(sacct -aX -p --starttime="$PROBE_START" --endtime="$PROBE_END" --format=User 2>/dev/null | tail -n +2 | cut -d'|' -f1 | sort -u | grep -cv "^$(whoami)$")
if [ "$all_jobs" -gt 0 ] && [ "$other_users" -gt 0 ]; then
    report "job trace (all users)" "job_dump" "OK" "sacct -a returned $all_jobs jobs from $other_users other users (yesterday)"
elif [ "$all_jobs" -gt 0 ]; then
    report "job trace (all users)" "job_dump" "PARTIAL" "sacct works but only your own jobs are visible (PrivateData=jobs); a site job-history database is needed for a full trace"
else
    report "job trace (all users)" "job_dump" "DENIED" "sacct returned nothing; check slurmdbd access"
fi

# --- associations / shares ---
assoc_rows=$(sacctmgr -p show association format=User,Account,ParentName,Partition,Shares 2>/dev/null | tail -n +2 | wc -l)
assoc_users=$(sacctmgr -p show association format=User 2>/dev/null | tail -n +2 | sort -u | grep -cv '^|*$')
if [ "$assoc_rows" -gt 1 ] && [ "$assoc_users" -gt 1 ]; then
    report "associations" "assocs_dump" "OK" "$assoc_rows assoc rows visible ($assoc_users users)"
elif [ "$assoc_rows" -ge 1 ]; then
    report "associations" "assocs_dump" "PARTIAL" "only your own associations visible (PrivateData=users); omit assocs_dump and let FastSim synthesize"
else
    report "associations" "assocs_dump" "DENIED" "sacctmgr show association returned nothing"
fi

# --- QOS ---
qos_rows=$(sacctmgr -p list qos format=Name,Priority,GrpTRES,GrpJobs,GrpSubmit,MaxTRESPU,MaxJobsPU,MaxJobs,MaxSubmitPU,MaxSubmit 2>/dev/null | tail -n +2 | wc -l)
if [ "$qos_rows" -gt 0 ]; then
    report "QOS" "qos_dump" "OK" "$qos_rows QOS entries visible"
else
    report "QOS" "qos_dump" "DENIED" "sacctmgr list qos returned nothing"
fi

# --- node events ---
event_rows=$(sacctmgr -p show events Start="$PROBE_START" End="$PROBE_END" 2>/dev/null | tail -n +2 | wc -l)
if [ "$event_rows" -gt 0 ]; then
    report "node events" "node_events_dump" "OK" "$event_rows events yesterday"
else
    report "node events" "node_events_dump" "UNKNOWN" "no rows: either no events yesterday or PrivateData=events; try a wider window"
fi

# --- current reservations ---
resv_rows=$(sinfo --reservation 2>/dev/null | tail -n +2 | wc -l)
if [ "$resv_rows" -gt 0 ]; then
    report "current reservations" "resv_dump_current" "OK" "$resv_rows reservations visible via sinfo"
else
    report "current reservations" "resv_dump_current" "UNKNOWN" "no rows: either none active or not visible (PrivateData=reservations)"
fi

# --- historic reservations ---
sreport_rows=$(sreport reservation Utilization -p -n start="$PROBE_START" end="$PROBE_END" format=Name,Nodes,Start,End 2>/dev/null | wc -l)
if [ "$sreport_rows" -gt 0 ]; then
    report "historic reservations" "resv_dump_historic" "OK" "$sreport_rows rows yesterday via sreport"
else
    report "historic reservations" "resv_dump_historic" "UNKNOWN" "no rows: either none yesterday or restricted (PrivateData=usage)"
fi

echo
echo "Any OK row can be dumped with the matching commands in scripts/slurm_dump.sh"
echo "and referenced from the YAML config; omit the config key for anything else"
echo "and FastSim will synthesize it (see README 'Input Data')."
