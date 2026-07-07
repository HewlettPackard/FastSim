#!/bin/bash
# Collect FastSim inputs using only NORMAL USER privileges (no operator/admin).
#
# Collects what an unprivileged account can typically read, and says exactly
# what it is pulling and over which time window. Every item is optional and
# individually selectable; anything you cannot access is skipped with a
# warning rather than aborting the rest.
#
# NOT collected: the all-users job trace. sacct -a is typically restricted
# (PrivateData=jobs); obtain the trace from your site's job-history database
# instead (see README "Input Data" for the required columns). Run
# scripts/check_dump_access.sh first to see what your account can access.

set -u

usage () {
    cat <<'EOF'
Usage: slurm_dump_user.sh [options]

Options:
  -s, --start DATETIME   Window start for HISTORY items (node events, historic
                         reservations). Format YYYY-MM-DD[THH:MM:SS].
                         Default: 90 days ago.
  -e, --end DATETIME     Window end for history items. Default: now.
  -o, --outdir DIR       Output directory. Default: slurm_dump
  -c, --slurm-conf PATH  slurm.conf location. Default: $SLURM_CONF if set,
                         else /nopt/slurm/etc/slurm.conf, else /etc/slurm/slurm.conf
  -i, --include LIST     Comma-separated subset to collect. Default: all of
                         qos,events,resv_current,resv_historic,assocs,slurm_conf
  -n, --dry-run          Print what would be collected (and the exact commands)
                         without running anything.
  -h, --help             Show this help.

Window-scoped items (respect --start/--end): events, resv_historic
Snapshot items (state at the moment you run this): qos, assocs, resv_current, slurm_conf

Each output maps to a FastSim YAML config key:
  qos           -> qos_dump            (sacctmgr_qos.csv)
  events        -> node_events_dump    (sacctmgr_events.csv)
  resv_current  -> resv_dump_current   (sinfo_resv.csv)
  resv_historic -> resv_dump_historic  (sreport_resv.csv)
  assocs        -> assocs_dump         (sacctmgr_assocs.csv, real shares via sshare)
  slurm_conf    -> slurm_conf          (slurm.conf)
EOF
}

# ---- defaults ----
default_start () { date -d "90 days ago" +%Y-%m-%dT00:00:00 2>/dev/null || date -v-90d +%Y-%m-%dT00:00:00; }
STARTTIME=$(default_start)
ENDTIME=$(date +%Y-%m-%dT%H:%M:%S)
OUTDIR="slurm_dump"
INCLUDE="qos,events,resv_current,resv_historic,assocs,slurm_conf"
DRY_RUN=0
SLURM_CONF_PATH="${SLURM_CONF:-}"
if [ -z "$SLURM_CONF_PATH" ]; then
    for candidate in /nopt/slurm/etc/slurm.conf /etc/slurm/slurm.conf; do
        [ -r "$candidate" ] && SLURM_CONF_PATH="$candidate" && break
    done
fi

while [ $# -gt 0 ]; do
    case "$1" in
        -s|--start)      STARTTIME="$2"; shift 2 ;;
        -e|--end)        ENDTIME="$2"; shift 2 ;;
        -o|--outdir)     OUTDIR="$2"; shift 2 ;;
        -c|--slurm-conf) SLURM_CONF_PATH="$2"; shift 2 ;;
        -i|--include)    INCLUDE="$2"; shift 2 ;;
        -n|--dry-run)    DRY_RUN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

included () { case ",$INCLUDE," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

run () { # description, output file, command...
    local desc="$1" outfile="$2"; shift 2
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] $desc"
        echo "          -> $outfile"
        echo "          $*"
        return 0
    fi
    echo "$desc"
    if "$@" > "$outfile" 2> /tmp/dump_err.$$; then
        echo "          -> $outfile ($(wc -l < "$outfile") lines)"
    else
        echo "  WARNING: failed (likely restricted); skipping. $(head -1 /tmp/dump_err.$$ 2>/dev/null)" >&2
        rm -f "$outfile"
    fi
    rm -f /tmp/dump_err.$$
}

echo "FastSim unprivileged dump"
echo "  history window : $STARTTIME -> $ENDTIME  (events, historic reservations)"
echo "  snapshots as of: $(date +%Y-%m-%dT%H:%M:%S)  (qos, assocs, current reservations, slurm.conf)"
echo "  output dir     : $OUTDIR"
echo "  collecting     : $INCLUDE"
echo

[ "$DRY_RUN" -eq 1 ] || mkdir -p "$OUTDIR"

if included qos; then
    run "QOS definitions (snapshot)" "$OUTDIR/sacctmgr_qos.csv" \
        sacctmgr -p list qos format=Name,Priority,GrpTRES,GrpJobs,GrpSubmit,MaxTRESPU,MaxJobsPU,MaxJobs,MaxSubmitPU,MaxSubmit
fi

if included events; then
    run "Node down/drain events ($STARTTIME -> $ENDTIME)" "$OUTDIR/sacctmgr_events.csv" \
        sacctmgr -p show events Start="$STARTTIME" End="$ENDTIME"
fi

if included resv_current; then
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] Current reservations (snapshot)"
        echo "          -> $OUTDIR/sinfo_resv.csv"
        echo "          sinfo --reservation | sed 's/  */|/g'"
    else
        echo "Current reservations (snapshot)"
        if sinfo --reservation 2>/tmp/dump_err.$$ | sed 's/  */|/g' > "$OUTDIR/sinfo_resv.csv"; then
            echo "          -> $OUTDIR/sinfo_resv.csv ($(wc -l < "$OUTDIR/sinfo_resv.csv") lines)"
        else
            echo "  WARNING: sinfo failed; skipping." >&2; rm -f "$OUTDIR/sinfo_resv.csv"
        fi
        rm -f /tmp/dump_err.$$
    fi
fi

if included resv_historic; then
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] Historic reservations ($STARTTIME -> $ENDTIME)"
        echo "          -> $OUTDIR/sreport_resv.csv"
        echo "          sreport reservation Utilization -p -n start=$STARTTIME end=$ENDTIME format=Name,Nodes,Start,End"
    else
        echo "Historic reservations ($STARTTIME -> $ENDTIME)"
        if { echo "Name|Nodes|Start|End|"; sreport reservation Utilization -p -n start="$STARTTIME" end="$ENDTIME" format=Name,Nodes,Start,End; } > "$OUTDIR/sreport_resv.csv" 2>/tmp/dump_err.$$; then
            echo "          -> $OUTDIR/sreport_resv.csv ($(wc -l < "$OUTDIR/sreport_resv.csv") lines)"
        else
            echo "  WARNING: sreport failed (PrivateData=usage?); skipping." >&2; rm -f "$OUTDIR/sreport_resv.csv"
        fi
        rm -f /tmp/dump_err.$$
    fi
fi

if included assocs; then
    # sshare -aP: the Account field is indented one leading space per tree
    # depth, which lets us recover ParentName. sacctmgr show association is
    # often restricted to the caller's own rows, while sshare -a exposes the
    # full fairshare tree with real shares.
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] Associations with real shares (snapshot, via sshare)"
        echo "          -> $OUTDIR/sacctmgr_assocs.csv"
        echo "          sshare -aP --format=Account,User,RawShares | awk <tree reconstruction>"
    else
        echo "Associations with real shares (snapshot, via sshare)"
        if sshare -aP --format=Account,User,RawShares 2>/tmp/dump_err.$$ | awk -F'|' '
            BEGIN { print "Account|User|ParentName|Partition|Shares" }
            NR == 1 { next }
            {
                account = $1
                depth = 0
                while (substr(account, 1, 1) == " ") { depth++; account = substr(account, 2) }
                user = $2
                shares = ($3 ~ /^[0-9]+$/) ? $3 : 1   # "parent" or empty -> 1

                if (user == "") {
                    if (account == "root") { parent_at[0] = "root"; print "root||||1"; next }
                    print account "||" parent_at[depth-1] "||" shares
                    parent_at[depth] = account
                } else {
                    print account "|" user "|||" shares
                }
            }
        ' > "$OUTDIR/sacctmgr_assocs.csv"; then
            echo "          -> $OUTDIR/sacctmgr_assocs.csv ($(wc -l < "$OUTDIR/sacctmgr_assocs.csv") lines)"
        else
            echo "  WARNING: sshare failed; skipping. Omit assocs_dump and FastSim synthesizes identical shares." >&2
            rm -f "$OUTDIR/sacctmgr_assocs.csv"
        fi
        rm -f /tmp/dump_err.$$
    fi
fi

if included slurm_conf; then
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] slurm.conf (snapshot)"
        echo "          -> $OUTDIR/slurm.conf"
        echo "          cp $SLURM_CONF_PATH $OUTDIR/slurm.conf"
    elif [ -n "$SLURM_CONF_PATH" ] && [ -r "$SLURM_CONF_PATH" ]; then
        echo "slurm.conf (snapshot, from $SLURM_CONF_PATH)"
        cp "$SLURM_CONF_PATH" "$OUTDIR/slurm.conf"
        echo "          -> $OUTDIR/slurm.conf"
    else
        echo "  WARNING: no readable slurm.conf found; try 'scontrol write config' or ask admins." >&2
    fi
fi

echo
echo "Done. Reference the produced files from your YAML config (keys listed in"
echo "--help); omit keys for anything skipped and FastSim will synthesize them."
echo "Supply the job trace separately from your job-history database as"
echo "$OUTDIR/sacct_jobs.csv (see README 'Input Data' for required columns)."
