#!/bin/bash
# Drive one TabPFN variant across every date of the panel.
#
# TabPFN is refitted from scratch at each date, so each run handles one date and
# prints the next one; this loop feeds that back in until the script prints STOP.
# A full sweep is hours on GPU/MPS, which is why the per-date outputs are
# appended to CSV and audited later by scripts/audit_tabpfn.py rather than
# recomputed.
#
#   ./scripts/tabpfn/sweep.sh run_delta_dom_new.py 2003-08-01
#
# Requires TABPFN_TOKEN and HF_TOKEN in the environment.

set -u
SCRIPT="${1:?usage: sweep.sh <run_*.py> [start-date]}"
CURRENT_ARG="${2:-2003-08-01}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval "$(conda shell.bash hook)"
conda activate bond

while [ -n "$CURRENT_ARG" ]; do
    echo "--- $SCRIPT $CURRENT_ARG ---"
    OUTPUT=$(python3 "$HERE/$SCRIPT" "$CURRENT_ARG" 2>&1)
    echo "$OUTPUT"
    NEXT_ARG=$(echo "$OUTPUT" | tail -n 1)
    if [ "$NEXT_ARG" == "STOP" ] || [ -z "$NEXT_ARG" ]; then
        echo "sweep complete."
        break
    fi
    CURRENT_ARG="$NEXT_ARG"
done
