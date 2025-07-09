#!/bin/bash

local_mode="false"
args="$@"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --local)
        local_mode="true"
        ;;
    esac
    shift
done

if [ $local_mode = "false" ]; then
    singularity exec "plot.sif" python3 export_results.py $args
else
    python3 export_results.py $args
fi
