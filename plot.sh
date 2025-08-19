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

export MPLCONFIGDIR=/tmp/matplotlib
export FONTCONFIG_CACHE=/tmp/fontconfig

if [ $local_mode = "false" ]; then
    singularity exec "plot.sif" python3 plot.py $args
else
    python3 plot.py $args
fi
