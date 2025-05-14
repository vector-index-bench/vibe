#!/bin/bash

local_mode="false"
singularity_args=""
script_args=""

while [[ $# -gt 0 ]]; do
    case "$1" in
    --local)
        local_mode="true"
        ;;
    --singularity-args)
        shift
        singularity_args="$1"
        ;;
    *)
        script_args="$script_args $1"
        ;;
    esac
    shift
done

if [ $local_mode = "false" ]; then
    singularity exec $singularity_args "dataset.sif" python3 create_dataset.py $script_args
else
    python3 create_dataset.py $args
fi
