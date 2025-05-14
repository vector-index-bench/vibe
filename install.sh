#!/bin/bash

show_help() {
    echo "Usage: $0 [--algorithm <name>] [--build-dir <dir>] [--force] [--help]"
    echo
    echo "Options:"
    echo "  --algorithm <name>  Specify the algorithm to build. If not specified, build all algorithms."
    echo "  --build-dir <dir>   Specify a temporary directory where the images are built."
    echo "  --force             Force rebuilding images that already exist."
    echo "  --help              Display this help message."
}

algorithm_name=""
build_dir="$(pwd)"
force_build="false"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
    --algorithm)
        if [[ -n "$2" && "$2" != --* ]]; then
            algorithm_name="$2"
            shift 2
        else
            echo "Error: --algorithm requires a value."
            exit 1
        fi
        ;;
    --build-dir)
        if [[ -n "$2" && "$2" != --* ]]; then
            build_dir="$2"
            shift 2
        else
            echo "Error: --build-dir requires a value."
            exit 1
        fi
        ;;
    --force)
        force_build="true"
        shift
        ;;
    --help)
        show_help
        exit 0
        ;;
    *)
        echo "Error: Invalid option $1"
        show_help
        exit 1
        ;;
    esac
done

build_singularity_image() {
    local name="$(basename $1)"

    if [ ! -e "./images/${name}.sif" ] || [ "$force_build" = "true" ]; then
        cp "$1/image.def" "$build_dir/images/${name}.def"
        pushd "$build_dir/images" >/dev/null
        singularity build -F "${name}.sif" "${name}.def"
        popd >/dev/null
        if [ "$build_dir" != "$(pwd)" ]; then
            mv "$build_dir/images/${name}.sif" "./images/${name}.sif"
        fi
    else
        echo "./images/${name}.sif already exists; skipping"
    fi
}

export build_dir
export -f build_singularity_image

set -e
clean_up() {
    ARG=$?
    rm -f "$build_dir/images/environment.yml"
    find "$build_dir/images" -maxdepth 1 -name "*.def" -type f -exec rm {} +
    exit $ARG
}
trap clean_up EXIT

mkdir -p "./images"
mkdir -p "$build_dir/images"

if [ "$build_dir" != "$(pwd)" ]; then
    export SINGULARITY_TMPDIR="$build_dir"
    export SINGULARITY_CACHEDIR="$build_dir"
fi

if [ ! -e "./images/base.sif" ]; then
    cp environment.yml "$build_dir/images/environment.yml"
    build_singularity_image "vibe/algorithms/base"
fi
if [ "$build_dir" != "$(pwd)" ]; then
    cp "./images/base.sif" "$build_dir/images/base.sif"
fi

if [ -n "$algorithm_name" ]; then
    if [ ! -e "vibe/algorithms/$algorithm_name/image.def" ]; then
        echo "image.def does not exist for algorithm $algorithm_name"
        exit 1
    fi
    build_singularity_image "vibe/algorithms/$algorithm_name"
else
    directories=$(find vibe/algorithms -type f -name "image.def" -exec dirname {} \;)

    for dir in $directories; do
        build_singularity_image "$dir"
    done
fi
