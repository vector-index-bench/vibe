#!/bin/bash

show_help() {
    echo "Usage: $0 [--algorithm <name>] [--build-dir <dir>] [--image-dir <dir>] [--skip-gpu] [--force] [--help]"
    echo
    echo "Options:"
    echo "  --algorithm <name>  Specify the algorithm to build. If not specified, build all algorithms."
    echo "  --build-dir <dir>   Specify a temporary directory where the images are built."
    echo "  --image-dir <dir>   Specify the directory where completed Singularity images are stored."
    echo "  --skip-gpu          Skip GPU algorithms when building all algorithms."
    echo "  --force             Force rebuilding images that already exist."
    echo "  --help              Display this help message."
}

algorithm_name=""
build_dir="$(pwd)"
image_dir="./images"
force_build="false"
skip_gpu="false"

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
    --image-dir)
        if [[ -n "$2" && "$2" != --* ]]; then
            image_dir="$2"
            shift 2
        else
            echo "Error: --image-dir requires a value."
            exit 1
        fi
        ;;
    --force)
        force_build="true"
        shift
        ;;
    --skip-gpu)
        skip_gpu="true"
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
    local name
    name="$(basename "$1")"

    if [ ! -e "$image_dir/${name}.sif" ] || [ "$force_build" = "true" ]; then
        cp "$1/image.def" "$build_images_dir/${name}.def"
        pushd "$build_images_dir" >/dev/null
        singularity build -F "${name}.sif" "${name}.def"
        popd >/dev/null
        if [ "$build_images_dir" != "$image_dir" ]; then
            mv "$build_images_dir/${name}.sif" "$image_dir/${name}.sif"
        fi
    else
        echo "$image_dir/${name}.sif already exists; skipping"
    fi
}

export build_dir image_dir
export -f build_singularity_image

set -e

mkdir -p "$image_dir"
mkdir -p "$build_dir/images"

image_dir="$(cd "$image_dir" && pwd -P)"
build_images_dir="$(cd "$build_dir/images" && pwd -P)"

clean_up() {
    ARG=$?
    rm -f "$build_images_dir/environment.yml"
    find "$build_images_dir" -maxdepth 1 -name "*.def" -type f -exec rm {} +
    exit $ARG
}
trap clean_up EXIT

if [ "$build_dir" != "$(pwd)" ]; then
    export SINGULARITY_TMPDIR="$build_dir"
    export SINGULARITY_CACHEDIR="$build_dir"
fi

if [ ! -e "$image_dir/base.sif" ]; then
    cp environment.yml "$build_images_dir/environment.yml"
    build_singularity_image "vibe/algorithms/base"
fi
if [ "$build_images_dir" != "$image_dir" ]; then
    cp "$image_dir/base.sif" "$build_images_dir/base.sif"
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
        if [ "$skip_gpu" = "true" ] && grep -qE '^[[:space:]]*gpu:[[:space:]]*true[[:space:]]*$' "$dir/config.yml"; then
            echo "Skipping GPU algorithm $(basename "$dir")"
            continue
        fi
        build_singularity_image "$dir"
    done
fi
