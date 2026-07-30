#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
JOBS=${JOBS:-$(nproc)}
TARGET=${TARGET:-"llama-cli llama-server"}

usage() {
    printf 'Usage: %s [all|rocm|vulkan|rocwmma]...\n' "${0##*/}"
    printf 'Default variant: rocm (output: build/)\n'
    printf 'Environment: ROCM_PATH, JOBS, TARGET (default: "llama-cli llama-server")\n'
}

configure_and_build() {
    local name=$1
    local build_dir="$ROOT_DIR/build-$name"
    local targets
    shift

    if [[ "$name" == "rocm" ]]; then
        build_dir="$ROOT_DIR/build"
    fi

    read -r -a targets <<< "$TARGET"

    printf '\n==> Building %s in %s\n' "$name" "${build_dir#"$ROOT_DIR"/}"
    cmake -S "$ROOT_DIR" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF \
        "$@"
    cmake --build "$build_dir" --config Release --target "${targets[@]}" -- -j "$JOBS"

    printf 'Built binaries are in %s/bin\n' "$build_dir"
}

build_rocm() {
    if [[ ! -x "$ROCM_PATH/bin/hipcc" ]]; then
        printf 'error: hipcc not found at %s/bin/hipcc\n' "$ROCM_PATH" >&2
        exit 1
    fi

    PATH="$ROCM_PATH/bin:$PATH" configure_and_build rocm \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS=gfx1151 \
        -DGGML_CUDA_FA_ALL_QUANTS=ON
}

build_vulkan() {
    configure_and_build vulkan -DGGML_VULKAN=ON
}

build_rocwmma() {
    if [[ ! -f "$ROCM_PATH/include/rocwmma/rocwmma.hpp" ]]; then
        printf 'error: rocWMMA headers not found under %s/include/rocwmma\n' "$ROCM_PATH" >&2
        exit 1
    fi

    printf 'note: CachyLlama currently excludes gfx1151 from rocWMMA FA runtime selection.\n'
    PATH="$ROCM_PATH/bin:$PATH" configure_and_build rocwmma \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS=gfx1151 \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_HIP_ROCWMMA_FATTN=ON
}

if [[ ${1:-all} == "--help" || ${1:-all} == "-h" ]]; then
    usage
    exit 0
fi

variants=("${@:-rocm}")
for variant in "${variants[@]}"; do
    case "$variant" in
        all)
            build_rocm
            build_vulkan
            build_rocwmma
            ;;
        rocm) build_rocm ;;
        vulkan) build_vulkan ;;
        rocwmma) build_rocwmma ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

printf '\nBuild complete. Do not set GGML_CUDA_ENABLE_UNIFIED_MEMORY for DeepSeek V4 Flash; it corrupts output on gfx1151.\n'
