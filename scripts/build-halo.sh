#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
JOBS=${JOBS:-$(nproc)}
TARGET=${TARGET:-llama-cli}

usage() {
    printf 'Usage: %s [all|rocm|vulkan|rocwmma]...\n' "${0##*/}"
    printf 'Environment: ROCM_PATH, JOBS, TARGET (default: llama-cli)\n'
}

configure_and_build() {
    local name=$1
    shift

    printf '\n==> Building %s in build-%s\n' "$name" "$name"
    cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build-$name" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF \
        "$@"
    cmake --build "$ROOT_DIR/build-$name" --config Release --target "$TARGET" -- -j "$JOBS"
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

variants=("${@:-all}")
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

printf '\nBuild complete. For Strix Halo HIP runs, set:\n'
printf '  GGML_CUDA_ENABLE_UNIFIED_MEMORY=1\n'
