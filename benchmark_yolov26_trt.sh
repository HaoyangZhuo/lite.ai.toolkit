#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${YOLOV26_BUILD_DIR:-${ROOT_DIR}/build/yolo26-trt}"
ENGINE_PATH="${1:-${ROOT_DIR}/build/yolo26-export/yolo26n_fp32.engine}"
IMAGE_PATH="${2:-${ROOT_DIR}/examples/lite/resources/test_lite_detection_1.jpg}"
WARMUP="${3:-20}"
ITERATIONS="${4:-200}"
JOBS="${JOBS:-1}"

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  echo "Missing configured build directory: ${BUILD_DIR}" >&2
  echo "Set YOLOV26_BUILD_DIR to a CMake build configured with ENABLE_TENSORRT=ON and ENABLE_TEST=ON." >&2
  exit 1
fi

if [[ ! -f "${ENGINE_PATH}" ]]; then
  echo "Missing TensorRT engine: ${ENGINE_PATH}" >&2
  exit 1
fi

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Missing benchmark image: ${IMAGE_PATH}" >&2
  exit 1
fi

CONFIGURE_LOG="${BUILD_DIR}/yolov26-benchmark-configure.log"
if ! cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" >"${CONFIGURE_LOG}" 2>&1; then
  echo "YOLOv26 benchmark configure failed; last 100 log lines:" >&2
  tail -n 100 "${CONFIGURE_LOG}" >&2
  exit 1
fi

BUILD_LOG="${BUILD_DIR}/yolov26-benchmark-build.log"
if ! cmake --build "${BUILD_DIR}" --target lite_yolov26_benchmark -j "${JOBS}" >"${BUILD_LOG}" 2>&1; then
  echo "YOLOv26 benchmark build failed; last 100 log lines:" >&2
  tail -n 100 "${BUILD_LOG}" >&2
  exit 1
fi
echo "Incremental YOLOv26 benchmark build completed. Log: ${BUILD_LOG}"

BENCHMARK_BIN="${BUILD_DIR}/install/bin/lite_yolov26_benchmark"
if [[ ! -x "${BENCHMARK_BIN}" ]]; then
  echo "Benchmark executable was not generated: ${BENCHMARK_BIN}" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${ROOT_DIR}/third_party/opencv/lib:${LD_LIBRARY_PATH:-}"
"${BENCHMARK_BIN}" "${ENGINE_PATH}" "${IMAGE_PATH}" "${WARMUP}" "${ITERATIONS}"
