#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRECISION="${1:-all}"
ONNX_PATH="${2:-${ROOT_DIR}/build/yolo26-export/yolo26n.onnx}"
OUTPUT_DIR="${3:-${ROOT_DIR}/build/yolo26-export}"
TRTEXEC="${TRTEXEC:-/root/autodl-tmp/third_party/TensorRT-10.9.0.34/bin/trtexec}"

if [[ ! -x "${TRTEXEC}" ]]; then
  echo "Missing trtexec executable: ${TRTEXEC}" >&2
  exit 1
fi
if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "Missing YOLOv26 ONNX model: ${ONNX_PATH}" >&2
  exit 1
fi
if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "Missing output directory: ${OUTPUT_DIR}" >&2
  exit 1
fi
if [[ "${PRECISION}" != "fp32" && "${PRECISION}" != "fp16" && "${PRECISION}" != "all" ]]; then
  echo "Usage: $0 [fp32|fp16|all] [onnx_path] [output_dir]" >&2
  exit 1
fi

build_engine() {
  local precision="$1"
  local stem="yolo26n_${precision}"
  local precision_args=()

  if [[ "${precision}" == "fp32" ]]; then
    stem="yolo26n_fp32_stage4"
  else
    precision_args+=(--fp16)
  fi

  local engine_path="${OUTPUT_DIR}/${stem}.engine"
  local layer_path="${OUTPUT_DIR}/${stem}_layers.json"
  local log_path="${OUTPUT_DIR}/${stem}_build.log"

  echo "Building ${precision} engine: ${engine_path}"
  "${TRTEXEC}" \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${engine_path}" \
    --inputIOFormats=fp32:chw \
    --outputIOFormats=fp32:chw \
    --builderOptimizationLevel=3 \
    --profilingVerbosity=detailed \
    --exportLayerInfo="${layer_path}" \
    --skipInference \
    "${precision_args[@]}" \
    >"${log_path}" 2>&1
  echo "Built ${engine_path}; log: ${log_path}"
}

if [[ "${PRECISION}" == "all" || "${PRECISION}" == "fp32" ]]; then
  build_engine fp32
fi
if [[ "${PRECISION}" == "all" || "${PRECISION}" == "fp16" ]]; then
  build_engine fp16
fi
