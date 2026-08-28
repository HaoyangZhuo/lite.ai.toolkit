# YOLOv26 TensorRT Optimization Log

This document records each optimization independently. Every stage uses the same model, image, warmup count, sample count, output-consistency check, and metric definitions unless explicitly noted.

## Environment and methodology

- Branch: `perf/yolo26-tensorrt`
- GPU: NVIDIA GeForce RTX 3090, 24 GiB
- NVIDIA driver: 570.124.04
- CUDA: 12.8
- TensorRT: 10.9.0.34
- Model: YOLOv26n, static batch 1, 640x640, end-to-end `[1,300,6]` output
- Baseline engine: FP32
- Input: `examples/lite/resources/test_lite_detection_1.jpg`
- Warmup: 20 iterations
- Measured samples: 200 iterations
- Correctness guard: every measured iteration must match the warmed-up reference boxes in count, labels, scores, and coordinates within `1e-3`

Run the benchmark with:

```bash
./benchmark_yolov26_trt.sh [engine] [image] [warmup] [iterations]
```

The script reuses `build/yolo26-trt`, refreshes only the CMake generation step, incrementally builds the `lite_yolov26_benchmark` target, and runs it. Build output is saved under the build directory instead of flooding benchmark output.

`build/yolo26-export/export_yolo26.py` only exports the PyTorch model to ONNX. It cannot validate changes made to the C++ TensorRT path or CUDA kernels, so the targeted C++ benchmark remains necessary.

### Metric definitions

- `preprocess_cpu`: letterbox resize, BGR-to-RGB, FP32 normalization, and HWC-to-CHW conversion measured with `std::chrono::steady_clock`.
- `h2d_gpu`: input transfer duration on the TensorRT CUDA stream, measured with CUDA events.
- `inference_gpu`: `enqueueV3` device execution, measured with CUDA events.
- `d2h_gpu`: output transfer duration on the same stream, measured with CUDA events.
- `gpu_pipeline`: sum of H2D, inference, and D2H event durations.
- `backend_wall`: host-observed duration covering output allocation, copies, enqueue, and final stream synchronization.
- `postprocess_cpu`: filtering, coordinate restoration, sorting, and top-k.
- `total_wall`: host-observed end-to-end `detect` latency excluding model/image loading.

CUDA event durations and CPU wall durations intentionally remain separate. Their difference exposes host allocation, API launch, pageable-memory staging, and synchronization overhead.

## Stage 0: reproducible FP32 baseline

### Change

- Added an optional instrumented YOLOv26 TensorRT detection path; the existing `detect` API keeps its non-instrumented path.
- Added CUDA events to split H2D, TensorRT execution, and D2H without introducing intermediate synchronization.
- Added a dedicated repeated-sampling benchmark with percentile reporting and output-consistency checks.
- Added a YOLOv26-only incremental build/run script.

### Results

| Metric | Mean (ms) | P50 (ms) | P95 (ms) | Min (ms) |
|---|---:|---:|---:|---:|
| CPU preprocess | 2.7208 | 2.6993 | 2.9046 | 2.5760 |
| H2D | 0.3102 | 0.3082 | 0.3212 | 0.3011 |
| TensorRT inference | 1.7748 | 1.7302 | 1.8806 | 1.7114 |
| D2H | 0.0131 | 0.0134 | 0.0168 | 0.0085 |
| GPU pipeline | 2.0981 | 2.0596 | 2.2044 | 2.0306 |
| Backend wall | 2.1076 | 2.0691 | 2.2138 | 2.0399 |
| CPU postprocess | 0.0030 | 0.0028 | 0.0038 | 0.0025 |
| Total wall | **4.8338** | **4.8037** | **5.0028** | **4.6823** |

- Mean throughput: **206.8784 FPS**
- Detected boxes: **5**
- Output stability check: **passed for all 200 samples**

### Finding

CPU preprocessing consumes 56.3% of mean end-to-end latency and is slower than TensorRT execution itself. H2D is the second model-external target. This justifies optimizing reusable/pinned host storage first and then replacing the CPU preprocessing chain with a fused CUDA path.

## Planned isolated stages

1. Reusable host buffers and page-locked transfers.
2. Fused CUDA letterbox, bilinear resize, BGR-to-RGB, normalization, and HWC-to-CHW.
3. CUDA Graph replay for the fixed-shape pipeline.
4. FP16 engine build and accuracy/latency comparison.
5. INT8 post-training calibration and accuracy/latency comparison.

Each stage will append its exact code change, A/B delta, correctness result, limitations, and commit identifier after measurement.
