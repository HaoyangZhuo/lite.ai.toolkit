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

## Stage 1: reusable page-locked host buffers

### Motivation and implementation

The Stage 0 profile showed that the pageable input transfer was the largest non-preprocessing opportunity. The optimized path now allocates the fixed-shape input/output host vectors once during detector construction and registers both allocations with `cudaHostRegister`. Per-frame local input/output vector allocation is eliminated. The original implementation remains available as `PipelineMode::Baseline`; it still constructs local vectors backed by pageable memory on every call. `PipelineMode::Optimized` selects the reusable page-locked buffers and is the default for the public API.

The benchmark was changed from a single-path loop to paired A/B sampling in one process and one detector. Every round runs both modes, reverses their order on alternating rounds, and compares box count, classes, scores, and coordinates both against the warmed reference and against each other.

### Reproduction

```bash
./benchmark_yolov26_trt.sh \
  build/yolo26-export/yolo26n_fp32.engine \
  examples/lite/resources/test_lite_detection_1.jpg \
  20 200
```

This was measured in the environment documented above. `20` means 20 warmup **pairs** and `200` means 200 measured **pairs**, for 400 measured inference calls.

### Paired A/B results

| Metric | Baseline mean (ms) | Optimized mean (ms) | Absolute delta (ms) | Delta | Optimized P50 / P95 / min (ms) |
|---|---:|---:|---:|---:|---:|
| CPU preprocess | 2.9436 | 2.7855 | -0.1581 | -5.3723% | 2.7600 / 2.9558 / 2.5789 |
| H2D | 0.3297 | 0.2043 | -0.1254 | -38.0368% | 0.2040 / 0.2052 / 0.2032 |
| TensorRT inference | 1.8017 | 1.8094 | +0.0077 | +0.4274% | 1.8096 / 1.8812 / 1.7441 |
| D2H | 0.0139 | 0.0075 | -0.0064 | -45.9488% | 0.0071 / 0.0112 / 0.0030 |
| GPU pipeline | 2.1452 | 2.0212 | -0.1241 | -5.7837% | 2.0208 / 2.0948 / 1.9528 |
| Backend wall | 2.1491 | 2.0251 | -0.1240 | -5.7694% | 2.0246 / 2.0978 / 1.9568 |
| CPU postprocess | 0.0016 | 0.0012 | -0.0004 | -25.4218% | 0.0012 / 0.0015 / 0.0010 |
| Total wall | **5.0970** | **4.8145** | **-0.2826** | **-5.5436%** | **4.7839 / 5.0611 / 4.5710** |

| Throughput | Baseline | Optimized | Absolute delta | Delta |
|---|---:|---:|---:|---:|
| FPS | 196.1930 | 207.7075 | +11.5145 | +5.8689% |

Paired Stage 1 speedup: **1.0587x**.

### Relative to the original Stage 0 run and previous stage

Stage 0 is both the original baseline and the immediately previous stage. This cross-run table is included as requested, but the paired same-process table above is the primary causal comparison because it controls execution order, GPU Boost, and temperature.

| Metric | Stage 0 mean | Stage 1 optimized mean | Absolute delta | Delta |
|---|---:|---:|---:|---:|
| CPU preprocess (ms) | 2.7208 | 2.7855 | +0.0647 | +2.3780% |
| H2D (ms) | 0.3102 | 0.2043 | -0.1059 | -34.1393% |
| TensorRT inference (ms) | 1.7748 | 1.8094 | +0.0346 | +1.9495% |
| D2H (ms) | 0.0131 | 0.0075 | -0.0056 | -42.7481% |
| GPU pipeline (ms) | 2.0981 | 2.0212 | -0.0769 | -3.6652% |
| Backend wall (ms) | 2.1076 | 2.0251 | -0.0825 | -3.9144% |
| CPU postprocess (ms) | 0.0030 | 0.0012 | -0.0018 | -60.0000% |
| Total wall (ms) | 4.8338 | 4.8145 | -0.0193 | -0.3993% |
| FPS | 206.8784 | 207.7075 | +0.8291 | +0.4008% |

Cross-run total speedup over Stage 0: **1.0040x**. The smaller cross-run gain and the slower Stage 1 paired baseline illustrate why the alternating same-process A/B result is more reliable than comparing isolated executions.

### Correctness, risks, and limitations

- Detected boxes: **5** in both modes.
- Correctness: **all 200 measured pairs passed** the count/class/score/coordinate checks at `1e-3`; warmup pairs also matched.
- The reusable vectors are registered rather than allocated with `cudaHostAlloc`; their fixed size and address are checked before transfer. The detector remains single-call-at-a-time because its context, stream, and reusable buffers are shared.
- Pinned host memory is a limited OS/CUDA resource and remains registered for the detector lifetime. The destructor unregisters it.
- Input preprocessing is still CPU OpenCV plus scalar HWC-to-CHW and dominates latency. Stage 2 targets this bottleneck.
- Commit: `perf(trt): reuse pinned YOLOv26 host buffers` (the immutable hash is added by the next documentation commit because a commit cannot contain its own hash).

## Remaining isolated stages

1. Fused CUDA letterbox, bilinear resize, BGR-to-RGB, normalization, and HWC-to-CHW.
2. CUDA Graph replay for the fixed-shape pipeline.
3. FP16 engine build and accuracy/latency comparison.
4. INT8 post-training calibration and accuracy/latency comparison.

Each stage will append its exact code change, A/B delta, correctness result, limitations, and commit identifier after measurement.
