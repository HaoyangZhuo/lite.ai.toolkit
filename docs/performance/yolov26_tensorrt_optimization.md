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
- Commit: `dbd3ae9` (`perf(trt): reuse pinned YOLOv26 host buffers`).

## Stage 2: fused CUDA preprocessing

### Motivation and implementation

After Stage 1, CPU letterbox/color conversion/normalization/CHW conversion still consumed about 2.8 ms and dominated end-to-end latency. Stage 2 adds YOLOv26-specific `yolov26_preprocess.cu/.cuh`. The optimized path now:

1. packs the logical `CV_8UC3` rows into a reusable pinned raw-image staging allocation, respecting `cv::Mat::step` for non-contiguous inputs;
2. asynchronously copies the compact BGR uint8 payload to a reusable device allocation;
3. launches one CUDA kernel that performs bilinear resize, letterbox padding, BGR-to-RGB, uint8-to-FP32, `/255` normalization, and HWC-to-CHW;
4. writes directly into the TensorRT input address `buffers[0]`, with no intermediate FP32 device tensor or tensor H2D copy.

The kernel reproduces the 11-bit fixed-point `CV_8U` linear interpolation and two-stage rounding used by the [OpenCV 4.9 resize implementation](https://github.com/opencv/opencv/blob/4.9.0/modules/imgproc/src/resize.cpp), rather than using approximate float bilinear interpolation. Content was rephrased for compliance with licensing restrictions.

Three modes are retained in one binary and detector instance:

- `Baseline`: original per-frame local vectors, pageable FP32 tensor H2D, OpenCV CPU preprocessing.
- `PinnedCpu`: the exact Stage 1 reusable/pinned FP32 host-tensor path.
- `Optimized`: cumulative Stage 2 raw-image staging plus fused CUDA preprocessing.

The benchmark uses six balanced permutations. Every mode occupies every execution slot, while the relative order of Baseline and Optimized alternates each set.

### Reproduction

```bash
./benchmark_yolov26_trt.sh \
  build/yolo26-export/yolo26n_fp32.engine \
  examples/lite/resources/test_lite_detection_1.jpg \
  20 200
```

`20` is the number of three-mode warmup sets and `200` is the number of measured three-mode sets, for 600 measured inference calls.

### Exact preprocessing validation

Validation is a hard benchmark gate: any non-finite or nonzero element difference fails before timing. Each tensor has 1,228,800 FP32 elements.

| Input geometry/layout | Mismatched elements | Mean abs error | P99 abs error | Max abs error |
|---|---:|---:|---:|---:|
| Benchmark image, continuous | 0 | 0 | 0 | 0 |
| Benchmark image, non-contiguous ROI | 0 | 0 | 0 | 0 |
| 319x511 upscale, continuous / ROI | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 721x1283 downscale, continuous / ROI | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 640x640 no-resize, continuous / ROI | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 1x97 single-row edge case | 0 | 0 | 0 | 0 |

This covers up/down scaling, no scaling, padding, synthetic full-range uint8 patterns, a single-row edge case, and stepped ROI storage. It establishes exactness for these cases and this OpenCV build; it is not a proof for every possible image dimension or a different OpenCV implementation.

### Three-way benchmark results

| Metric | Original Baseline mean (ms) | Previous PinnedCpu mean (ms) | Optimized CUDA mean (ms) | Optimized P50 / P95 / min (ms) |
|---|---:|---:|---:|---:|
| CPU preprocess / raw pack | 2.995302 | 2.782893 | 0.234316 | 0.226407 / 0.305197 / 0.186061 |
| H2D | 0.335271 | 0.203921 | 0.111384 | 0.111136 / 0.112608 / 0.110368 |
| GPU preprocess | 0 | 0 | 0.042053 | 0.041984 / 0.043968 / 0.039968 |
| TensorRT inference | 1.637370 | 1.636653 | 1.638656 | 1.635328 / 1.638304 / 1.620992 |
| D2H | 0.013487 | 0.007321 | 0.007398 | 0.007296 / 0.011392 / 0.003200 |
| GPU pipeline | 1.986129 | 1.847895 | 1.799491 | 1.795232 / 1.802048 / 1.777376 |
| Backend wall | 1.996916 | 1.858204 | 1.814061 | 1.802408 / 1.809540 / 1.784404 |
| CPU postprocess | 0.001777 | 0.001399 | 0.001334 | 0.001297 / 0.001559 / 0.001146 |
| Total wall | **4.996411** | **4.644927** | **2.052735** | **2.032542 / 2.115163 / 1.990706** |

| Throughput | Original Baseline | Previous PinnedCpu | Optimized CUDA |
|---|---:|---:|---:|
| FPS | 200.143651 | 215.288630 | **487.155005** |

### Deltas relative to original Baseline and previous stage

| Metric | vs original absolute (ms) | vs original | vs Stage 1 absolute (ms) | vs Stage 1 |
|---|---:|---:|---:|---:|
| CPU preprocess / raw pack | -2.760986 | -92.177211% | -2.548577 | -91.580123% |
| H2D | -0.223887 | -66.777892% | -0.092537 | -45.378887% |
| GPU preprocess | +0.042053 | N/A | +0.042053 | N/A |
| TensorRT inference | +0.001286 | +0.078546% | +0.002004 | +0.122416% |
| D2H | -0.006089 | -45.149236% | +0.000077 | +1.049020% |
| GPU pipeline | -0.186637 | -9.397038% | -0.048404 | -2.619403% |
| Backend wall | -0.182856 | -9.156895% | -0.044143 | -2.375598% |
| CPU postprocess | -0.000443 | -24.945770% | -0.000065 | -4.661053% |
| Total wall | **-2.943677** | **-58.915818%** | **-2.592192** | **-55.806955%** |

- Speedup over same-run original Baseline: **2.434027x**; FPS gain: **+287.011354**.
- Speedup over same-run Stage 1 PinnedCpu: **2.262800x**; FPS gain: **+271.866375**.

Relative to the separately recorded Stage 0 run, total latency changed from 4.8338 to 2.052735 ms (**-2.781065 ms, -57.533721%, 2.354810x**) and throughput changed from 206.8784 to 487.155005 FPS (**+280.276605 FPS, +135.478912%**). The same-run balanced comparisons remain the primary attribution data.

### Correctness, risks, and limitations

- All **200/200 measured sets** matched across all three paths: 5 boxes with identical labels and scores/coordinates within `1e-3`.
- The raw pack time is reported as CPU preprocess for Optimized; GPU preprocessing is a separate CUDA-event metric. These overlap categories must not be added to CPU wall metrics without considering stream execution.
- H2D payloads differ by design: Baseline/PinnedCpu copy a 4.69 MiB FP32 CHW tensor, while Optimized copies the compact raw BGR uint8 image. H2D latency is comparable as pipeline cost, not as equal-byte bandwidth.
- The fused path explicitly requires `CV_8UC3`. Baseline and PinnedCpu retain the earlier three-channel OpenCV behavior for other depths.
- Raw host/device staging grows on demand and remains allocated for the detector lifetime. The detector is still not safe for concurrent calls because it shares context, stream, events, and buffers.
- Any exception after async work is submitted drains the stream before returning, preventing reuse of pinned staging while DMA is in flight.
- Commit: `perf(trt): fuse YOLOv26 CUDA preprocessing` (hash added by the next documentation commit).

## Remaining isolated stages

1. CUDA Graph replay for the fixed-shape pipeline.
2. FP16 engine build and accuracy/latency comparison.
3. INT8 post-training calibration and accuracy/latency comparison.

Each stage will append its exact code change, A/B delta, correctness result, limitations, and commit identifier after measurement.
