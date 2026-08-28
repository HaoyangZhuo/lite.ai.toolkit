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
./benchmark_yolov26_trt.sh [engine] [image] [warmup] [iterations] [reference_engine]
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
- Commit: `cf713c1` (`perf(trt): fuse YOLOv26 CUDA preprocessing`).

## Stage 3: CUDA Graph replay

### Motivation and implementation

After fused preprocessing, the remaining pipeline still submitted raw H2D, the preprocess kernel, every TensorRT kernel, D2H, and timing events separately on each frame. Stage 3 captures that fixed pipeline after a normal-stream TensorRT warmup and replays one `cudaGraphExec_t` per frame.

The three benchmark modes are now:

- `Baseline`: original pageable/OpenCV path.
- `CudaStream`: the exact cumulative Stage 2 fused path submitted normally; this is the previous-stage comparator.
- `Optimized`: CUDA Graph replay of raw H2D, fused preprocessing, `enqueueV3`, D2H, and five timing-event nodes.

Graph reuse requires the same source width, height, type, compact row/image byte counts, pinned host addresses, raw/input/output device addresses, and pinned output address. A mismatched source shape uses `CudaStream` fallback without discarding the valid graph. Staging growth synchronizes, destroys the graph before freeing captured addresses, grows both buffers, and permits recapture. Capture is attempted up to three times so a transient error does not permanently disable acceleration. A launch/synchronization error quarantines the graph; stream fallback is allowed only if synchronization confirms that the stream remains usable.

The graph uses `cudaEventRecordWithFlags(..., cudaEventRecordExternal)` so captured events are actual event-record nodes and remain valid for elapsed-time queries after replay. This behavior follows the [CUDA Runtime event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html). Content was rephrased for compliance with licensing restrictions.

### Reproduction

```bash
./benchmark_yolov26_trt.sh \
  build/yolo26-export/yolo26n_fp32.engine \
  examples/lite/resources/test_lite_detection_1.jpg \
  20 200
```

Before measured sampling, the benchmark hard-gates:

- graph capture plus first replay for the benchmark shape;
- guaranteed-different source dimensions using normal-stream fallback with matching detections;
- reuse of the original graph after that shape fallback;
- post-capture growth to a 2048x2048 staging payload, graph destruction before address replacement, stream correctness versus Baseline, and recapture for the benchmark shape.

### Three-way benchmark results

| Metric | Original Baseline mean (ms) | Previous CudaStream mean (ms) | Optimized Graph mean (ms) | Graph P50 / P95 / min (ms) |
|---|---:|---:|---:|---:|
| CPU preprocess / raw pack | 3.082117 | 0.198051 | 0.196054 | 0.183591 / 0.259552 / 0.169290 |
| H2D | 0.360364 | 0.111211 | 0.110156 | 0.110112 / 0.111488 / 0.108800 |
| GPU preprocess | 0 | 0.042003 | 0.044955 | 0.045280 / 0.048640 / 0.040832 |
| TensorRT inference | 1.665680 | 1.655743 | 1.474074 | 1.462272 / 1.670144 / 1.449984 |
| D2H | 0.013034 | 0.007449 | 0.007598 | 0.007264 / 0.011360 / 0.003168 |
| GPU pipeline | 2.039078 | 1.816405 | 1.636782 | 1.624800 / 1.833440 / 1.609728 |
| Backend wall | 2.050048 | 1.823696 | 1.646518 | 1.634457 / 1.841854 / 1.618930 |
| CPU postprocess | 0.001711 | 0.001139 | 0.001028 | 0.001010 / 0.001204 / 0.000856 |
| Total wall | **5.136434** | **2.025846** | **1.846252** | **1.821228 / 2.029509 / 1.795716** |

| Throughput | Original Baseline | Previous CudaStream | Optimized Graph |
|---|---:|---:|---:|
| FPS | 194.687593 | 493.621056 | **541.637771** |

### Deltas relative to same-run original and previous stage

| Metric | vs original absolute (ms) | vs original | vs Stage 2 absolute (ms) | vs Stage 2 |
|---|---:|---:|---:|---:|
| CPU preprocess / raw pack | -2.886063 | -93.638988% | -0.001998 | -1.008597% |
| H2D | -0.250208 | -69.432125% | -0.001055 | -0.948687% |
| GPU preprocess | +0.044955 | N/A | +0.002952 | +7.028905% |
| TensorRT inference | -0.191607 | -11.503201% | -0.181669 | -10.972051% |
| D2H | -0.005436 | -41.707791% | +0.000149 | +1.995403% |
| GPU pipeline | -0.402296 | -19.729310% | -0.179623 | -9.888930% |
| Backend wall | -0.403530 | -19.683931% | -0.177178 | -9.715331% |
| CPU postprocess | -0.000684 | -39.947468% | -0.000111 | -9.772918% |
| Total wall | **-3.290182** | **-64.055758%** | **-0.179593** | **-8.865097%** |

- Speedup over same-run original Baseline: **2.782087x**; FPS gain: **+346.950178**.
- Speedup over same-run Stage 2 CudaStream: **1.097274x**; FPS gain: **+48.016715**.

Relative to the separately recorded Stage 0 run:

| Metric | Stage 0 mean | Stage 3 Graph mean | Absolute delta | Delta |
|---|---:|---:|---:|---:|
| CPU preprocess / raw pack (ms) | 2.7208 | 0.196054 | -2.524746 | -92.794252% |
| H2D (ms) | 0.3102 | 0.110156 | -0.200044 | -64.488717% |
| TensorRT inference (ms) | 1.7748 | 1.474074 | -0.300726 | -16.944219% |
| D2H (ms) | 0.0131 | 0.007598 | -0.005502 | -42.000000% |
| GPU pipeline (ms) | 2.0981 | 1.636782 | -0.461318 | -21.987417% |
| Backend wall (ms) | 2.1076 | 1.646518 | -0.461082 | -21.877111% |
| CPU postprocess (ms) | 0.0030 | 0.001028 | -0.001972 | -65.733333% |
| Total wall (ms) | 4.8338 | 1.846252 | -2.987548 | -61.805371% |
| FPS | 206.8784 | 541.637771 | +334.759371 | +161.814559% |

Cross-run final speedup over Stage 0: **2.618169x**.

### Correctness, risks, and limitations

- **200/200 measured Optimized calls used graph replay**, with **0 measured fallbacks**.
- All 200 measured sets matched Baseline/CudaStream/Graph: 5 boxes with identical labels and scores/coordinates within `1e-3`.
- Shape fallback, graph reuse after fallback, staging growth, graph destruction before captured-address replacement, and recapture all passed before timing.
- The inference event segment improves because TensorRT's captured device launches are replayed as graph work, reducing gaps between internal kernels; it is not a change to FP32 arithmetic or engine precision.
- Graph replay remains shape/address-specific. Nonmatching images use safe stream fallback rather than graph updates. A detector holds one active graph signature at a time.
- Five external event nodes are intentionally part of both measured graph and normal-stream paths. They make segment timing reproducible but add small overhead to production calls.
- The detector remains single-call-at-a-time and is not thread-safe.
- Commit: `a1500d8` (`perf(trt): replay YOLOv26 pipeline with CUDA Graph`).

## Stage 4: FP16 internal compute with FP32 I/O

### Engine construction and precision boundary

Stage 4 adds `build_yolov26_trt_engines.sh` so the precision comparison does not depend on the older FP32 engine whose exact build command is unknown. It builds a new FP32 control and an FP16 candidate from the same ONNX file and explicit flags:

```bash
./build_yolov26_trt_engines.sh all \
  build/yolo26-export/yolo26n.onnx \
  build/yolo26-export
```

Both engines use `--inputIOFormats=fp32:chw`, `--outputIOFormats=fp32:chw`, `--builderOptimizationLevel=3`, detailed layer export, and `--skipInference`. The FP16 candidate alone adds `--fp16`. TensorRT's default TF32 behavior remains enabled for the FP32 control; it is therefore an FP32 control with default TF32 tactic selection, not a strict no-TF32 engine.

The external precision boundary is intentional: the existing fused CUDA kernel writes `float`, the handler allocates binding storage using `sizeof(float)`, and postprocessing consumes FP32 output. The FP16 layer dump reports Float input `[1,3,640,640]`, Half internal reformats/convolutions (681 Half datatype entries), and Float `output0` `[1,300,6]`. This stage tests reduced internal compute precision without conflating it with FP16 I/O.

Source model and generated artifacts:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `yolo26n.onnx` | 9,942,064 | `e11058a62d4532cf3af29e7a3f32968f9782bad067ef4a2dbdb3a61d6d037ac0` |
| FP32 control engine | 15,756,292 | `ea41d6b4f7e818e18e6d30020bd772ed5b77e277ff04059178a5de1a2398108c` |
| FP16 engine | 8,309,244 | `8a171552ec9964367d71b3ed39f514a83f52b930b5dccfafce25051bce293e3d` |

FP16 reduces the serialized engine by **7,447,048 bytes (47.263963%)**, from 15.026371 MiB to 7.924313 MiB. Engines, build logs, layer dumps, and benchmark logs stay in ignored `build/` paths and are not committed.

### Benchmark and comparison method

The benchmark now accepts an optional reference engine. It completes all primary-engine warmup and measured samples before loading the reference engine, so reference deserialization/inference cannot affect reported primary latency. Cross-precision detections are matched one-to-one by numeric class and descending IoU with an IoU floor of 0.5; the report includes unmatched counts, same-index class changes, minimum/mean IoU, and maximum score/coordinate differences. This matcher is separate from the strict `1e-3` same-engine execution-path gate.

GPU memory is sampled with synchronized `cudaMemGetInfo` calls before detector construction, after detector load, after initial graph preparation, and after all preflight/warmup work. The final preflight snapshot explicitly includes retained 2048x2048 raw-image staging from the graph growth test. FP32 and FP16 memory deltas are measured in separate processes, before the optional reference engine is loaded.

Four independent processes were run in ABBA order, each with 20 warmup sets and 200 measured sets:

```bash
# A1 and A2
./benchmark_yolov26_trt.sh \
  build/yolo26-export/yolo26n_fp32_stage4.engine \
  examples/lite/resources/test_lite_detection_1.jpg 20 200

# B1 and B2; the FP32 control is loaded only after FP16 timing
./benchmark_yolov26_trt.sh \
  build/yolo26-export/yolo26n_fp16.engine \
  examples/lite/resources/test_lite_detection_1.jpg 20 200 \
  build/yolo26-export/yolo26n_fp32_stage4.engine
```

All four runs passed all nine exact preprocessing gates, 200/200 same-engine detection checks, shape fallback, staging growth/recapture, and 200 graph replays with zero measured graph fallbacks.

### Independent-run results

| Run | Engine | Inference (ms) | GPU pipeline (ms) | Backend wall (ms) | Total wall (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|
| A1 | FP32 control | 1.326541 | 1.488915 | 1.508882 | 1.739050 | 575.026509 |
| B1 | FP16 internal | 0.747069 | 0.909686 | 0.919489 | 1.099937 | 909.142627 |
| B2 | FP16 internal | 0.748078 | 0.915606 | 0.925776 | 1.106076 | 904.097075 |
| A2 | FP32 control | 1.338020 | 1.501965 | 1.512656 | 1.709258 | 585.049076 |

The aggregate below is the arithmetic mean of the two per-engine run means, not a selectively chosen run:

| Metric | FP32 control mean | FP16 mean | Absolute delta | Delta |
|---|---:|---:|---:|---:|
| CPU raw pack (ms) | 0.2094975 | 0.1766975 | -0.0328000 | -15.656511% |
| Raw H2D (ms) | 0.1106260 | 0.1106820 | +0.0000560 | +0.050621% |
| GPU preprocess (ms) | 0.0449105 | 0.0469900 | +0.0020795 | +4.630320% |
| TensorRT inference (ms) | **1.3322805** | **0.7475735** | **-0.5847070** | **-43.887680%** |
| D2H (ms) | 0.0076230 | 0.0074005 | -0.0002225 | -2.918798% |
| GPU pipeline (ms) | **1.4954400** | **0.9126460** | **-0.5827940** | **-38.971406%** |
| Backend wall (ms) | 1.5107690 | 0.9226325 | -0.5881365 | -38.929611% |
| CPU postprocess (ms) | 0.0010695 | 0.0009610 | -0.0001085 | -10.144928% |
| Total wall (ms) | **1.7241540** | **1.1030065** | **-0.6211475** | **-36.026219%** |
| Throughput (FPS) | **580.0377925** | **906.6198510** | **+326.5820585** | **+56.303583%** |

FP16 speedups are **1.782140x** for TensorRT inference, **1.638576x** for the GPU pipeline, and **1.563140x** end-to-end versus the matched FP32 control.

### Engine and process memory

| Measurement | FP32 control | FP16 | FP16 - FP32 |
|---|---:|---:|---:|
| Builder-reported host persistent (bytes) | 611,680 | 598,000 | -13,680 |
| Builder-reported device persistent (bytes) | 9,728 | 0 | -9,728 |
| Builder scratch (bytes) | 2,953,728 | 1,477,632 | -1,476,096 |
| Builder activation (bytes) | 18,662,400 | 9,420,800 | -9,241,600 |
| Builder weights (bytes) | 10,510,272 | 4,908,832 | -5,601,440 |
| Runtime runner allocation reported as scratch (bytes) | 18,662,400 | 9,420,800 | -9,241,600 |
| `cudaMemGetInfo` detector-load delta (bytes) | 73,400,320 (70 MiB) | 102,760,448 (98 MiB) | +29,360,128 (+28 MiB) |
| `cudaMemGetInfo` post-preflight delta (bytes) | 85,983,232 (82 MiB) | 115,343,360 (110 MiB) | +29,360,128 (+28 MiB) |

Both independent runs of each precision produced the same byte deltas. The TensorRT engine-specific activation, weight, and runner allocation reports are smaller for FP16, but the observed process-level free-memory delta is **28 MiB larger**. These values are reported rather than normalized away: `cudaMemGetInfo` measures device-global free memory and includes allocator granularity plus precision/tactic-specific lazily loaded runtime resources, not only serialized weights or activation tensors. Therefore this environment does not support claiming a lower total process footprint for the FP16 engine even though its engine-specific allocations are smaller.

### Detection-output comparison

Both B runs produced identical FP16-versus-FP32 smoke-test results on the benchmark image:

| Measurement | Result |
|---|---:|
| FP16 / FP32 boxes | 5 / 5 |
| Class+IoU matched boxes | 5 |
| Unmatched FP16 / FP32 boxes | 0 / 0 |
| Same-index numeric class mismatches | 0 |
| Minimum / mean matched IoU | 0.997070 / 0.998537 |
| Maximum absolute score difference | 0.002752 |
| Maximum absolute coordinate difference | 0.444672 pixels |

This verifies stable detections for one image and exposes the actual numerical differences; it is **not an accuracy or mAP claim**. No COCO validation set is present in the workspace, so dataset-level FP16 quality remains unmeasured.

### Relative to prior recorded stages

Because Stage 4 uses newly built precision-matched controls and averages independent processes, same-stage FP32-versus-FP16 is the primary causal comparison. Cross-run cumulative figures are included only for historical context:

- Versus Stage 3 Graph: total latency 1.846252 -> 1.1030065 ms (**-40.256991%, 1.673836x**); throughput 541.637771 -> 906.619851 FPS (**+67.384902%**).
- Versus Stage 0: total latency 4.8338 -> 1.1030065 ms (**-77.181379%, 4.382386x**); throughput 206.8784 -> 906.619851 FPS (**+338.238043%**).

### Risks and limitations

- FP16 is hardware-, TensorRT-version-, and tactic-dependent. Engines are not portable substitutes for the reproducible ONNX-plus-build-script path.
- External bindings remain FP32. Changing I/O to FP16 would require coordinated kernel, allocator, host-output, and postprocessing changes and would be a separate optimization.
- The reference engine is loaded after timing but remains simultaneously resident during output comparison; memory metrics are captured before that load.
- `cudaMemGetInfo` is not per-process accounting and may be perturbed by concurrent GPU users. Independent runs and raw snapshots reduce but do not eliminate that limitation.
- The benchmark image validates postprocessed output only. Dataset-level mAP requires real labeled validation assets and must not be inferred from this smoke test.
- Commit: `perf(trt): benchmark YOLOv26 FP16 precision` (hash added after commit).

## Remaining isolated stage

1. INT8 post-training calibration and accuracy/latency comparison, only if real calibration and validation assets are available.

The stage will append its exact code change, A/B delta, correctness result, limitations, and commit identifier after measurement, or document the concrete asset/API blocker without fabricating accuracy data.
