//
// YOLOv26 TensorRT paired latency benchmark.
//

#include "lite/lite.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ENABLE_TENSORRT
#include <cuda_runtime_api.h>

namespace
{
  using Detector = lite::trt::cv::detection::YOLOV26;
  using Timing = Detector::Timing;
  using PipelineMode = Detector::PipelineMode;

  struct Statistics
  {
    double mean = 0.0;
    double minimum = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
  };

  struct Samples
  {
    std::vector<double> preprocess;
    std::vector<double> h2d;
    std::vector<double> gpu_preprocess;
    std::vector<double> inference;
    std::vector<double> d2h;
    std::vector<double> gpu_pipeline;
    std::vector<double> backend_wall;
    std::vector<double> postprocess;
    std::vector<double> total;

    void reserve(std::size_t count)
    {
      preprocess.reserve(count);
      h2d.reserve(count);
      gpu_preprocess.reserve(count);
      inference.reserve(count);
      d2h.reserve(count);
      gpu_pipeline.reserve(count);
      backend_wall.reserve(count);
      postprocess.reserve(count);
      total.reserve(count);
    }

    void append(const Timing &timing)
    {
      preprocess.push_back(timing.preprocess_ms);
      h2d.push_back(timing.h2d_ms);
      gpu_preprocess.push_back(timing.gpu_preprocess_ms);
      inference.push_back(timing.inference_ms);
      d2h.push_back(timing.d2h_ms);
      gpu_pipeline.push_back(timing.gpu_pipeline_ms());
      backend_wall.push_back(timing.backend_wall_ms);
      postprocess.push_back(timing.postprocess_ms);
      total.push_back(timing.total_ms);
    }
  };

  Statistics summarize(std::vector<double> values)
  {
    if (values.empty()) throw std::invalid_argument("Cannot summarize empty samples");
    std::sort(values.begin(), values.end());
    const auto percentile = [&values](double p)
    {
      const std::size_t index = static_cast<std::size_t>(
          std::ceil(p * static_cast<double>(values.size()))) - 1;
      return values[std::min(index, values.size() - 1)];
    };
    Statistics stats;
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                 static_cast<double>(values.size());
    stats.minimum = values.front();
    stats.p50 = percentile(0.50);
    stats.p95 = percentile(0.95);
    return stats;
  }

  bool same_boxes(const std::vector<lite::types::Boxf> &expected,
                  const std::vector<lite::types::Boxf> &actual,
                  float tolerance = 1e-3f)
  {
    if (expected.size() != actual.size()) return false;
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
      if (expected[i].label != actual[i].label ||
          std::fabs(expected[i].score - actual[i].score) > tolerance ||
          std::fabs(expected[i].x1 - actual[i].x1) > tolerance ||
          std::fabs(expected[i].y1 - actual[i].y1) > tolerance ||
          std::fabs(expected[i].x2 - actual[i].x2) > tolerance ||
          std::fabs(expected[i].y2 - actual[i].y2) > tolerance)
        return false;
    }
    return true;
  }

  struct CudaMemorySnapshot
  {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
  };

  struct PrecisionComparison
  {
    std::size_t primary_boxes = 0;
    std::size_t reference_boxes = 0;
    std::size_t matched = 0;
    std::size_t unmatched_primary = 0;
    std::size_t unmatched_reference = 0;
    std::size_t same_index_label_mismatches = 0;
    double minimum_iou = 0.0;
    double mean_iou = 0.0;
    double maximum_score_abs_error = 0.0;
    double maximum_coordinate_abs_error = 0.0;
  };

  CudaMemorySnapshot cuda_memory_snapshot()
  {
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") +
                               cudaGetErrorString(status));

    CudaMemorySnapshot snapshot;
    status = cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes);
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("cudaMemGetInfo failed: ") +
                               cudaGetErrorString(status));
    return snapshot;
  }

  long long consumed_bytes(const CudaMemorySnapshot &before,
                           const CudaMemorySnapshot &after)
  {
    return static_cast<long long>(before.free_bytes) -
           static_cast<long long>(after.free_bytes);
  }

  float box_iou(const lite::types::Boxf &first, const lite::types::Boxf &second)
  {
    const float left = std::max(first.x1, second.x1);
    const float top = std::max(first.y1, second.y1);
    const float right = std::min(first.x2, second.x2);
    const float bottom = std::min(first.y2, second.y2);
    const float intersection = std::max(0.0f, right - left) *
                               std::max(0.0f, bottom - top);
    const float first_area = std::max(0.0f, first.x2 - first.x1) *
                             std::max(0.0f, first.y2 - first.y1);
    const float second_area = std::max(0.0f, second.x2 - second.x1) *
                              std::max(0.0f, second.y2 - second.y1);
    const float union_area = first_area + second_area - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
  }

  PrecisionComparison compare_precision_outputs(
      const std::vector<lite::types::Boxf> &primary,
      const std::vector<lite::types::Boxf> &reference,
      float minimum_match_iou = 0.5f)
  {
    struct Candidate
    {
      std::size_t primary_index;
      std::size_t reference_index;
      float iou;
    };

    PrecisionComparison comparison;
    comparison.primary_boxes = primary.size();
    comparison.reference_boxes = reference.size();
    const std::size_t common_size = std::min(primary.size(), reference.size());
    for (std::size_t i = 0; i < common_size; ++i)
      if (primary[i].label != reference[i].label)
        ++comparison.same_index_label_mismatches;

    std::vector<Candidate> candidates;
    for (std::size_t i = 0; i < primary.size(); ++i)
      for (std::size_t j = 0; j < reference.size(); ++j)
      {
        if (primary[i].label != reference[j].label) continue;
        const float iou = box_iou(primary[i], reference[j]);
        if (iou >= minimum_match_iou) candidates.push_back({i, j, iou});
      }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &first, const Candidate &second)
              {
                if (first.iou != second.iou) return first.iou > second.iou;
                if (first.primary_index != second.primary_index)
                  return first.primary_index < second.primary_index;
                return first.reference_index < second.reference_index;
              });

    std::vector<bool> primary_matched(primary.size(), false);
    std::vector<bool> reference_matched(reference.size(), false);
    double iou_sum = 0.0;
    for (const Candidate &candidate : candidates)
    {
      if (primary_matched[candidate.primary_index] ||
          reference_matched[candidate.reference_index])
        continue;
      primary_matched[candidate.primary_index] = true;
      reference_matched[candidate.reference_index] = true;
      const auto &primary_box = primary[candidate.primary_index];
      const auto &reference_box = reference[candidate.reference_index];
      ++comparison.matched;
      iou_sum += candidate.iou;
      if (comparison.matched == 1 || candidate.iou < comparison.minimum_iou)
        comparison.minimum_iou = candidate.iou;
      comparison.maximum_score_abs_error = std::max(
          comparison.maximum_score_abs_error,
          static_cast<double>(std::fabs(primary_box.score - reference_box.score)));
      comparison.maximum_coordinate_abs_error = std::max({
          comparison.maximum_coordinate_abs_error,
          static_cast<double>(std::fabs(primary_box.x1 - reference_box.x1)),
          static_cast<double>(std::fabs(primary_box.y1 - reference_box.y1)),
          static_cast<double>(std::fabs(primary_box.x2 - reference_box.x2)),
          static_cast<double>(std::fabs(primary_box.y2 - reference_box.y2))});
    }

    comparison.unmatched_primary = primary.size() - comparison.matched;
    comparison.unmatched_reference = reference.size() - comparison.matched;
    if (comparison.matched != 0)
      comparison.mean_iou = iou_sum / static_cast<double>(comparison.matched);
    return comparison;
  }

  int positive_integer(const char *value, const char *name)
  {
    const int parsed = std::stoi(value);
    if (parsed <= 0)
      throw std::invalid_argument(std::string(name) + " must be positive");
    return parsed;
  }

  void run_once(Detector &detector, const cv::Mat &image, PipelineMode mode,
                std::vector<lite::types::Boxf> &boxes, Timing *timing)
  {
    if (timing)
      detector.detect_with_timing(image, boxes, *timing, 0.25f, 100, mode);
    else
      detector.detect(image, boxes, 0.25f, 100, mode);
  }

  void print_metric(const char *mode, const char *name,
                    const std::vector<double> &samples)
  {
    const Statistics stats = summarize(samples);
    std::cout << mode << ',' << name << ',' << stats.mean << ',' << stats.p50 << ','
              << stats.p95 << ',' << stats.minimum << '\n';
  }

  void print_samples(const char *mode, const Samples &samples)
  {
    print_metric(mode, "preprocess_cpu", samples.preprocess);
    print_metric(mode, "h2d_gpu", samples.h2d);
    print_metric(mode, "preprocess_gpu", samples.gpu_preprocess);
    print_metric(mode, "inference_gpu", samples.inference);
    print_metric(mode, "d2h_gpu", samples.d2h);
    print_metric(mode, "gpu_pipeline", samples.gpu_pipeline);
    print_metric(mode, "backend_wall", samples.backend_wall);
    print_metric(mode, "postprocess_cpu", samples.postprocess);
    print_metric(mode, "total_wall", samples.total);
  }

  void print_comparison(const char *comparison, const char *name,
                        const std::vector<double> &reference,
                        const std::vector<double> &candidate)
  {
    const double reference_mean = summarize(reference).mean;
    const double candidate_mean = summarize(candidate).mean;
    const double delta = candidate_mean - reference_mean;
    std::cout << comparison << ',' << name << ',' << reference_mean << ','
              << candidate_mean << ',' << delta << ',';
    if (reference_mean == 0.0)
      std::cout << "N/A\n";
    else
      std::cout << delta * 100.0 / reference_mean << '\n';
  }

  void print_comparisons(const char *comparison, const Samples &reference,
                         const Samples &candidate)
  {
    print_comparison(comparison, "preprocess_cpu", reference.preprocess, candidate.preprocess);
    print_comparison(comparison, "h2d_gpu", reference.h2d, candidate.h2d);
    print_comparison(comparison, "preprocess_gpu", reference.gpu_preprocess,
                     candidate.gpu_preprocess);
    print_comparison(comparison, "inference_gpu", reference.inference, candidate.inference);
    print_comparison(comparison, "d2h_gpu", reference.d2h, candidate.d2h);
    print_comparison(comparison, "gpu_pipeline", reference.gpu_pipeline,
                     candidate.gpu_pipeline);
    print_comparison(comparison, "backend_wall", reference.backend_wall,
                     candidate.backend_wall);
    print_comparison(comparison, "postprocess_cpu", reference.postprocess,
                     candidate.postprocess);
    print_comparison(comparison, "total_wall", reference.total, candidate.total);
  }

  void print_preprocess_comparison(const char *layout,
                                   const Detector::PreprocessComparison &comparison)
  {
    std::cout << "preprocess_validation," << layout << ',' << comparison.elements << ','
              << comparison.mismatched << ',' << comparison.mean_abs_error << ','
              << comparison.p99_abs_error << ',' << comparison.max_abs_error << '\n';
  }

  void require_exact_preprocess(Detector &detector, const std::string &name,
                                const cv::Mat &image)
  {
    const Detector::PreprocessComparison contiguous = detector.compare_preprocess(image);
    print_preprocess_comparison((name + "_contiguous").c_str(), contiguous);
    if (contiguous.mismatched != 0 || contiguous.max_abs_error != 0.0f)
      throw std::runtime_error("Fused preprocess differs from OpenCV for " + name);

    if (image.rows == 1) return;

    cv::Mat storage(image.rows + 4, image.cols + 6, image.type());
    cv::Mat roi = storage(cv::Rect(3, 2, image.cols, image.rows));
    image.copyTo(roi);
    if (roi.isContinuous())
      throw std::runtime_error("Internal non-contiguous ROI validation setup failed");
    const Detector::PreprocessComparison stepped = detector.compare_preprocess(roi);
    print_preprocess_comparison((name + "_non_contiguous_roi").c_str(), stepped);
    if (stepped.mismatched != 0 || stepped.max_abs_error != 0.0f)
      throw std::runtime_error("Fused preprocess differs from OpenCV for stepped " + name);
  }

  cv::Mat make_pattern(int rows, int cols)
  {
    cv::Mat image(rows, cols, CV_8UC3);
    for (int y = 0; y < rows; ++y)
    {
      cv::Vec3b *row = image.ptr<cv::Vec3b>(y);
      for (int x = 0; x < cols; ++x)
      {
        row[x][0] = static_cast<unsigned char>((x * 17 + y * 29 + 3) & 255);
        row[x][1] = static_cast<unsigned char>((x * 7 + y * 13 + 91) & 255);
        row[x][2] = static_cast<unsigned char>((x * 31 + y * 5 + 47) & 255);
      }
    }
    return image;
  }

  std::array<PipelineMode, 3> balanced_order(int iteration)
  {
    static const std::array<std::array<PipelineMode, 3>, 6> orders = {{
        {{PipelineMode::Baseline, PipelineMode::CudaStream, PipelineMode::Optimized}},
        {{PipelineMode::Optimized, PipelineMode::Baseline, PipelineMode::CudaStream}},
        {{PipelineMode::CudaStream, PipelineMode::Baseline, PipelineMode::Optimized}},
        {{PipelineMode::Optimized, PipelineMode::CudaStream, PipelineMode::Baseline}},
        {{PipelineMode::Baseline, PipelineMode::Optimized, PipelineMode::CudaStream}},
        {{PipelineMode::CudaStream, PipelineMode::Optimized, PipelineMode::Baseline}}
    }};
    return orders[static_cast<std::size_t>(iteration) % orders.size()];
  }

  void run_mode(Detector &detector, const cv::Mat &image, PipelineMode mode,
                std::vector<lite::types::Boxf> &baseline_boxes,
                std::vector<lite::types::Boxf> &stream_boxes,
                std::vector<lite::types::Boxf> &optimized_boxes,
                Timing *baseline_timing, Timing *stream_timing,
                Timing *optimized_timing)
  {
    if (mode == PipelineMode::Baseline)
      run_once(detector, image, mode, baseline_boxes, baseline_timing);
    else if (mode == PipelineMode::CudaStream)
      run_once(detector, image, mode, stream_boxes, stream_timing);
    else
      run_once(detector, image, mode, optimized_boxes, optimized_timing);
  }
}
#endif

int main(int argc, char *argv[])
{
#ifndef ENABLE_TENSORRT
  std::cerr << "This benchmark requires ENABLE_TENSORRT=ON" << std::endl;
  return EXIT_FAILURE;
#else
  if (argc < 3 || argc > 6)
  {
    std::cerr << "Usage: " << argv[0]
              << " <engine_path> <image_path> [warmup=20] [iterations=200]"
              << " [reference_engine_path]"
              << std::endl;
    return EXIT_FAILURE;
  }

  try
  {
    const std::string engine_path = argv[1];
    const std::string image_path = argv[2];
    const int warmup = argc >= 4 ? positive_integer(argv[3], "warmup") : 20;
    const int iterations = argc >= 5 ? positive_integer(argv[4], "iterations") : 200;
    const std::string reference_engine_path = argc >= 6 ? argv[5] : "";

    cv::Mat image = cv::imread(image_path);
    if (image.empty())
      throw std::runtime_error("Failed to read benchmark image: " + image_path);

    const CudaMemorySnapshot memory_before_detector = cuda_memory_snapshot();
    Detector detector(engine_path);
    const CudaMemorySnapshot memory_after_detector = cuda_memory_snapshot();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "preprocess_validation,layout,elements,mismatched_exact,mean_abs_error,p99_abs_error,max_abs_error\n";
    require_exact_preprocess(detector, "benchmark_image", image);
    require_exact_preprocess(detector, "upscale_319x511", make_pattern(319, 511));
    require_exact_preprocess(detector, "downscale_721x1283", make_pattern(721, 1283));
    require_exact_preprocess(detector, "square_640x640", make_pattern(640, 640));
    require_exact_preprocess(detector, "single_row_1x97", make_pattern(1, 97));

    std::vector<lite::types::Boxf> reference_boxes;
    std::vector<lite::types::Boxf> baseline_boxes;
    std::vector<lite::types::Boxf> stream_boxes;
    std::vector<lite::types::Boxf> optimized_boxes;

    Timing graph_prepare_timing;
    run_once(detector, image, PipelineMode::Optimized, optimized_boxes,
             &graph_prepare_timing);
    if (!graph_prepare_timing.graph_replayed || graph_prepare_timing.graph_fallback)
      throw std::runtime_error("CUDA Graph capture/replay is unavailable for benchmark input");
    const CudaMemorySnapshot memory_after_initial_graph = cuda_memory_snapshot();

    const int fallback_rows = image.rows == 319 ? 320 : 319;
    const int fallback_cols = image.cols == 511 ? 512 : 511;
    cv::Mat fallback_image = make_pattern(fallback_rows, fallback_cols);
    std::vector<lite::types::Boxf> fallback_stream_boxes;
    std::vector<lite::types::Boxf> fallback_graph_boxes;
    Timing fallback_stream_timing;
    Timing fallback_graph_timing;
    run_once(detector, fallback_image, PipelineMode::CudaStream,
             fallback_stream_boxes, &fallback_stream_timing);
    run_once(detector, fallback_image, PipelineMode::Optimized,
             fallback_graph_boxes, &fallback_graph_timing);
    if (!fallback_graph_timing.graph_fallback || fallback_graph_timing.graph_replayed ||
        !same_boxes(fallback_stream_boxes, fallback_graph_boxes))
      throw std::runtime_error("CUDA Graph shape-mismatch fallback validation failed");

    run_once(detector, image, PipelineMode::Optimized, optimized_boxes,
             &graph_prepare_timing);
    if (!graph_prepare_timing.graph_replayed || graph_prepare_timing.graph_fallback)
      throw std::runtime_error("CUDA Graph was not reusable after shape fallback");

    cv::Mat growth_image = make_pattern(2048, 2048);
    std::vector<lite::types::Boxf> growth_baseline_boxes;
    std::vector<lite::types::Boxf> growth_stream_boxes;
    run_once(detector, growth_image, PipelineMode::Baseline,
             growth_baseline_boxes, nullptr);
    run_once(detector, growth_image, PipelineMode::CudaStream,
             growth_stream_boxes, nullptr);
    if (!same_boxes(growth_baseline_boxes, growth_stream_boxes))
      throw std::runtime_error("CUDA Graph staging-growth correctness validation failed");

    run_once(detector, image, PipelineMode::Optimized, optimized_boxes,
             &graph_prepare_timing);
    if (!graph_prepare_timing.graph_replayed || graph_prepare_timing.graph_fallback)
      throw std::runtime_error("CUDA Graph did not recapture after staging growth");

    for (int i = 0; i < warmup; ++i)
    {
      for (const PipelineMode mode : balanced_order(i))
        run_mode(detector, image, mode, baseline_boxes, stream_boxes,
                 optimized_boxes, nullptr, nullptr, nullptr);
      if (!same_boxes(baseline_boxes, stream_boxes) ||
          !same_boxes(baseline_boxes, optimized_boxes))
        throw std::runtime_error("Baseline, CUDA stream, and CUDA Graph detections differ during warmup");
      reference_boxes = baseline_boxes;
    }
    // Includes the 2048x2048 staging-growth validation and the recaptured graph.
    const CudaMemorySnapshot memory_after_preflight = cuda_memory_snapshot();

    Samples baseline;
    Samples stream_samples;
    Samples optimized;
    baseline.reserve(static_cast<std::size_t>(iterations));
    stream_samples.reserve(static_cast<std::size_t>(iterations));
    optimized.reserve(static_cast<std::size_t>(iterations));
    int measured_graph_replays = 0;
    int measured_graph_fallbacks = 0;

    for (int i = 0; i < iterations; ++i)
    {
      Timing baseline_timing;
      Timing stream_timing;
      Timing optimized_timing;
      for (const PipelineMode mode : balanced_order(i))
        run_mode(detector, image, mode, baseline_boxes, stream_boxes,
                 optimized_boxes, &baseline_timing, &stream_timing,
                 &optimized_timing);

      if (!same_boxes(reference_boxes, baseline_boxes) ||
          !same_boxes(reference_boxes, stream_boxes) ||
          !same_boxes(reference_boxes, optimized_boxes))
        throw std::runtime_error("Detection output changed during three-way benchmark");

      if (!optimized_timing.graph_replayed || optimized_timing.graph_fallback)
        throw std::runtime_error("Optimized sample did not use CUDA Graph replay");
      ++measured_graph_replays;
      if (optimized_timing.graph_fallback) ++measured_graph_fallbacks;

      baseline.append(baseline_timing);
      stream_samples.append(stream_timing);
      optimized.append(optimized_timing);
    }

    const double baseline_total = summarize(baseline.total).mean;
    const double stream_samples_total = summarize(stream_samples.total).mean;
    const double optimized_total = summarize(optimized.total).mean;
    const double baseline_fps = 1000.0 / baseline_total;
    const double stream_samples_fps = 1000.0 / stream_samples_total;
    const double optimized_fps = 1000.0 / optimized_total;

    PrecisionComparison precision_comparison;
    if (!reference_engine_path.empty())
    {
      Detector reference_detector(reference_engine_path);
      std::vector<lite::types::Boxf> cross_precision_reference_boxes;
      run_once(reference_detector, image, PipelineMode::Baseline,
               cross_precision_reference_boxes, nullptr);
      precision_comparison = compare_precision_outputs(
          reference_boxes, cross_precision_reference_boxes);
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "engine," << engine_path << '\n';
    std::cout << "image," << image_path << '\n';
    std::cout << "cuda_total_bytes," << memory_before_detector.total_bytes << '\n';
    std::cout << "cuda_free_before_detector_bytes," << memory_before_detector.free_bytes << '\n';
    std::cout << "cuda_free_after_detector_load_bytes," << memory_after_detector.free_bytes << '\n';
    std::cout << "cuda_free_after_initial_graph_bytes," << memory_after_initial_graph.free_bytes << '\n';
    std::cout << "cuda_free_after_preflight_bytes," << memory_after_preflight.free_bytes << '\n';
    std::cout << "cuda_detector_load_consumed_bytes,"
              << consumed_bytes(memory_before_detector, memory_after_detector) << '\n';
    std::cout << "cuda_preprocess_and_initial_graph_consumed_bytes,"
              << consumed_bytes(memory_after_detector, memory_after_initial_graph) << '\n';
    std::cout << "cuda_preflight_consumed_bytes,"
              << consumed_bytes(memory_before_detector, memory_after_preflight) << '\n';
    std::cout << "cuda_preflight_includes_2048_staging,true\n";
    std::cout << "warmup_sets," << warmup << '\n';
    std::cout << "measured_sets," << iterations << '\n';
    std::cout << "boxes," << reference_boxes.size() << '\n';
    std::cout << "consistency_checks," << iterations << '\n';
    std::cout << "consistency,passed\n";
    std::cout << "graph_replays," << measured_graph_replays << '\n';
    std::cout << "graph_fallbacks," << measured_graph_fallbacks << '\n';
    std::cout << "shape_fallback_validation,passed\n";
    std::cout << "staging_growth_validation,passed\n";
    if (!reference_engine_path.empty())
    {
      std::cout << "precision_reference_engine," << reference_engine_path << '\n';
      std::cout << "precision_comparison_scope,postprocessed_single_image_smoke_test\n";
      std::cout << "precision_match_iou_threshold,0.500000\n";
      std::cout << "precision_primary_boxes," << precision_comparison.primary_boxes << '\n';
      std::cout << "precision_reference_boxes," << precision_comparison.reference_boxes << '\n';
      std::cout << "precision_matched_boxes," << precision_comparison.matched << '\n';
      std::cout << "precision_unmatched_primary_boxes,"
                << precision_comparison.unmatched_primary << '\n';
      std::cout << "precision_unmatched_reference_boxes,"
                << precision_comparison.unmatched_reference << '\n';
      std::cout << "precision_same_index_label_mismatches,"
                << precision_comparison.same_index_label_mismatches << '\n';
      std::cout << "precision_min_iou," << precision_comparison.minimum_iou << '\n';
      std::cout << "precision_mean_iou," << precision_comparison.mean_iou << '\n';
      std::cout << "precision_max_score_abs_error,"
                << precision_comparison.maximum_score_abs_error << '\n';
      std::cout << "precision_max_coordinate_abs_error,"
                << precision_comparison.maximum_coordinate_abs_error << '\n';
    }
    std::cout << "mode,metric,mean_ms,p50_ms,p95_ms,min_ms\n";
    print_samples("baseline", baseline);
    print_samples("previous_cuda_stream", stream_samples);
    print_samples("optimized_cuda_graph", optimized);
    std::cout << "mode,throughput_fps\n";
    std::cout << "baseline," << baseline_fps << '\n';
    std::cout << "previous_cuda_stream," << stream_samples_fps << '\n';
    std::cout << "optimized_cuda_graph," << optimized_fps << '\n';
    std::cout << "comparison,metric,reference_mean_ms,optimized_mean_ms,delta_ms,delta_percent\n";
    print_comparisons("from_baseline", baseline, optimized);
    print_comparisons("from_previous_cuda_stream", stream_samples, optimized);
    std::cout << "speedup_from_baseline," << baseline_total / optimized_total << '\n';
    std::cout << "speedup_from_previous_cuda_stream," << stream_samples_total / optimized_total << '\n';
    std::cout << "fps_delta_from_baseline," << optimized_fps - baseline_fps << '\n';
    std::cout << "fps_delta_from_previous_cuda_stream," << optimized_fps - stream_samples_fps << '\n';
    return EXIT_SUCCESS;
  }
  catch (const std::exception &error)
  {
    std::cerr << "Benchmark failed: " << error.what() << std::endl;
    return EXIT_FAILURE;
  }
#endif
}
