//
// YOLOv26 TensorRT paired latency benchmark.
//

#include "lite/lite.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ENABLE_TENSORRT
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
    print_metric(mode, "inference_gpu", samples.inference);
    print_metric(mode, "d2h_gpu", samples.d2h);
    print_metric(mode, "gpu_pipeline", samples.gpu_pipeline);
    print_metric(mode, "backend_wall", samples.backend_wall);
    print_metric(mode, "postprocess_cpu", samples.postprocess);
    print_metric(mode, "total_wall", samples.total);
  }

  void print_comparison(const char *name, const std::vector<double> &baseline,
                        const std::vector<double> &optimized)
  {
    const double baseline_mean = summarize(baseline).mean;
    const double optimized_mean = summarize(optimized).mean;
    const double delta = optimized_mean - baseline_mean;
    const double percent = baseline_mean == 0.0 ? 0.0 : delta * 100.0 / baseline_mean;
    std::cout << "delta," << name << ',' << baseline_mean << ',' << optimized_mean
              << ',' << delta << ',' << percent << '\n';
  }
}
#endif

int main(int argc, char *argv[])
{
#ifndef ENABLE_TENSORRT
  std::cerr << "This benchmark requires ENABLE_TENSORRT=ON" << std::endl;
  return EXIT_FAILURE;
#else
  if (argc < 3 || argc > 5)
  {
    std::cerr << "Usage: " << argv[0]
              << " <engine_path> <image_path> [warmup=20] [iterations=200]"
              << std::endl;
    return EXIT_FAILURE;
  }

  try
  {
    const std::string engine_path = argv[1];
    const std::string image_path = argv[2];
    const int warmup = argc >= 4 ? positive_integer(argv[3], "warmup") : 20;
    const int iterations = argc >= 5 ? positive_integer(argv[4], "iterations") : 200;

    cv::Mat image = cv::imread(image_path);
    if (image.empty())
      throw std::runtime_error("Failed to read benchmark image: " + image_path);

    Detector detector(engine_path);
    std::vector<lite::types::Boxf> reference_boxes;
    std::vector<lite::types::Boxf> baseline_boxes;
    std::vector<lite::types::Boxf> optimized_boxes;

    for (int i = 0; i < warmup; ++i)
    {
      if ((i & 1) == 0)
      {
        run_once(detector, image, PipelineMode::Baseline, baseline_boxes, nullptr);
        run_once(detector, image, PipelineMode::Optimized, optimized_boxes, nullptr);
      }
      else
      {
        run_once(detector, image, PipelineMode::Optimized, optimized_boxes, nullptr);
        run_once(detector, image, PipelineMode::Baseline, baseline_boxes, nullptr);
      }
      if (!same_boxes(baseline_boxes, optimized_boxes))
        throw std::runtime_error("Baseline and optimized detections differ during warmup");
      reference_boxes = baseline_boxes;
    }

    Samples baseline;
    Samples optimized;
    baseline.reserve(static_cast<std::size_t>(iterations));
    optimized.reserve(static_cast<std::size_t>(iterations));

    for (int i = 0; i < iterations; ++i)
    {
      Timing baseline_timing;
      Timing optimized_timing;
      if ((i & 1) == 0)
      {
        run_once(detector, image, PipelineMode::Baseline, baseline_boxes, &baseline_timing);
        run_once(detector, image, PipelineMode::Optimized, optimized_boxes, &optimized_timing);
      }
      else
      {
        run_once(detector, image, PipelineMode::Optimized, optimized_boxes, &optimized_timing);
        run_once(detector, image, PipelineMode::Baseline, baseline_boxes, &baseline_timing);
      }

      if (!same_boxes(reference_boxes, baseline_boxes) ||
          !same_boxes(reference_boxes, optimized_boxes) ||
          !same_boxes(baseline_boxes, optimized_boxes))
        throw std::runtime_error("Paired detection output changed during benchmark");

      baseline.append(baseline_timing);
      optimized.append(optimized_timing);
    }

    const double baseline_total = summarize(baseline.total).mean;
    const double optimized_total = summarize(optimized.total).mean;
    const double baseline_fps = 1000.0 / baseline_total;
    const double optimized_fps = 1000.0 / optimized_total;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "engine," << engine_path << '\n';
    std::cout << "image," << image_path << '\n';
    std::cout << "warmup_pairs," << warmup << '\n';
    std::cout << "measured_pairs," << iterations << '\n';
    std::cout << "boxes," << reference_boxes.size() << '\n';
    std::cout << "consistency_checks," << iterations << '\n';
    std::cout << "consistency,passed\n";
    std::cout << "mode,metric,mean_ms,p50_ms,p95_ms,min_ms\n";
    print_samples("baseline", baseline);
    print_samples("optimized", optimized);
    std::cout << "mode,throughput_fps\n";
    std::cout << "baseline," << baseline_fps << '\n';
    std::cout << "optimized," << optimized_fps << '\n';
    std::cout << "comparison,metric,baseline_mean_ms,optimized_mean_ms,delta_ms,delta_percent\n";
    print_comparison("preprocess_cpu", baseline.preprocess, optimized.preprocess);
    print_comparison("h2d_gpu", baseline.h2d, optimized.h2d);
    print_comparison("inference_gpu", baseline.inference, optimized.inference);
    print_comparison("d2h_gpu", baseline.d2h, optimized.d2h);
    print_comparison("gpu_pipeline", baseline.gpu_pipeline, optimized.gpu_pipeline);
    print_comparison("backend_wall", baseline.backend_wall, optimized.backend_wall);
    print_comparison("postprocess_cpu", baseline.postprocess, optimized.postprocess);
    print_comparison("total_wall", baseline.total, optimized.total);
    std::cout << "speedup," << baseline_total / optimized_total << '\n';
    std::cout << "fps_delta," << optimized_fps - baseline_fps << '\n';
    std::cout << "fps_delta_percent,"
              << (optimized_fps - baseline_fps) * 100.0 / baseline_fps << '\n';
    return EXIT_SUCCESS;
  }
  catch (const std::exception &error)
  {
    std::cerr << "Benchmark failed: " << error.what() << std::endl;
    return EXIT_FAILURE;
  }
#endif
}
