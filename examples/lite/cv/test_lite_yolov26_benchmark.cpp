//
// YOLOv26 TensorRT latency benchmark.
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
  struct Statistics
  {
    double mean = 0.0;
    double minimum = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
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

  void print_metric(const char *name, const std::vector<double> &samples)
  {
    const Statistics stats = summarize(samples);
    std::cout << name << ',' << stats.mean << ',' << stats.p50 << ','
              << stats.p95 << ',' << stats.minimum << '\n';
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

    lite::trt::cv::detection::YOLOV26 detector(engine_path);
    std::vector<lite::types::Boxf> reference_boxes;
    for (int i = 0; i < warmup; ++i)
      detector.detect(image, reference_boxes);

    using Timing = lite::trt::cv::detection::YOLOV26::Timing;
    std::vector<double> preprocess;
    std::vector<double> h2d;
    std::vector<double> inference;
    std::vector<double> d2h;
    std::vector<double> gpu_pipeline;
    std::vector<double> backend_wall;
    std::vector<double> postprocess;
    std::vector<double> total;
    preprocess.reserve(iterations);
    h2d.reserve(iterations);
    inference.reserve(iterations);
    d2h.reserve(iterations);
    gpu_pipeline.reserve(iterations);
    backend_wall.reserve(iterations);
    postprocess.reserve(iterations);
    total.reserve(iterations);

    for (int i = 0; i < iterations; ++i)
    {
      Timing timing;
      std::vector<lite::types::Boxf> boxes;
      detector.detect_with_timing(image, boxes, timing);
      if (!same_boxes(reference_boxes, boxes))
        throw std::runtime_error("Detection output changed during benchmark");

      preprocess.push_back(timing.preprocess_ms);
      h2d.push_back(timing.h2d_ms);
      inference.push_back(timing.inference_ms);
      d2h.push_back(timing.d2h_ms);
      gpu_pipeline.push_back(timing.gpu_pipeline_ms());
      backend_wall.push_back(timing.backend_wall_ms);
      postprocess.push_back(timing.postprocess_ms);
      total.push_back(timing.total_ms);
    }

    const Statistics total_stats = summarize(total);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "engine," << engine_path << '\n';
    std::cout << "image," << image_path << '\n';
    std::cout << "warmup," << warmup << '\n';
    std::cout << "iterations," << iterations << '\n';
    std::cout << "boxes," << reference_boxes.size() << '\n';
    std::cout << "metric,mean_ms,p50_ms,p95_ms,min_ms\n";
    print_metric("preprocess_cpu", preprocess);
    print_metric("h2d_gpu", h2d);
    print_metric("inference_gpu", inference);
    print_metric("d2h_gpu", d2h);
    print_metric("gpu_pipeline", gpu_pipeline);
    print_metric("backend_wall", backend_wall);
    print_metric("postprocess_cpu", postprocess);
    print_metric("total_wall", total);
    std::cout << "throughput_fps," << 1000.0 / total_stats.mean << '\n';
    return EXIT_SUCCESS;
  }
  catch (const std::exception &error)
  {
    std::cerr << "Benchmark failed: " << error.what() << std::endl;
    return EXIT_FAILURE;
  }
#endif
}
