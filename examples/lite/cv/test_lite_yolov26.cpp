//
// Created by zhuohaoyang on 2026/8/26.
//

#include "lite/lite.h"

#include <iomanip>

static void print_timing(const std::string &backend,
                         const lite::types::InferenceTiming &timing)
{
  std::cout << std::fixed << std::setprecision(3)
            << backend << " Preprocess Time: " << timing.preprocess_ms << " ms\n"
            << backend << " Inference Time: " << timing.inference_ms << " ms\n"
            << backend << " Postprocess Time: " << timing.postprocess_ms << " ms\n"
            << backend << " Total Time: " << timing.total_ms() << " ms" << std::endl;
}

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolov26n-640x640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_1.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov26_1.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::detection::YoloV26 *yolov26 =
      new lite::cv::detection::YoloV26(onnx_path); // default

  cv::Mat img_bgr = cv::imread(test_img_path);
  std::vector<lite::types::Boxf> warmup_boxes;
  yolov26->detect(img_bgr, warmup_boxes);

  std::vector<lite::types::Boxf> detected_boxes;
  lite::types::InferenceTiming timing;
  yolov26->detect_with_timing(img_bgr, detected_boxes, 0.25f, 100, &timing);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;
  print_timing("Default", timing);

  delete yolov26;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolov26n-640x640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_1.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov26_2.jpg";

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::detection::YoloV26 *yolov26 =
      new lite::onnxruntime::cv::detection::YoloV26(onnx_path);

  cv::Mat img_bgr = cv::imread(test_img_path);
  std::vector<lite::types::Boxf> warmup_boxes;
  yolov26->detect(img_bgr, warmup_boxes);

  std::vector<lite::types::Boxf> detected_boxes;
  lite::types::InferenceTiming timing;
  yolov26->detect_with_timing(img_bgr, detected_boxes, 0.25f, 100, &timing);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Boxes Num: " << detected_boxes.size() << std::endl;
  print_timing("ONNXRuntime", timing);

  delete yolov26;
#endif
}

static void test_tensorrt()
{
#ifdef ENABLE_TENSORRT
  std::string engine_path = "../../../examples/hub/trt/yolov26n_fp32.engine";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_1.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov26_1_trt.jpg";

  // 3. Test Specific Engine TensorRT
  lite::trt::cv::detection::YOLOV26 *yolov26 =
      new lite::trt::cv::detection::YOLOV26(engine_path);

  cv::Mat img_bgr = cv::imread(test_img_path);
  std::vector<lite::types::Boxf> warmup_boxes;
  yolov26->detect(img_bgr, warmup_boxes);

  std::vector<lite::types::Boxf> detected_boxes;
  lite::types::InferenceTiming timing;
  yolov26->detect_with_timing(img_bgr, detected_boxes, 0.25f, 100, &timing);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TensorRT Version Detected Boxes Num: " << detected_boxes.size() << std::endl;
  print_timing("TensorRT", timing);

  delete yolov26;
#endif
}

static void test_lite()
{
  test_default();
  test_onnxruntime();
  test_tensorrt();
}

int main(__unused int argc, __unused char *argv[])
{
  test_lite();
  return 0;
}
