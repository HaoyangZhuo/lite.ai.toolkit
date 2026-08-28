//
// Created by zhuohaoyang on 2026/8/27.
//

#ifndef LITE_AI_TOOLKIT_TRT_YOLOV26_H
#define LITE_AI_TOOLKIT_TRT_YOLOV26_H

#include "lite/trt/core/trt_core.h"

namespace trtcv
{
  class LITE_EXPORTS TRTYoloV26 : public BasicTRTHandler
  {
  public:
    struct Timing
    {
      double preprocess_ms = 0.0;
      double h2d_ms = 0.0;
      double inference_ms = 0.0;
      double d2h_ms = 0.0;
      double backend_wall_ms = 0.0;
      double postprocess_ms = 0.0;
      double total_ms = 0.0;

      double gpu_pipeline_ms() const
      {
        return h2d_ms + inference_ms + d2h_ms;
      }
    };

    explicit TRTYoloV26(const std::string &_trt_model_path,
                       unsigned int _num_threads = 1);

    ~TRTYoloV26() override;

  private:
    const char *coco_class_names[80] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    typedef struct
    {
      float scale;
      int left;
      int top;
    } ScaleParams;

    cudaEvent_t timing_events[4] = {nullptr, nullptr, nullptr, nullptr};

  private:
    void letterbox(const cv::Mat &mat, cv::Mat &mat_rs,
                   int target_height, int target_width,
                   ScaleParams &scale_params);

    void detect_impl(const cv::Mat &mat,
                     std::vector<types::Boxf> &detected_boxes,
                     float score_threshold, unsigned int topk,
                     Timing *timing);

    void generate_bboxes(const ScaleParams &scale_params,
                         std::vector<types::Boxf> &detected_boxes,
                         const float *output, std::size_t num_predictions,
                         float score_threshold, unsigned int topk,
                         int img_height, int img_width);

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.25f, unsigned int topk = 100);

    void detect_with_timing(const cv::Mat &mat,
                            std::vector<types::Boxf> &detected_boxes,
                            Timing &timing,
                            float score_threshold = 0.25f,
                            unsigned int topk = 100);
  };
}

#endif //LITE_AI_TOOLKIT_TRT_YOLOV26_H
