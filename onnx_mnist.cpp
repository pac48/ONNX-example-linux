#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// Taken from https://github.com/microsoft/onnxruntime-inference-examples/ C++ example
template<typename T>
static void softmax(T &input) {
  float rowmax = *std::max_element(input.begin(), input.end());
  std::vector<float> y(input.size());
  float sum = 0.0f;
  for (size_t i = 0; i != input.size(); ++i) {
    sum += y[i] = std::exp(input[i] - rowmax);
  }
  for (size_t i = 0; i != input.size(); ++i) {
    input[i] = y[i] / sum;
  }
}

// Taken from https://github.com/microsoft/onnxruntime-inference-examples/ C++ example
struct MNIST {
  MNIST() {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
                                                    input_shape_.data(), input_shape_.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
                                                     output_shape_.data(), output_shape_.size());
  }

  std::ptrdiff_t Run() {
    const char *input_names[] = {"Input3"};
    const char *output_names[] = {"Plus214_Output_0"};

    Ort::RunOptions run_options;
    session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    softmax(results_);
    result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
    return result_;
  }

  static constexpr const int width_ = 28;
  static constexpr const int height_ = 28;

  std::array<float, width_ * height_> input_image_{};
  std::array<float, 10> results_{};
  int64_t result_{0};

private:
  Ort::Env env;
  Ort::Session session_{env, "mnist.onnx", Ort::SessionOptions{nullptr}};

  Ort::Value input_tensor_{nullptr};
  std::array<int64_t, 4> input_shape_{1, 1, width_, height_};

  Ort::Value output_tensor_{nullptr};
  std::array<int64_t, 2> output_shape_{1, 10};
};

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Incorrect number of arguments provided.\n"
              << "Usage: " << argv[0] << " <path_to_input_image>\n"
              << "Please provide the file path to an input image as the argument.\n";
    return 1;
  }
  auto file_path = std::string(argv[1]);

  // load image as gray scale using openCV
  cv::Mat img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
  cv::Mat img_float;
  img.convertTo(img_float, CV_32F);

  // create float array to store loaded image in
  std::array<float, 28 * 28> img_vector = {};
  img_float.reshape(0, 1).copyTo(img_vector);
  std::transform(img_vector.begin(), img_vector.end(), img_vector.begin(), [](float val) { return val / 255; });

  // create onnx struct wrapper, set input data to image, and run the model
  MNIST mnist;
  mnist.input_image_ = img_vector;
  mnist.Run();

  // print the prediction with the highest probability
  std::cout << "The predicted number is: " << mnist.result_ << "\n";

  return 0;
}
