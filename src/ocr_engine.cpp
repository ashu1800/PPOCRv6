#include "ppocr/ocr_engine.h"

#include "ppocr/base64.h"
#include "ppocr/logger.h"
#include "ppocr/ocr_postprocess.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <limits>
#include <sstream>
#include <stdexcept>

#if PPOCR_WITH_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#endif

#if PPOCR_WITH_NCNN
#include <net.h>
#if NCNN_VULKAN
#include <gpu.h>
#endif
#endif

namespace ppocr {
namespace {

std::filesystem::path model_dir_for(const std::filesystem::path& runtime_dir, const std::string& level) {
    return runtime_dir / "models" / level;
}

void require_file(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("required model file missing: " + path.string());
    }
}

std::vector<std::string> load_keys(const std::filesystem::path& path) {
    require_file(path);
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open keys file: " + path.string());
    }

    std::vector<std::string> keys;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        keys.push_back(line);
    }
    if (keys.empty()) {
        throw std::runtime_error("keys file is empty: " + path.string());
    }
    return keys;
}

#if PPOCR_WITH_OPENCV
cv::Mat decode_image_or_throw(const std::vector<std::uint8_t>& image_bytes) {
    if (image_bytes.empty()) {
        throw std::invalid_argument("image_base64 decoded to empty bytes");
    }
    cv::Mat buffer(1, static_cast<int>(image_bytes.size()), CV_8UC1, const_cast<std::uint8_t*>(image_bytes.data()));
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::invalid_argument("decoded bytes are not a supported image");
    }
    return image;
}
#endif

#if PPOCR_WITH_NCNN && PPOCR_WITH_OPENCV

struct DetectionObject {
    std::array<Point, 4> box{};
    float confidence = 0.0F;
};

struct RecognizedText {
    std::string text;
    float confidence = 0.0F;
};

float point_distance(const Point& lhs, const Point& rhs) {
    const float dx = lhs.x - rhs.x;
    const float dy = lhs.y - rhs.y;
    return std::sqrt(dx * dx + dy * dy);
}

cv::Point2f to_cv_point(const Point& point) {
    return {point.x, point.y};
}

Point to_point(const cv::Point2f& point) {
    return {point.x, point.y};
}

Point clamp_point_to_image(Point point, int width, int height) {
    point.x = std::clamp(point.x, 0.0F, static_cast<float>(std::max(0, width - 1)));
    point.y = std::clamp(point.y, 0.0F, static_cast<float>(std::max(0, height - 1)));
    return point;
}

std::array<Point, 4> ordered_box_from_rotated_rect(const cv::RotatedRect& rect) {
    cv::Point2f corners[4];
    rect.points(corners);
    std::array<Point, 4> box{};
    for (int i = 0; i < 4; ++i) {
        box[static_cast<std::size_t>(i)] = to_point(corners[i]);
    }
    return order_box_points(box);
}

double contour_score(const cv::Mat& probability, const std::vector<cv::Point>& contour) {
    cv::Rect rect = cv::boundingRect(contour);
    rect.x = std::max(rect.x, 0);
    rect.y = std::max(rect.y, 0);
    rect.width = std::min(rect.width, probability.cols - rect.x);
    rect.height = std::min(rect.height, probability.rows - rect.y);
    if (rect.width <= 0 || rect.height <= 0) {
        return 0.0;
    }

    cv::Mat mask = cv::Mat::zeros(rect.height, rect.width, CV_8UC1);
    std::vector<cv::Point> roi_contour;
    roi_contour.reserve(contour.size());
    for (const auto& point : contour) {
        roi_contour.emplace_back(point.x - rect.x, point.y - rect.y);
    }
    std::vector<std::vector<cv::Point>> contours{roi_contour};
    cv::fillPoly(mask, contours, cv::Scalar(255));
    return cv::mean(probability(rect), mask).val[0] / 255.0;
}

cv::Mat crop_text_region(const cv::Mat& image, const DetectionObject& object) {
    const auto crop_box = expand_text_box(object.box, static_cast<float>(image.cols), static_cast<float>(image.rows));
    const float width = std::max(point_distance(crop_box[0], crop_box[1]), point_distance(crop_box[3], crop_box[2]));
    const float height = std::max(point_distance(crop_box[0], crop_box[3]), point_distance(crop_box[1], crop_box[2]));
    const int target_height = 48;
    const int target_width = std::max(8, static_cast<int>(std::round(width * target_height / std::max(height, 1.0F))));

    std::vector<cv::Point2f> src_pts{
        to_cv_point(crop_box[0]),
        to_cv_point(crop_box[1]),
        to_cv_point(crop_box[2]),
        to_cv_point(crop_box[3]),
    };
    std::vector<cv::Point2f> dst_pts{
        cv::Point2f(0.0F, 0.0F),
        cv::Point2f(static_cast<float>(target_width - 1), 0.0F),
        cv::Point2f(static_cast<float>(target_width - 1), static_cast<float>(target_height - 1)),
        cv::Point2f(0.0F, static_cast<float>(target_height - 1)),
    };

    cv::Mat transform = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat crop;
    cv::warpPerspective(image, crop, transform, cv::Size(target_width, target_height), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return crop;
}

std::vector<DetectionObject> split_wide_text_lines(const cv::Mat& image) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    const int kernel_w = std::max(25, image.cols / 30);
    const int kernel_h = std::max(3, image.rows / 80);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_w, kernel_h));
    cv::Mat merged;
    cv::dilate(binary, merged, kernel);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(merged, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<DetectionObject> objects;
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        if (rect.width < image.cols / 30 || rect.height < 6) {
            continue;
        }
        rect.x = std::max(0, rect.x - 4);
        rect.y = std::max(0, rect.y - 4);
        rect.width = std::min(image.cols - rect.x, rect.width + 8);
        rect.height = std::min(image.rows - rect.y, rect.height + 8);

        std::array<Point, 4> box{
            Point{static_cast<float>(rect.x), static_cast<float>(rect.y)},
            Point{static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y)},
            Point{static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y + rect.height)},
            Point{static_cast<float>(rect.x), static_cast<float>(rect.y + rect.height)},
        };
        objects.push_back({order_box_points(box), 0.35F});
    }
    return objects;
}

#endif

class BasicOcrEngine final : public OcrEngine {
public:
    BasicOcrEngine(const Config& config, const std::filesystem::path& runtime_dir) {
        const auto dir = model_dir_for(runtime_dir, config.model_level);
        require_file(dir / "det.ncnn.param");
        require_file(dir / "det.ncnn.bin");
        require_file(dir / "rec.ncnn.param");
        require_file(dir / "rec.ncnn.bin");
        keys_ = load_keys(dir / "keys.txt");

        runtime_.inference_mode = "CPU";
        runtime_.gpu_enabled = false;
        runtime_.fallback_reason = "ncnn development package was not available at build time";

        log::warn("ncnn backend is not compiled in; OCR endpoint will return 503 until ncnn is installed and the project is rebuilt.");
    }

    OcrResult recognize(const std::vector<std::uint8_t>& image_bytes) override {
#if PPOCR_WITH_OPENCV
        (void)decode_image_or_throw(image_bytes);
#else
        if (image_bytes.empty()) {
            throw std::invalid_argument("image_base64 decoded to empty bytes");
        }
#endif
        throw std::runtime_error("OCR backend is unavailable: rebuild with ncnn to enable PPOCR inference");
    }

    const RuntimeInfo& runtime_info() const override {
        return runtime_;
    }

private:
    RuntimeInfo runtime_;
    std::vector<std::string> keys_;
};

#if PPOCR_WITH_NCNN

class NcnnOcrEngine final : public OcrEngine {
public:
    NcnnOcrEngine(const Config& config, const std::filesystem::path& runtime_dir)
        : model_dir_(model_dir_for(runtime_dir, config.model_level)),
          gpu_device_index_(config.gpu_device),
          rec_max_width_(config.rec_max_width),
          enable_orientation_retry_(config.enable_orientation_retry),
          keys_(load_keys(model_dir_ / "keys.txt")) {
        require_file(model_dir_ / "det.ncnn.param");
        require_file(model_dir_ / "det.ncnn.bin");
        require_file(model_dir_ / "rec.ncnn.param");
        require_file(model_dir_ / "rec.ncnn.bin");

        initialize_runtime(config);
        load_models();
    }

    ~NcnnOcrEngine() override {
#if NCNN_VULKAN
        if (gpu_instance_created_) {
            ncnn::destroy_gpu_instance();
        }
#endif
    }

    OcrResult recognize(const std::vector<std::uint8_t>& image_bytes) override {
#if !PPOCR_WITH_OPENCV
        (void)image_bytes;
        throw std::runtime_error("OpenCV is required for image decoding");
#else
        const cv::Mat image = decode_image_or_throw(image_bytes);
        auto detections = detect(image);
        OcrResult result;
        result.items.reserve(detections.size());
        for (const auto& detection : detections) {
            auto item = recognize_one(image, detection);
            if (!item.text.empty()) {
                result.items.push_back(std::move(item));
            }
        }
        sort_boxes_reading_order(result.items);
        for (std::size_t i = 0; i < result.items.size(); ++i) {
            if (i != 0) {
                result.full_text += "\n";
            }
            result.full_text += result.items[i].text;
        }
        return result;
#endif
    }

    const RuntimeInfo& runtime_info() const override {
        return runtime_;
    }

private:
#if PPOCR_WITH_OPENCV
    std::vector<DetectionObject> detect(const cv::Mat& image) {
        const int target_size = 960;
        const int target_stride = 32;
        int resized_w = image.cols;
        int resized_h = image.rows;
        float scale = 1.0F;

        if (std::max(resized_w, resized_h) > target_size) {
            if (resized_w > resized_h) {
                scale = static_cast<float>(target_size) / static_cast<float>(resized_w);
                resized_w = target_size;
                resized_h = static_cast<int>(std::round(static_cast<float>(resized_h) * scale));
            } else {
                scale = static_cast<float>(target_size) / static_cast<float>(resized_h);
                resized_h = target_size;
                resized_w = static_cast<int>(std::round(static_cast<float>(resized_w) * scale));
            }
        }

        ncnn::Mat input = ncnn::Mat::from_pixels_resize(
            image.data,
            ncnn::Mat::PIXEL_BGR,
            image.cols,
            image.rows,
            resized_w,
            resized_h);

        const int wpad = (resized_w + target_stride - 1) / target_stride * target_stride - resized_w;
        const int hpad = (resized_h + target_stride - 1) / target_stride * target_stride - resized_h;
        ncnn::Mat padded;
        ncnn::copy_make_border(
            input,
            padded,
            hpad / 2,
            hpad - hpad / 2,
            wpad / 2,
            wpad - wpad / 2,
            ncnn::BORDER_CONSTANT,
            114.0F);

        const float mean_vals[3] = {0.485F * 255.0F, 0.456F * 255.0F, 0.406F * 255.0F};
        const float norm_vals[3] = {1.0F / 0.229F / 255.0F, 1.0F / 0.224F / 255.0F, 1.0F / 0.225F / 255.0F};
        padded.substract_mean_normalize(mean_vals, norm_vals);

        auto extractor = det_net_.create_extractor();
        extractor.input("in0", padded);

        ncnn::Mat output;
        if (extractor.extract("out0", output) != 0) {
            throw std::runtime_error("det model inference failed");
        }

        const float denorm_vals[1] = {255.0F};
        output.substract_mean_normalize(nullptr, denorm_vals);
        cv::Mat probability(output.h, output.w, CV_8UC1);
        output.to_pixels(probability.data, ncnn::Mat::PIXEL_GRAY);
        cv::Mat bitmap;
        constexpr float threshold = 0.3F;
        cv::threshold(probability, bitmap, threshold * 255.0F, 255.0, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        if (contours.size() > 1000) {
            contours.resize(1000);
        }

        std::vector<DetectionObject> objects;
        constexpr float box_threshold = 0.35F;
        constexpr float enlarge_ratio = 1.95F;
        const float min_size = std::max(3.0F * scale, 3.0F);

        for (const auto& contour : contours) {
            if (contour.size() <= 2) {
                continue;
            }
            const double score = contour_score(probability, contour);
            if (score < box_threshold) {
                continue;
            }

            cv::RotatedRect rect = cv::minAreaRect(contour);
            if (std::max(rect.size.width, rect.size.height) < min_size) {
                continue;
            }

            int orientation = 0;
            if (rect.angle >= -30.0F && rect.angle <= 30.0F && rect.size.height > rect.size.width * 2.7F) {
                orientation = 1;
            }
            if ((rect.angle <= -60.0F || rect.angle >= 60.0F) && rect.size.width > rect.size.height * 2.7F) {
                orientation = 1;
            }
            if (rect.angle < -30.0F) {
                rect.angle += 180.0F;
            }
            if (orientation == 0 && rect.angle < 30.0F) {
                rect.angle += 90.0F;
                std::swap(rect.size.width, rect.size.height);
            }
            if (orientation == 1 && rect.angle >= 60.0F) {
                rect.angle -= 90.0F;
                std::swap(rect.size.width, rect.size.height);
            }

            rect.size.height += rect.size.width * (enlarge_ratio - 1.0F);
            rect.size.width *= enlarge_ratio;
            rect.center.x = (rect.center.x - static_cast<float>(wpad / 2)) / scale;
            rect.center.y = (rect.center.y - static_cast<float>(hpad / 2)) / scale;
            rect.size.width /= scale;
            rect.size.height /= scale;

            auto box = ordered_box_from_rotated_rect(rect);
            for (auto& point : box) {
                point = clamp_point_to_image(point, image.cols, image.rows);
            }
            objects.push_back({box, static_cast<float>(score)});
        }

        if (objects.size() == 1 && image.cols > image.rows * 3 &&
            point_distance(objects[0].box[0], objects[0].box[1]) > image.cols * 0.8F) {
            auto line_objects = split_wide_text_lines(image);
            if (line_objects.size() > 1) {
                log::info("Detect fallback: split wide screenshot into " + std::to_string(line_objects.size()) + " text lines");
                return line_objects;
            }
        }

        return objects;
    }

    OcrItem recognize_one(const cv::Mat& image, const DetectionObject& detection) {
        cv::Mat crop = crop_text_region(image, detection);
        if (crop.empty()) {
            return {};
        }
        RecognizedText decoded = recognize_text(crop);
        if (enable_orientation_retry_ && decoded.confidence < 0.75F && crop.cols > crop.rows * 1.2F) {
            cv::Mat rotated;
            cv::rotate(crop, rotated, cv::ROTATE_180);
            auto rotated_decoded = recognize_text(rotated);
            if (rotated_decoded.confidence > decoded.confidence) {
                decoded = std::move(rotated_decoded);
            }
        }

        OcrItem item;
        item.text = decoded.text;
        item.confidence = decoded.confidence <= 0.0F ? detection.confidence : decoded.confidence;
        item.box = detection.box;
        return item;
    }

    std::vector<cv::Mat> split_recognition_crop(const cv::Mat& crop) const {
        constexpr int direct_recognition_max_width = 4096;
        if (crop.cols <= direct_recognition_max_width) {
            return {crop};
        }

        cv::Mat gray;
        cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
        cv::Mat binary;
        cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

        std::vector<int> column_ink(static_cast<std::size_t>(binary.cols), 0);
        for (int x = 0; x < binary.cols; ++x) {
            column_ink[static_cast<std::size_t>(x)] = cv::countNonZero(binary.col(x));
        }

        std::vector<cv::Mat> segments;
        constexpr int overlap = 32;
        int start = 0;
        while (crop.cols - start > static_cast<int>(rec_max_width_)) {
            const int search_start = std::min(crop.cols - 1, start + static_cast<int>(rec_max_width_ * 3 / 4));
            const int search_end = std::min(crop.cols - 1, start + static_cast<int>(rec_max_width_));
            int cut = search_end;
            int best_score = std::numeric_limits<int>::max();
            for (int x = search_start; x <= search_end; ++x) {
                int score = 0;
                for (int offset = -2; offset <= 2; ++offset) {
                    const int sample_x = std::clamp(x + offset, 0, binary.cols - 1);
                    score += column_ink[static_cast<std::size_t>(sample_x)];
                }
                if (score < best_score) {
                    best_score = score;
                    cut = x;
                }
            }

            if (cut - start < static_cast<int>(rec_max_width_ / 2)) {
                cut = std::min(crop.cols, start + static_cast<int>(rec_max_width_));
            }
            segments.push_back(crop(cv::Rect(start, 0, cut - start, crop.rows)).clone());
            start = std::max(cut - overlap, start + 1);
        }
        if (start < crop.cols) {
            segments.push_back(crop(cv::Rect(start, 0, crop.cols - start, crop.rows)).clone());
        }
        for (auto& segment : segments) {
            if (segment.cols != static_cast<int>(rec_max_width_)) {
                cv::Mat resized;
                cv::resize(segment, resized, cv::Size(static_cast<int>(rec_max_width_), segment.rows), 0.0, 0.0, cv::INTER_LINEAR);
                segment = std::move(resized);
            }
        }
        return segments;
    }

    RecognizedText recognize_text(const cv::Mat& crop) {
        auto segments = split_recognition_crop(crop);
        if (segments.size() == 1) {
            return recognize_segment(segments[0]);
        }

        std::vector<std::string> texts;
        texts.reserve(segments.size());
        float confidence_sum = 0.0F;
        int confidence_count = 0;
        for (const auto& segment : segments) {
            auto decoded = recognize_segment(segment);
            if (!decoded.text.empty()) {
                texts.push_back(std::move(decoded.text));
                confidence_sum += decoded.confidence;
                ++confidence_count;
            }
        }

        return {
            merge_text_segments(texts),
            confidence_count == 0 ? 0.0F : confidence_sum / static_cast<float>(confidence_count),
        };
    }

    RecognizedText recognize_segment(const cv::Mat& crop) {
        if (crop.empty()) {
            return {};
        }
        ncnn::Mat input = ncnn::Mat::from_pixels(crop.data, ncnn::Mat::PIXEL_BGR, crop.cols, crop.rows);
        const float mean_vals[3] = {127.5F, 127.5F, 127.5F};
        const float norm_vals[3] = {1.0F / 127.5F, 1.0F / 127.5F, 1.0F / 127.5F};
        input.substract_mean_normalize(mean_vals, norm_vals);

        auto extractor = rec_net_.create_extractor();
        extractor.input("in0", input);

        ncnn::Mat output;
        if (extractor.extract("out0", output) != 0) {
            throw std::runtime_error("rec model inference failed");
        }

        std::vector<int> indices;
        std::vector<float> probabilities;
        indices.reserve(static_cast<std::size_t>(output.h));
        probabilities.reserve(static_cast<std::size_t>(output.h));

        for (int row = 0; row < output.h; ++row) {
            const float* values = output.row(row);
            int best_index = 0;
            float best_score = values[0];
            for (int col = 1; col < output.w; ++col) {
                if (values[col] > best_score) {
                    best_score = values[col];
                    best_index = col;
                }
            }
            indices.push_back(best_index);
            probabilities.push_back(best_score);
        }

        const auto decoded = ctc_decode(indices, probabilities, keys_);
        return {decoded.text, decoded.confidence};
    }
#endif

    void initialize_runtime(const Config& config) {
        opt_.num_threads = static_cast<int>(std::max<std::size_t>(1, config.max_concurrent_requests));
        opt_.use_fp16_packed = false;
        opt_.use_fp16_storage = false;
        opt_.use_fp16_arithmetic = false;

#if NCNN_VULKAN
        if (config.prefer_gpu) {
            try {
                ncnn::create_gpu_instance();
                gpu_instance_created_ = true;
                const int gpu_count = ncnn::get_gpu_count();
                if (gpu_count <= 0) {
                    runtime_.fallback_reason = "no Vulkan GPU device reported by ncnn";
                } else if (config.gpu_device >= gpu_count) {
                    runtime_.fallback_reason = "configured gpu_device is out of range";
                } else {
                    opt_.use_vulkan_compute = true;
                    opt_.use_bf16_storage = false;
                    runtime_.gpu_enabled = true;
                    runtime_.inference_mode = "GPU(Vulkan)";
                    runtime_.gpu_device_name = ncnn::get_gpu_info(config.gpu_device).device_name();
                    return;
                }
            } catch (const std::exception& ex) {
                runtime_.fallback_reason = ex.what();
            } catch (...) {
                runtime_.fallback_reason = "unknown Vulkan initialization failure";
            }
        } else {
            runtime_.fallback_reason = "prefer_gpu is false";
        }
#else
        runtime_.fallback_reason = "ncnn was built without Vulkan support";
#endif
        opt_.use_vulkan_compute = false;
        runtime_.gpu_enabled = false;
        runtime_.inference_mode = "CPU";
    }

    void load_models() {
        det_net_.opt = opt_;
        rec_net_.opt = opt_;
#if NCNN_VULKAN
        if (runtime_.gpu_enabled) {
            det_net_.set_vulkan_device(gpu_device_index_);
            rec_net_.set_vulkan_device(gpu_device_index_);
        }
#endif

        if (try_load_models()) {
            return;
        }

        if (!runtime_.gpu_enabled) {
            throw std::runtime_error("failed to load ncnn models");
        }

        log::warn("GPU model load failed; retrying model load with CPU backend.");
        runtime_.gpu_enabled = false;
        runtime_.inference_mode = "CPU";
        runtime_.fallback_reason = "model load failed with Vulkan backend";
        opt_.use_vulkan_compute = false;
        det_net_.clear();
        rec_net_.clear();
        det_net_.opt = opt_;
        rec_net_.opt = opt_;

        if (!try_load_models()) {
            throw std::runtime_error("failed to load ncnn models after CPU fallback");
        }
    }

    bool try_load_models() {
        return det_net_.load_param((model_dir_ / "det.ncnn.param").string().c_str()) == 0 &&
               det_net_.load_model((model_dir_ / "det.ncnn.bin").string().c_str()) == 0 &&
               rec_net_.load_param((model_dir_ / "rec.ncnn.param").string().c_str()) == 0 &&
               rec_net_.load_model((model_dir_ / "rec.ncnn.bin").string().c_str()) == 0;
    }

    std::filesystem::path model_dir_;
    RuntimeInfo runtime_;
    int gpu_device_index_ = 0;
    std::size_t rec_max_width_ = 960;
    bool enable_orientation_retry_ = true;
    std::vector<std::string> keys_;
    ncnn::Option opt_;
    ncnn::Net det_net_;
    ncnn::Net rec_net_;
    bool gpu_instance_created_ = false;
};

#endif

} // namespace

std::shared_ptr<OcrEngine> create_ocr_engine(const Config& config, const std::filesystem::path& runtime_dir) {
#if PPOCR_WITH_NCNN
    return std::make_shared<NcnnOcrEngine>(config, runtime_dir);
#else
    return std::make_shared<BasicOcrEngine>(config, runtime_dir);
#endif
}

} // namespace ppocr
