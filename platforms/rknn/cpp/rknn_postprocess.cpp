#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

struct Tensor {
    py::array_t<float, py::array::c_style | py::array::forcecast> data;
    const float* p;
    ssize_t c, h, w;
    bool nhwc;

    explicit Tensor(py::handle value)
        : data(py::reinterpret_borrow<py::object>(value)), p(data.data()), c(0), h(0), w(0), nhwc(false) {
        auto b = data.request();
        if (b.ndim == 3) {
            if (b.shape[0] == 1 || b.shape[0] == 51 || b.shape[0] == 64) {
                c = b.shape[0]; h = b.shape[1]; w = b.shape[2];
            } else if (b.shape[2] == 1 || b.shape[2] == 51 || b.shape[2] == 64) {
                h = b.shape[0]; w = b.shape[1]; c = b.shape[2]; nhwc = true;
            }
        } else if (b.ndim == 4 && b.shape[0] == 1) {
            if (b.shape[1] == 1 || b.shape[1] == 51 || b.shape[1] == 64) {
                c = b.shape[1]; h = b.shape[2]; w = b.shape[3];
            } else if (b.shape[3] == 1 || b.shape[3] == 51 || b.shape[3] == 64) {
                h = b.shape[1]; w = b.shape[2]; c = b.shape[3]; nhwc = true;
            }
        }
    }

    float at(ssize_t channel, ssize_t index) const {
        return nhwc ? p[index * c + channel] : p[channel * h * w + index];
    }
};

struct Detection {
    std::array<float, 4> box;
    float score;
    std::array<std::array<float, 3>, 17> keypoints;
};

static float sigmoid(float x) {
    x = std::max(-60.0f, std::min(60.0f, x));
    return 1.0f / (1.0f + std::exp(-x));
}

static float iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.box[0], b.box[0]);
    float y1 = std::max(a.box[1], b.box[1]);
    float x2 = std::min(a.box[2], b.box[2]);
    float y2 = std::min(a.box[3], b.box[3]);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float aa = std::max(0.0f, a.box[2] - a.box[0]) * std::max(0.0f, a.box[3] - a.box[1]);
    float ba = std::max(0.0f, b.box[2] - b.box[0]) * std::max(0.0f, b.box[3] - b.box[1]);
    return inter / std::max(aa + ba - inter, 1e-9f);
}

static py::list decode_pose(py::iterable outputs, float confidence, float nms_threshold, int input_size) {
    if (input_size <= 0) throw std::invalid_argument("input_size must be positive");
    std::vector<Tensor> boxes, classes, keypoints;
    for (py::handle output : outputs) {
        Tensor tensor(output);
        if (tensor.c == 64) boxes.push_back(std::move(tensor));
        else if (tensor.c == 1) classes.push_back(std::move(tensor));
        else if (tensor.c == 51) keypoints.push_back(std::move(tensor));
    }
    auto by_width = [](const Tensor& a, const Tensor& b) { return a.w > b.w; };
    std::sort(boxes.begin(), boxes.end(), by_width);
    std::sort(classes.begin(), classes.end(), by_width);
    std::sort(keypoints.begin(), keypoints.end(), by_width);
    if (boxes.size() != classes.size() || boxes.size() != keypoints.size())
        throw std::invalid_argument("expected matched 64/1/51-channel RKNN outputs");

    std::vector<Detection> detections;
    for (size_t level = 0; level < boxes.size(); ++level) {
        const Tensor& bd = boxes[level];
        const Tensor& cl = classes[level];
        const Tensor& kp = keypoints[level];
        if (bd.h != cl.h || bd.w != cl.w || bd.h != kp.h || bd.w != kp.w)
            throw std::invalid_argument("output spatial shapes do not match");
        float min_score = 0.0f, max_score = 0.0f;
        const ssize_t count = bd.h * bd.w;
        for (ssize_t i = 0; i < count; ++i) {
            min_score = std::min(min_score, cl.at(0, i));
            max_score = std::max(max_score, cl.at(0, i));
        }
        bool logits = min_score < 0.0f || max_score > 1.0f;
        float stride = static_cast<float>(input_size) / static_cast<float>(bd.h);
        for (ssize_t i = 0; i < count; ++i) {
            float score = cl.at(0, i);
            if (logits) score = sigmoid(score);
            if (score < confidence) continue;
            float gx = static_cast<float>(i % bd.w);
            float gy = static_cast<float>(i / bd.w);
            std::array<float, 4> dist{};
            for (int side = 0; side < 4; ++side) {
                float peak = -INFINITY;
                for (int bin = 0; bin < 16; ++bin) peak = std::max(peak, bd.at(side * 16 + bin, i));
                float sum = 0.0f, weighted = 0.0f;
                for (int bin = 0; bin < 16; ++bin) {
                    float e = std::exp(bd.at(side * 16 + bin, i) - peak);
                    sum += e; weighted += e * static_cast<float>(bin);
                }
                dist[side] = weighted / sum;
            }
            Detection d;
            d.score = score;
            d.box = {{(gx + .5f - dist[0]) * stride, (gy + .5f - dist[1]) * stride,
                      (gx + .5f + dist[2]) * stride, (gy + .5f + dist[3]) * stride}};
            for (int k = 0; k < 17; ++k) {
                d.keypoints[k] = {{(kp.at(k * 3, i) * 2.0f + gx - .5f) * stride,
                                   (kp.at(k * 3 + 1, i) * 2.0f + gy - .5f) * stride,
                                   sigmoid(kp.at(k * 3 + 2, i))}};
            }
            detections.push_back(d);
        }
    }

    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });
    std::vector<Detection> keep;
    for (const Detection& candidate : detections) {
        bool suppressed = false;
        for (const Detection& accepted : keep) {
            if (iou(candidate, accepted) > nms_threshold) { suppressed = true; break; }
        }
        if (!suppressed) keep.push_back(candidate);
    }

    py::list result;
    for (const Detection& d : keep) {
        py::list box;
        for (float v : d.box) box.append(std::max(0.0f, std::min(static_cast<float>(input_size), v)));
        py::list points;
        for (const auto& p : d.keypoints) points.append(py::make_tuple(p[0], p[1], p[2]));
        py::dict item;
        item["box"] = box;
        item["score"] = d.score;
        item["keypoints"] = points;
        result.append(item);
    }
    return result;
}

PYBIND11_MODULE(rknn_postprocess, module) {
    module.doc() = "Native YOLO pose raw-head decode and NMS";
    module.def("decode_pose", &decode_pose, py::arg("outputs"), py::arg("confidence") = 0.35f,
               py::arg("nms_threshold") = 0.45f, py::arg("input_size") = 640);
}
