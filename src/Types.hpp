#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace meme {

inline constexpr std::size_t kFeatureDim = 54;
inline constexpr std::size_t kNumKeypoints = 17;
inline constexpr std::size_t kNumMemeClasses = 9;

struct Keypoint {
    float x{0.0f};
    float y{0.0f};
    float confidence{1.0f};
};

using PoseSkeleton = std::array<Keypoint, kNumKeypoints>;
using FeatureVector = std::array<float, kFeatureDim>;

struct MemeClassInfo {
    int index{0};
    std::string id;
    std::string title;
    std::string subtitle;
    std::string hint;
    std::string accent_hex;
    std::string image_path;
};

struct PredictionResult {
    int class_index{8};
    float confidence{0.0f};
    std::array<float, kNumMemeClasses> probabilities{};
    std::string backend_used{"GPU"};
};

} // namespace meme
