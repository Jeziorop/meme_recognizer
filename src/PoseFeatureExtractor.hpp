#pragma once

#include "Types.hpp"

namespace meme {

class PoseFeatureExtractor {
public:
    [[nodiscard]] static FeatureVector extract(const PoseSkeleton& kp) noexcept;
    [[nodiscard]] static PoseSkeleton getCanonicalPose(int class_index, bool mirror = false) noexcept;
};

} // namespace meme
