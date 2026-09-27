#include "PoseFeatureExtractor.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace meme {

namespace {

struct UnitVec2D {
    float ux{0.0f};
    float uy{0.0f};
    float norm{0.0f};
};

[[nodiscard]] inline UnitVec2D computeUnitVec(float ax, float ay, float bx, float by) noexcept {
    const float dx = bx - ax;
    const float dy = by - ay;
    const float norm = std::hypot(dx, dy);
    const float inv = 1.0f / std::max(norm, 1e-6f);
    return UnitVec2D{dx * inv, dy * inv, norm};
}

} // namespace

FeatureVector PoseFeatureExtractor::extract(const PoseSkeleton& kp) noexcept {
    FeatureVector feat{};

    const float l_sh_x = kp[5].x;
    const float l_sh_y = kp[5].y;
    const float r_sh_x = kp[6].x;
    const float r_sh_y = kp[6].y;

    const float center_x = 0.5f * (l_sh_x + r_sh_x);
    const float center_y = 0.5f * (l_sh_y + r_sh_y);
    const float shoulder_dist = std::hypot(l_sh_x - r_sh_x, l_sh_y - r_sh_y);
    const float scale = std::max(shoulder_dist, 1e-4f);

    std::array<float, 11> nx{};
    std::array<float, 11> ny{};
    for (std::size_t i = 0; i < 11; ++i) {
        nx[i] = std::clamp((kp[i].x - center_x) / scale, -4.0f, 4.0f);
        ny[i] = std::clamp((kp[i].y - center_y) / scale, -4.0f, 4.0f);
        feat[2 * i] = nx[i];
        feat[2 * i + 1] = ny[i];
    }

    constexpr std::array<std::pair<std::size_t, std::size_t>, 6> kSegments{{
        {5, 7}, {7, 9}, {6, 8}, {8, 10}, {0, 9}, {0, 10}
    }};

    std::array<UnitVec2D, 6> unit_vecs{};
    for (std::size_t idx = 0; idx < kSegments.size(); ++idx) {
        const auto& [a, b] = kSegments[idx];
        unit_vecs[idx] = computeUnitVec(nx[a], ny[a], nx[b], ny[b]);
        feat[22 + 2 * idx] = unit_vecs[idx].ux;
        feat[22 + 2 * idx + 1] = unit_vecs[idx].uy;
    }

    const UnitVec2D sh_r_to_l = computeUnitVec(nx[6], ny[6], nx[5], ny[5]);
    const UnitVec2D sh_l_to_r{-sh_r_to_l.ux, -sh_r_to_l.uy, sh_r_to_l.norm};

    const std::array<std::pair<UnitVec2D, UnitVec2D>, 4> angle_pairs{{
        {sh_r_to_l, unit_vecs[0]},
        {unit_vecs[0], unit_vecs[1]},
        {sh_l_to_r, unit_vecs[2]},
        {unit_vecs[2], unit_vecs[3]}
    }};

    for (std::size_t idx = 0; idx < angle_pairs.size(); ++idx) {
        const auto& [u1, u2] = angle_pairs[idx];
        const float cos_t = std::clamp(u1.ux * u2.ux + u1.uy * u2.uy, -1.0f, 1.0f);
        const float sin_t = std::clamp(u1.ux * u2.uy - u1.uy * u2.ux, -1.0f, 1.0f);
        feat[34 + 2 * idx] = cos_t;
        feat[34 + 2 * idx + 1] = sin_t;
    }

    feat[42] = std::hypot(nx[9] - nx[0], ny[9] - ny[0]);
    feat[43] = std::hypot(nx[10] - nx[0], ny[10] - ny[0]);
    feat[44] = std::hypot(nx[9] - nx[3], ny[9] - ny[3]);
    feat[45] = std::hypot(nx[10] - nx[4], ny[10] - ny[4]);
    feat[46] = std::hypot(nx[9] - nx[6], ny[9] - ny[6]);
    feat[47] = std::hypot(nx[10] - nx[5], ny[10] - ny[5]);
    feat[48] = std::hypot(nx[9] - nx[10], ny[9] - ny[10]);
    feat[49] = ny[9] - ny[0];
    feat[50] = ny[10] - ny[0];
    feat[51] = std::abs(nx[9] - nx[10]);
    feat[52] = 0.5f * (ny[7] + ny[8]);
    feat[53] = std::clamp((nx[9] - nx[5]) * (nx[10] - nx[6]), -4.0f, 4.0f);

    return feat;
}

PoseSkeleton PoseFeatureExtractor::getCanonicalPose(int class_index, bool mirror) noexcept {
    PoseSkeleton kp{};
    kp[0] = {320.0f, 175.0f, 1.0f};
    kp[1] = {334.0f, 162.0f, 1.0f};
    kp[2] = {306.0f, 162.0f, 1.0f};
    kp[3] = {348.0f, 170.0f, 1.0f};
    kp[4] = {292.0f, 170.0f, 1.0f};
    kp[5] = {370.0f, 250.0f, 1.0f};
    kp[6] = {270.0f, 250.0f, 1.0f};
    kp[11] = {355.0f, 410.0f, 1.0f};
    kp[12] = {285.0f, 410.0f, 1.0f};
    kp[13] = {355.0f, 465.0f, 1.0f};
    kp[14] = {285.0f, 465.0f, 1.0f};
    kp[15] = {355.0f, 478.0f, 1.0f};
    kp[16] = {285.0f, 478.0f, 1.0f};

    switch (class_index) {
        case 0:
            kp[7] = {425.0f, 265.0f, 1.0f};
            kp[8] = {215.0f, 265.0f, 1.0f};
            kp[9] = {415.0f, 182.0f, 1.0f};
            kp[10] = {225.0f, 182.0f, 1.0f};
            break;
        case 1:
            kp[7] = {425.0f, 245.0f, 1.0f};
            kp[9] = {358.0f, 168.0f, 1.0f};
            kp[8] = {252.0f, 325.0f, 1.0f};
            kp[10] = {255.0f, 390.0f, 1.0f};
            break;
        case 2:
            kp[7] = {445.0f, 248.0f, 1.0f};
            kp[9] = {515.0f, 240.0f, 1.0f};
            kp[8] = {245.0f, 305.0f, 1.0f};
            kp[10] = {295.0f, 290.0f, 1.0f};
            break;
        case 3:
            kp[7] = {425.0f, 265.0f, 1.0f};
            kp[9] = {445.0f, 220.0f, 1.0f};
            kp[8] = {330.0f, 275.0f, 1.0f};
            kp[10] = {385.0f, 235.0f, 1.0f};
            break;
        case 4:
            kp[7] = {385.0f, 310.0f, 1.0f};
            kp[8] = {255.0f, 310.0f, 1.0f};
            kp[9] = {285.0f, 272.0f, 1.0f};
            kp[10] = {355.0f, 272.0f, 1.0f};
            break;
        case 5:
            kp[7] = {430.0f, 190.0f, 1.0f};
            kp[8] = {210.0f, 190.0f, 1.0f};
            kp[9] = {352.0f, 145.0f, 1.0f};
            kp[10] = {288.0f, 145.0f, 1.0f};
            break;
        case 6:
            kp[7] = {445.0f, 250.0f, 1.0f};
            kp[8] = {195.0f, 250.0f, 1.0f};
            kp[9] = {525.0f, 250.0f, 1.0f};
            kp[10] = {115.0f, 250.0f, 1.0f};
            break;
        default:
            kp[7] = {382.0f, 325.0f, 1.0f};
            kp[8] = {258.0f, 325.0f, 1.0f};
            kp[9] = {385.0f, 395.0f, 1.0f};
            kp[10] = {255.0f, 395.0f, 1.0f};
            break;
    }

    if (mirror) {
        for (auto& pt : kp) {
            pt.x = 640.0f - pt.x;
        }
        constexpr std::array<std::pair<std::size_t, std::size_t>, 8> kSwapPairs{{
            {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}, {13, 14}, {15, 16}
        }};
        for (const auto& [a, b] : kSwapPairs) {
            std::swap(kp[a], kp[b]);
        }
    }

    return kp;
}

} // namespace meme
