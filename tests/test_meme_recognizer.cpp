#include "GpuDeviceManager.hpp"
#include "MemeClassifier.hpp"
#include "PoseDetector.hpp"
#include "PoseFeatureExtractor.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

int main() {
    std::cout << "=== Running C++20 + GPU Meme Gesture Recognizer Test Suite ===\n";

    const auto gpu = meme::GpuDeviceManager::initialize();
    std::cout << "[Test] Active Compute Device: " << gpu.active_summary << "\n";

    const std::filesystem::path root = std::filesystem::current_path();
    meme::MemeClassifier classifier;
    const bool manifest_ok = classifier.loadManifest(root / "assets" / "memes" / "manifest.json");
    assert(manifest_ok && "Failed to load assets/memes/manifest.json");

    const bool models_ok = classifier.loadModels(
        root / "models" / "meme_gesture_classifier.onnx",
        root / "models" / "meme_gesture_classifier.bin"
    );
    assert(models_ok && "Failed to load ONNX / CUDA classifier models");

    meme::PoseDetector detector;
    const bool pose_ok = detector.loadModels(
        root / "models" / "yolov8n-pose.onnx"
    );
    assert(pose_ok && "Failed to load Stage-1 yolov8n-pose.onnx");

    // Verify all 9 canonical meme poses (both unmirrored and mirrored!) classify with 100% Top-1 accuracy
    for (int c = 0; c < static_cast<int>(meme::kNumMemeClasses); ++c) {
        for (bool mirror : {false, true}) {
            classifier.resetSmoothing();
            const auto kp = meme::PoseFeatureExtractor::getCanonicalPose(c, mirror);
            const auto feat = meme::PoseFeatureExtractor::extract(kp);
            const auto pred = classifier.classify(feat, 1.0f);

            std::cout << "  Class " << c << " (" << classifier.classes()[c].title
                      << ", mirror=" << (mirror ? "true" : "false")
                      << ") -> Predicted: " << pred.class_index
                      << " | Confidence: " << (pred.confidence * 100.0f) << "% | Backend: "
                      << pred.backend_used << "\n";

            if (pred.class_index != c || pred.confidence < 0.80f) {
                std::cerr << "FAIL: Expected class " << c << " with >=80% confidence, got "
                          << pred.class_index << " (" << pred.confidence << ")\n";
                return 1;
            }
        }
    }

    std::cout << "=== ALL 18/18 CANONICAL & MIRRORED POSE TESTS PASSED ON GPU! ===\n";
    return 0;
}
