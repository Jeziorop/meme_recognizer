# Real-Time Meme Gesture Recognizer (PyTorch CUDA $\rightarrow$ ONNX $\rightarrow$ C++20 / OpenCV 4 / Qt 6)

A real-time webcam application that recognizes iconic meme gestures (such as Martin Scorsese's **"Absolute Cinema"**, **"Thinking / Roll Safe"**, **"Leo Pointing"**, **"Drake Reject"**, **"Arms Crossed"**, **"Shocked / Hands on Head"**, **"T-Pose"**, and **"Neutral / Chill"**) using a GPU-accelerated PyTorch model exported to ONNX and executed inside a C++20 desktop application built with OpenCV 4, CUDA, and Qt 6.

## Quick Start

### 1. Python ML Pipeline (`uv`)
```bash
uv sync
uv run python -m ml.pipeline
```

### 2. C++20 Application Build (`vcpkg` + `CMake` + `Ninja` + `g++`)
```bash
cmake --preset default
cmake --build --preset default
ctest --preset default --output-on-failure
./build/default/meme_recognizer
```
