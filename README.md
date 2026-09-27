# Real-Time Meme Gesture Recognizer

A real-time desktop application written in **C++20** that recognizes iconic meme gestures from a webcam feed using a GPU-accelerated PyTorch model exported to ONNX / native CUDA binary format. The application displays the live camera stream with upper-body skeleton tracking alongside the corresponding matched meme image and probability activations in a **Qt 6** GUI.

---

## Recognized Meme Gestures

1. **Absolute Cinema** (Martin Scorsese — hands open at head height)
2. **Thinking / Roll Safe** (Kayode Ewumi — finger tapping temple)
3. **Leo Pointing** (Rick Dalton / Leonardo DiCaprio — arm pointing forward)
4. **Drake Reject** (Drake hotline bling — palm facing camera turned away)
5. **Arms Crossed** (Arms folded across chest in an "X" pose)
6. **Shocked / Hands on Head** (Both hands placed on top of head)
7. **T-Pose Dominance** (Arms straight horizontally forming a "T")
8. **Neutral / Chill** (Resting upper body pose)

---

## Architecture & Technology Stack

- **ML & Data Pipeline (Python 3.11+)**:
  - `uv` for fast, reproducible dependency management and virtual environments.
  - `PyTorch 2.4.1` with CUDA acceleration.
  - `YOLOv8n-pose` (Ultralytics) for upper-body keypoint detection.
  - Pose feature extractor (joint angles, torso-normalized coordinates, and Euclidean distances).
  - Hands-free dataset recorder with automatic 3-second countdown and on-the-fly model retraining.
- **Desktop Application (C++20)**:
  - **OpenCV 4**: Video capture (V4L2), frame preprocessing, and ONNX pose estimation.
  - **CUDA Runtime / Custom Kernel**: Single-block cooperative inference kernel for millisecond neural classification.
  - **Qt 6**: Hardware-accelerated dark theme GUI showing live webcam tracking, confidence meters, and meme cards.
  - **Build System**: CMake 3.25+, Ninja, `g++-12` / `g++`, and `vcpkg` package manager.

---

## Prerequisites & System Requirements

### 1. Operating System & Hardware
- **OS**: Linux (tested on Ubuntu 22.04 / 24.04 LTS).
- **Webcam**: Standard USB or integrated webcam (`/dev/video0` or `/dev/video2`).
- **GPU (Optional but recommended)**: Any NVIDIA CUDA-capable GPU with driver version $\ge$ 525 (automatic fallback to CPU if no CUDA GPU is detected).

### 2. System Packages & Build Tools
Install the required system compilers and build utilities:
```bash
sudo apt update
sudo apt install -y build-essential g++ cmake ninja-build git pkg-config \
    libx11-dev libxext-dev libxrender-dev libgl1-mesa-dev libxkbcommon-dev
```

### 3. CUDA Toolkit (For GPU Acceleration)
```bash
# If using NVIDIA GPU
sudo apt install -y nvidia-cuda-toolkit
nvcc --version
```

### 4. Python Package Manager (`uv`)
Install [`uv`](https://github.com/astral-sh/uv) (fast Python package installer):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 5. C++ Dependencies (`vcpkg`)
Ensure `vcpkg` is installed and set in your environment:
```bash
git clone https://github.com/microsoft/vcpkg.git ~/.local/share/vcpkg
~/.local/share/vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT="$HOME/.local/share/vcpkg"
```

---

## Building the C++ Application

Configure and build the project using CMake presets:

```bash
# 1. Configure CMake with Ninja and vcpkg
cmake --preset default

# 2. Build the executable and test suite
cmake --build --preset default

# 3. Run automated regression tests
ctest --preset default --output-on-failure
```

---

## Running the Application

### Launch the GUI
```bash
./build/default/meme_recognizer
```

### Command-Line Options
| Option | Description |
| :--- | :--- |
| `-h, --help` | Display command-line options. |
| `--self-test` | Run headless GUI & inference self-test, verify model execution, and exit. |
| `--capture <file.png>` | Capture and save a snapshot of the application window to disk. |
| `--pose <0..7>` | Launch the app initialized with a specific synthetic pose preset (0 = Absolute Cinema, 1 = Roll Safe, etc.). |

### Interactive GUI Controls
- **Input Source Dropdown**: Switch between live webcams (`/dev/video0`, `/dev/video2`), auto-cycle demo, or fixed gesture presets.
- **Mirror Camera Checkbox**: Toggle selfie mirror mode.
- **Show Skeleton HUD Checkbox**: Toggle upper-body joint tracking and skeleton overlay.
- **Keyboard Shortcuts**:
  - `0`: Switch to live webcam.
  - `1` – `8`: Instantly switch to simulated poses 1 through 8.
  - `Q` or `Esc`: Quit application.

---

## Training & Dataset Collection (Python)

Data collection and model training are completely decoupled from the C++ presentation app.

### 1. Hands-Free Webcam Data Collection
To record customized gestures from your own webcam with audio countdowns and 3-second pose capture:
```bash
uv run record-dataset
```
Follow the interactive prompts:
- Select which gesture to record (or record all 8 in sequence).
- A 3-second countdown will prepare you before recording starts.
- After recording, the script will automatically retrain and re-export the model weights.

### 2. Manual Pipeline Execution
To execute the complete data generation, model training, and ONNX export pipeline:
```bash
# Fetch YOLOv8 pose weights and generate synthetic & recorded training data
uv run python -m ml.pipeline

# Or run training directly
uv run python ml/train.py
```
This updates:
- `models/meme_gesture_classifier.onnx` (OpenCV DNN compatible model)
- `models/meme_gesture_classifier.bin` (Direct binary weights for custom CUDA inference)
