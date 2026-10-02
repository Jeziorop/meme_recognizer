# Real-Time Meme Gesture Recognizer

A real-time desktop application written in **C++20** that recognizes iconic meme gestures from a webcam feed using a GPU-accelerated PyTorch model exported to ONNX / native CUDA binary format. The application displays the live camera stream with upper-body skeleton tracking alongside the corresponding matched meme image and probability activations in a lightweight **Dear ImGui + OpenGL** GUI.

---

## Recognized Meme Gestures

1. **Absolute Cinema** (`absolute_cinema.png` — Martin Scorsese: both hands raised open at head height)
2. **Thinking Monkey** (`thinking_monkey.png` — one hand resting on chin, other arm relaxed)
3. **Elon Salute** (`elon.png` — Elon Musk: one arm stretched straight up and forward)
4. **Kamehameha** (`kamehameha.png` — Goku: both arms stretched forward in front next to each other)
5. **Trade Offer** (`trade.png` — hands pressed together in a praying / steeple pose)
6. **Gagri Gagri** (`gagri_gagri.png` — scratching top of head with both hands)
7. **SpongeBob Biceps** (`spongebob.png` — arm extended and bent 90° upward flexing bicep)
8. **T-Pose Dominance** (`t_pose.png` — both arms stretched straight horizontally at shoulder level)
9. **Chill Guy** (`chill_guy.png` — relaxed idle posture with arms resting down)

---

## Architecture & Technology Stack

- **ML & Data Pipeline (Python 3.11+)**:
  - Flexible dependency management: works with [`uv`](https://github.com/astral-sh/uv) or standard Python `venv` + `pip` without code changes.
  - `PyTorch` with CUDA acceleration.
  - `YOLOv8n-pose` (Ultralytics) for real-time upper-body keypoint detection.
  - 54-D pose feature extractor (joint unit vectors, cosine/sine angles, torso-normalized coordinates, and Euclidean distances).
  - Standalone webcam recorder with auto camera detection, target ghost skeleton guides, target meme thumbnails, 3-second countdowns, and on-the-fly model retraining.
- **Desktop Application (C++20)**:
  - **OpenCV 4**: Video capture (V4L2 device discovery and auto-exposure), frame preprocessing, and ONNX pose estimation.
  - **CUDA Runtime / Custom Kernel**: Single-block cooperative inference kernel for millisecond neural classification (with CPU/OpenCL fallback).
  - **Dear ImGui & OpenGL (GLX)**: Portable, lightweight, hardware-accelerated dark theme GUI featuring crisp anti-aliased TrueType typography, dynamic DPI/window rescaling, live webcam tracking, and probability meters without heavyweight framework overhead.
  - **Build System**: CMake 3.25+, Ninja, `g++` / `clang++`, and `vcpkg` package manager.

---

## Prerequisites & System Requirements

### 1. Operating System & Hardware
- **OS**: Linux (tested on Ubuntu 22.04 / 24.04 LTS).
- **Webcam**: Standard USB or integrated webcam (`/dev/video0`, `/dev/video2`, etc.).
- **GPU (Optional but recommended)**: Any NVIDIA CUDA-capable GPU with driver version $\ge$ 525 (automatic fallback to CPU if no CUDA GPU is detected).

### 2. System Packages & Build Tools
Install compilers, build utilities, Autotools (needed by vcpkg ports), and development libraries:
```bash
sudo apt update
sudo apt install -y build-essential g++ cmake ninja-build git pkg-config \
    autoconf autoconf-archive automake libtool zip unzip tar curl \
    libx11-dev libgl1-mesa-dev libv4l-dev v4l-utils
```

### 3. CUDA Toolkit (For GPU Acceleration)
Check if the NVIDIA CUDA compiler is already installed:
```bash
nvcc --version
```
If not installed and you have an NVIDIA GPU:
```bash
sudo apt install -y nvidia-cuda-toolkit
```

### 4. Python Environment Setup

You can set up the Python ML pipeline using either **`uv`** (fastest) or standard **`venv` + `pip`** (no extra tools needed):

#### Option A: Using `uv` (Recommended)
Check if `uv` is installed, or install it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Create the virtual environment and install dependencies:
```bash
uv sync
```

#### Option B: Standard Python (`venv` + `pip` without `uv`)
No changes are required to run without `uv`. Simply create a standard virtual environment and install dependencies via `pip`:
```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install the package with CUDA-enabled PyTorch
pip install --upgrade pip
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
```

### 5. C++ Dependencies (`vcpkg`)
Check if `vcpkg` is already installed and `VCPKG_ROOT` is configured:
```bash
vcpkg version
echo "$VCPKG_ROOT"
```
If you do not already have `vcpkg`, clone and bootstrap it, then set `VCPKG_ROOT`:
```bash
git clone https://github.com/microsoft/vcpkg.git ~/.local/share/vcpkg
~/.local/share/vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT="$HOME/.local/share/vcpkg"
```
*(Tip: add `export VCPKG_ROOT="$HOME/.local/share/vcpkg"` to your `~/.bashrc` or `~/.zshrc` so it persists).*

---

## Building the C++ Application

Configure, build, and test using CMake presets:

```bash
# 1. Configure CMake with Ninja and vcpkg
cmake --preset default

# 2. Build the executable and test suite
cmake --build --preset default

# 3. Run automated regression tests (verifies all 9 canonical & mirrored poses pass on GPU)
./build/default/meme_recognizer_tests
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
| `--pose <0..8>` | Launch the app initialized with a specific synthetic pose preset (`0` = Absolute Cinema, `1` = Thinking Monkey, etc.). |

### Interactive GUI Controls
- **Input Source Dropdown**: Switch between live webcams, auto-cycle demo, or fixed gesture presets.
- **Mirror Camera Checkbox**: Toggle selfie mirror mode.
- **Show Skeleton HUD Checkbox**: Toggle upper-body joint tracking and skeleton overlay.
- **Keyboard Shortcuts**:
  - `0`: Switch to live webcam.
  - `1` – `9`: Instantly switch to simulated poses 1 through 9.
  - `Q` or `Esc`: Quit application.

---

## Training & Dataset Collection (Python)

Data collection and model training are completely decoupled from the C++ presentation app.

### 1. Interactive Webcam Data Collection
To record customized gestures from your webcam with visual pose guides, countdowns, and on-the-fly GPU training:

**With `uv`:**
```bash
uv run record-dataset
```

**Without `uv` (Standard Python):**
```bash
python -m ml.record_dataset
# or directly with active .venv:
.venv/bin/python ml/record_dataset.py
```

#### Camera Selection:
The script automatically probes and selects active camera devices (such as `/dev/video2`). If you have multiple devices or want to choose manually:
```bash
# List all detected capture devices
python ml/record_dataset.py --list-cameras

# Specify a camera index or path
python ml/record_dataset.py --camera 2
python ml/record_dataset.py --camera /dev/video2
```

#### In-App Controls:
- **`1` – `9`**: Select the target meme gesture to record.
- **`H`**: Toggle the ghost target pose skeleton guide and meme thumbnail preview.
- **`C`**: Cycle to the next detected camera device on the fly.
- **`SPACE`**: Start a 3-second countdown and record a burst of pose frames.
- **`T`**: Train the PyTorch classifier on GPU and re-export ONNX & CUDA model binaries.
- **`Q` / `Esc`**: Quit.

### 2. Manual Pipeline Execution
To synchronize the meme manifest, build the pose backbone, and retrain the classifier:

**With `uv`:**
```bash
# Run the complete end-to-end pipeline
uv run python -m ml.pipeline

# Or run training directly
uv run python ml/train.py
```

**Without `uv` (Standard Python):**
```bash
# 1. Synchronize manifest.json with real meme images in assets/memes/
python -m ml.collect_memes

# 2. Run the end-to-end pipeline (manifest sync + backbone export + training)
python -m ml.pipeline

# Or run classifier training directly
python ml/train.py
```

This updates:
- `assets/memes/manifest.json` (class metadata, titles, hints, accent colors)
- `models/meme_gesture_classifier.onnx` (OpenCV DNN compatible model)
- `models/meme_gesture_classifier.bin` (Direct binary weights for custom CUDA inference)
- `models/training_report.json` (Validation accuracy and numerical equivalence metrics)
