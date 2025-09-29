# Lightweight-Webcam-Eye-Tracking
A **webcam-only** eye-tracking system that works on **any screen size**.  
Edit this line in `main.py` to set your exact window size/flags:
```python
screen = pygame.display.set_mode((1920, 1080), pygame.NOFRAME)
```
It has been tested on large displays where traditional IR eye trackers often struggle or are impractical.
The system extracts 478×3 MediaPipe FaceMesh landmarks and learns lightweight ML regressors to predict on-screen gaze coordinates in real time. Includes fast calibration, testing UI, and tracking.

---

## TL;DR
- **Task**: Map facial mesh features → 2D screen coordinates (gaze point) on any display
- **Method:** Webcam + MediaPipe FaceMesh → SGD / Ridge / MLP / SVR / XGBoost regressors for (x, y)
- **Motivation:** Avoid specialized hardware; robust at larger viewing distances where classic IR trackers may not function well
- **UI:** Calibration, Test (with success tally), Track

## 📌 Overview
This project uses dense face/iris landmarks (MediaPipe FaceMesh) and direct regression to screen coordinates, paired with smooth-moving calibration that’s quick and tolerant to head motion. It supports any screen—just set the resolution you want with `pygame.display.set_mode(...)`. The approach is especially practical for large displays and kiosk-like setups.

**Key Features**
- Webcam-only pipeline with MediaPipe FaceMesh (refined landmarks)
- Fast calibration modes (smooth path, edges, random)
- Lightweight ML regressors (SGD, Ridge, MLP, SVR, XGB) selectable at runtime
- Guard-box: auto-pauses when you leave the allowed head region; resumes on return
- Optional region map heatmap of collected samples

## Method
- **Landmarks & Features:** Camera frames → MediaPipe FaceMesh → 478 × (x,y,z) flattened vector
- **Screen Prediction:** Two regressors: fₓ(features) → x and fᵧ(features) → y
- **Calibration:** Moving target / edge / random sequences; each frame logs `[features..., target_x, target_y]` to CSV
- **Testing:** Randomly placed green rectangle; move the dot inside and hit E to record success
- **Tracking:** Continuous prediction with temporal averaging for smoother motion

## 📦 Project Structure
```graphql
eye-tracking/
├─ main.py                 # Calibrate / Test / Track UI (pygame)
├─ Gaze.py                 # MediaPipe FaceMesh capture + guard-box (OpenCV)
├─ Target.py               # Target (dot) + Test rectangle rendering (pygame)
├─ utils.py                # Config, region map, helpers
├─ create_models.py        # (Optional) offline training example
├─ Eye_tracking_system.pdf # Method & results paper
├─ config.ini              # Your settings (see below)
├─ data/                   # region_map.npy (auto)
├─ data_csvs/              # calibration CSVs (auto)
└─ test_results/           # per-model test logs (auto)
```

## ⚙️ Setup
### 1. Clone the repo
```bash
git clone https://github.com/Anomaly33/Lightweight-Webcam-Eye-Tracking.git
cd Lightweight-Webcam-Eye-Tracking
```
### 2. Create Environment
```bash
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# On Windows: .venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Config(`config.ini`)
Controls the calibration setup. Increasing `target_speed` makes the calibration faster but lesser data points collected. `record_frame_rate` specifes the number of frames of data colleced per second. Adjust it according to individual webcam specs.
```ini
[DEFAULT]
image_size = 64
target_speed = 600
target_radius = 20
map_scale = 10
avg_window_length = 8
record_frame_rate = 30
number_of_test_points = 10
points_to_collect = 250

[COLOURS]
white = (255,255,255,255)
black = (0,0,0,255)
gray = (120,120,120,255)
red = (255,0,0,255)
green = (0,255,0,255)
blue = (0,128,255,255)
```
