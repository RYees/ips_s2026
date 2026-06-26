## RGB-D Data Collector with Orbbec Femto Bolt Camera

### About

This project captures synchronized RGB and depth frames from an Orbbec Femto Bolt camera, generates segmentation masks, writes YOLO-style annotations, saves point clouds, and logs camera intrinsics and capture metadata.

### Important requirements

- The camera code is designed around the Orbbec Femto Bolt RGB-D camera.
- `pyorbbecsdk` must be installed and configured before running this code.
- A connected Orbbec Femto Bolt camera is required to capture live frames.
- This repository is best supported on Linux; the current build and camera setup are Linux-focused.

### Folder structure

```
├── main.py                      # Main GUI capture app for dataset collection
├── camera/                      # Camera backends and example capture drivers
│   ├── camera_interface.py      # Primary Orbbec capture backend used by main.py
│   ├── camera_interface_fixed.py# Refined camera backend with more robust color-format handling
│   ├── camera_test.py           # Experimental capture backend for preview/testing
│   └── camera_test2.py          # Second experimental backend for preview/testing
├── processing/                  # Data processing helpers
│   ├── annotation_writer.py     # Writes YOLO-style annotations from masks
│   ├── segmentation_helper.py   # Depth segmentation and plane removal logic
│   └── utils.py                 # Frame conversion utilities
├── tools/                       # Utility scripts and experimental tools
│   ├── camera_preview.py        # Run different experimental camera backends
│   ├── ply_viewer.py            # View saved .ply point clouds
│   ├── view_numpy.py            # View saved NumPy depth or image arrays
│   ├── depth.info.py            # Convert depth `.npy` arrays into images
│   └── cam_data.py              # Camera parameter inspection utility
├── requirements.txt             # Python dependencies for this folder
├── README.md                    # This file
```

### Setup procedure

User guide to install Orbbec SDK V2 Python Wrapper can also be found at Orbbec SDK V2 Python Wrapper.

1. Clone the repository to get the latest version:

    ```bash
    git clone https://github.com/orbbec/pyorbbecsdk.git
    git checkout v2-main
    ```

2. Install the necessary python development packages:

    ```bash
    sudo apt-get install python3-dev python3-venv python3-pip python3-opencv
    ```

3. Create a virtual environment and build the project:

    ```bash
    cd pyorbbecsdk
    python3 -m venv ./venv
    source venv/bin/activate
    pip3 install -r requirements.txt
    mkdir build
    cd build
    cmake -Dpybind11_DIR=`pybind11-config --cmakedir` ..
    make -j4
    make install
    ```

4. Set up the environment in `pyorbbecsdk`:

    ```bash
    cd ..
    export PYTHONPATH=$PYTHONPATH:$(pwd)/install/lib/
    sudo bash ./scripts/install_udev_rules.sh
    sudo udevadm control --reload-rules && sudo udevadm trigger
    ```

5. Generate stubs for better IntelliSense support in your IDE:

    ```bash
    source env.sh
    pip3 install pybind11-stubgen
    pybind11-stubgen pyorbbecsdk
    ```

6. Clone this repository in the same parent folder as `pyorbbecsdk`, then install this project's Python dependencies:

    ```bash
    cd /Users/rzapp/Documents/A{sp}A/ips_s2026/rgbd
    python3 -m venv ./venv
    source ./venv/bin/activate
    pip install -r requirements.txt
    ```

7. Connect the Orbbec Femto Bolt camera to the Linux machine before running the app. The `rgbd/Makefile` assumes `pyorbbecsdk` is available at `../../pyorbbecsdk` relative to the `rgbd/` folder.

### Running the main capture app

From the `rgbd` folder:

```bash
cd /Users/rzapp/Documents/A{sp}A/ips_s2026/rgbd
source ./venv/bin/activate
python3 main.py
```

This launches the main GUI for dataset capture.

### Running the preview/example backends

The preview tool runs the experimental camera backends without using `main.py`:

```bash
cd /Users/rzapp/Documents/A{sp}A/ips_s2026/rgbd
source ./venv/bin/activate
python3 tools/camera_preview.py --backend test2
```

Available backends:
- `test` — uses `camera/camera_test.py`
- `test2` — uses `camera/camera_test2.py`
- `fixed` — uses `camera/camera_interface_fixed.py`

Or use the Makefile target:

```bash
make run-preview
```

### Controls in the main app

- `Enter` — capture current RGB-D frame
- `S` — save captured frame, mask, point cloud, and annotations
- `R` — retake / discard current capture
- `P` — preview the current point cloud
- `Q` — quit the application
- Class dropdown — select label class for annotation

### Output dataset structure

Captured data is saved under `dataset/`:

```
dataset/
├── images/         # RGB images
├── depth/          # Depth visualization images
├── labels/         # YOLO annotation text files
├── pointcloud/     # Saved point clouds in .ply format
├── info/           # Metadata logs and intrinsics info
```

### Live demo model files

The live inference demo loads `.pt` files from `rgdb/live/models/` and shows the selected file name in the navbar.

Current weights:

- `best_multiple_26.pt`
- `best_single_26.pt`
- `m11best.pt`
- `m8best.pt`
- `s11best.pt`
- `s8best.pt`

See `rgdb/docs/models.md` for the current model list.

### Notes

- The code is currently more suitable for Linux and the Orbbec Femto Bolt camera.
- If the camera is not connected or `pyorbbecsdk` is not installed, the app will not run.
- Keep `main.py` for production data capture, and use `tools/camera_preview.py` for experimental camera backends.
