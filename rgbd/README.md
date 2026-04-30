## RGB-D Data Collector with Orbbec Femto Bolt Camera

### About

This GUI-based tool allows you to capture synchronized RGB, depth, and segmentation mask images from an Orbbec Femto Bolt camera. It also automatically generates annotations in YOLO format, stores point clouds, and logs camera intrinsics and scene metadata.

### Camera Specifications 
Orbbec Femto Bolt

### Folder structure

```
├── main.py                  # Entry point: GUI app for capturing & labeling
├── camera_interface.py      # RealSense camera setup and frame retrieval
├── segmentation_helper.py   # Depth segmentation + plane removal
├── annotation_writer.py     # YOLO-style annotation writer
├── ply_viewer.py            # Script to view .ply images
├── requirements.txt         # Python dependencies
├── depth_info.py            # Script to convert .npy to depth image
├── utils.py                 # Frame to BGR conversion
├── view_numpy.py            # Script to view .npy file as an image
├── README.md                # This file
```

### Setup procedure

1. User guide to install Orbbec SDK V2 Python Wrapper can also be found at [Orbbec SDK V2 Python Wrapper](https://orbbec.github.io/pyorbbecsdk/index.html)
    1. Clone the repository to get the latest version
    ```
    git clone   https://github.com/orbbec/pyorbbecsdk.git
    git checkout v2-main
    ```

    2. Install the necessary python development packages
    ```
    sudo apt-get install python3-dev python3-venv python3-pip python3-opencv
    ```
    3. Create a virtual environment and build the project
    ```
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
    4. Set up the environment in pyorbbecsdk
    ```
    cd ..
    export PYTHONPATH=$PYTHONPATH:$(pwd)/install/lib/
    sudo bash ./scripts/install_udev_rules.sh
    sudo udevadm control --reload-rules && sudo udevadm trigger
    ```
    5. Generate stubs for better IntelliSense support in your IDE
    ```
    source env.sh
    pip3 install pybind11-stubgen
    pybind11-stubgen pyorbbecsdk
    ```

2. Add the files of this git repository to examples/rgbd

3. Install the requirements in examples/rgbd
    ```
    pip install -r requirements.txt
    ```

4. Run the main.py file
    ```
    cd examples/rgbd
    python3 main.py
    ```

5. Controls (using kevboard or gui)
    1. Enter - Capture current RGB-D frame
    2. S - Save captured frame and label
    3. R - Retake / discard frame
    4. P - Preview 3D point cloud
    5. Q - Quit the application
    6. Dropdown to select the class (0 - Copper, 1 - Steel)

6. Captured data is stored in a dataset/ folder
    ```
    dataset/
    ├── images/         # RGB images (img0000.png, ...)
    ├── depth/          # Normalized color depth images (.png)
    ├── labels/         # YOLO-format annotations (.txt) <class_label> x1 y1 x2 y2 x3 y3 ... xn yn
    ├── pointcloud/     # 3D point clouds (.ply)
    ├── info/           # Per-frame metadata logs like intrinsics and depth info (.txt)
    ```
### Features

- Synchronized RGB + Depth capture
- Depth-based segmentation with plane removal using RANSAC
- YOLO-style annotation writer
- Tkinter GUI with interactive controls
- Live preview and contour overlay
- Saves metadata, point cloud, and intrinsics per frame