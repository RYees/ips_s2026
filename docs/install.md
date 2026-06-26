# Installation

## 1. Python Environment

Create or activate the Python environment used for the project, then install the package requirements:

```bash
cd rgbd
pip install -r requirements.txt
```

The requirements file includes the Python packages used by the live demo, data collection UI, and masking helpers.

## 2. Orbbec SDK V2 Python Wrapper

The live demo and data-collection UI require the Orbbec SDK V2 Python Wrapper. The `rgbd/Makefile` assumes the SDK repository is cloned in the same parent directory as `ips_s2026`, so the layout looks like this:

```text
<workspace>/
  ips_s2026/
  pyorbbecsdk/
```

Follow these steps:

1. Clone the repository to get the latest version:

   ```bash
   git clone https://github.com/orbbec/pyorbbecsdk.git
   cd pyorbbecsdk
   git checkout v2-main
   ```

2. Install the necessary Python development packages:

   ```bash
   sudo apt-get install python3-dev python3-venv python3-pip python3-opencv
   ```

3. Create a virtual environment and build the project:

   ```bash
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

## 3. Camera Access

Make sure the camera is not open in another application before starting the demo or data collection UI. If the device is already in use, the camera open step will fail.

## 4. Recommended Check

Run the camera preview first if you want to confirm that the SDK and camera connection are healthy:

```bash
cd rgbd
make run-preview
```

## 5. Notes

- The `rgbd/Makefile` assumes `pyorbbecsdk` exists at `../../pyorbbecsdk` relative to the `rgbd/` folder.
- If your SDK lives somewhere else, update `SDK_ROOT` in `rgbd/Makefile` before running the app.
