# Installation

## 1. Python Environment

Create or activate the Python environment used for the project, then install the package requirements:

```bash
cd rgbd
pip install -r requirements.txt
```

The requirements file includes the Python packages used by the live demo, data collection UI, and masking helpers.

The data-collection UI expects the shared `offline_case/` masking utilities to be available in the full CPS workspace checkout.

## 2. Orbbec SDK

This project also needs the Orbbec Python bindings:

- `pyorbbecsdk`
- the matching Orbbec native libraries

The repo assumes the SDK is available in the lab environment and that the paths are set correctly in `rgbd/Makefile`.

If your SDK is installed somewhere else, update `SDK_ROOT` in `rgbd/Makefile` before running the app.

## 3. Camera Access

Make sure the camera is not open in another application before starting the demo or data collection UI. If the device is already in use, the camera open step will fail.

## 4. Recommended Check

Run the camera preview first if you want to confirm that the SDK and camera connection are healthy:

```bash
cd rgbd
make run-preview
```
