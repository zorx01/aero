
# Aero Tracking System

![Demo](pycuvslam.gif)

A custom Python package for VIO (Visual Inertial Odometry) tracking, refactored from NVlabs/PyCuVSLAM.

## Structure

The project is organized as a self-contained package:

*   `core/`: Core bindings to the underlying C++ libraries (`pycuvslam`).
*   `utils/`: Utility modules for camera and visualization.
*   `interface.py`: Main `AeroSystem` class for high-level control.
*   `examples/`: Example scripts.
*   `tests/`: Verification scripts.

## Prerequisites

*   **Hardware**: Intel RealSense camera (e.g., D435i, D455) with IMU.
*   **OS**: Linux (tested on Ubuntu).
*   **Drivers**: Intel RealSense SDK 2.0 (`librealsense2`).
*   **Git LFS**: Required to fetch the binary shared objects (`.so` files).

## Setup

1.  **Clone the repository and fetch binaries:**
    ```bash
    git clone <repository-url>
    cd aero
    git lfs pull  # Important: Fetches the actual .so files
    ```

2.  **Install Python dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Example

An example script is provided to demonstrate how to initialize and run the `AeroSystem`.

```bash
python3 examples/run_system.py
```

This script will:
1.  Initialize the `AeroSystem`.
2.  Start the camera and IMU pipelines.
3.  Launch the Rerun visualizer (if enabled).
4.  Process frames and output pose data.

### Using the API

You can import `AeroSystem` in your own scripts. If your script is in the root directory:

```python
from interface import AeroSystem

# Initialize
system = AeroSystem(config={"enable_visualization": True})
system.initialize()

# Start tracking
system.start()

try:
    while True:
        # Get the latest pose
        pose = system.process_next_frame()
        if pose:
            print(f"Position: {pose.translation}")
finally:
    system.stop()
```

## Troubleshooting

*   **Missing Module `cuvslam`**: Ensure `git lfs pull` was run successfully. The `pycuvslam.so` file in `core/` must be a binary file, not a text pointer.
*   **RealSense Device Not Found**: Check your USB connection and ensure `realsense-viewer` works.