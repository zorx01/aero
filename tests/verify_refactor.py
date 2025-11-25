
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Mock pyrealsense2 BEFORE importing aero
sys.modules["pyrealsense2"] = MagicMock()
import pyrealsense2 as rs

# Mock rerun
sys.modules["rerun"] = MagicMock()
sys.modules["rerun.blueprint"] = MagicMock()


# Setup RS mocks
mock_pipeline = MagicMock()
mock_config = MagicMock()
rs.pipeline.return_value = mock_pipeline
rs.config.return_value = mock_config

# Mock frames for initialization
mock_frames = MagicMock()
mock_pipeline.wait_for_frames.return_value = mock_frames
mock_extrinsics = MagicMock()
mock_extrinsics.rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1]
mock_extrinsics.translation = [0, 0, 0]
mock_frames.__getitem__.return_value.profile.get_extrinsics_to.return_value = mock_extrinsics
mock_frames.__getitem__.return_value.profile.as_video_stream_profile.return_value.intrinsics = MagicMock(
    fx=1.0, fy=1.0, ppx=320, ppy=240, width=640, height=480
)

# Mock aero.core.bindings
# We need to mock the classes that AeroSystem uses
mock_bindings = MagicMock()
sys.modules["aero.core.bindings"] = mock_bindings
sys.modules["core.bindings"] = mock_bindings # For flat structure
sys.modules["core"] = MagicMock()
sys.modules["core"].bindings = mock_bindings

# Mock specific classes in bindings
mock_bindings.Tracker = MagicMock() # Still needed if bindings has Tracker, but we use the real one from core
mock_bindings.Rig = MagicMock()
mock_bindings.Camera = MagicMock()
mock_bindings.ImuCalibration = MagicMock()
mock_bindings.Distortion = MagicMock()
mock_bindings.Pose = MagicMock()
mock_bindings.Odometry = MagicMock() # Mock Odometry class
mock_bindings.Odometry.Config = MagicMock() # Mock Config
mock_bindings.Odometry.OdometryMode.Inertial = 1
mock_bindings.Odometry.MulticameraMode.Precision = 0
mock_bindings.Odometry.RGBDSettings = MagicMock()

# Configure mocks for threads
# IMU data (bytes for np.frombuffer)
mock_frames.__getitem__.return_value.get_data.return_value = b'\x00' * 12 # 3 floats * 4 bytes

# Tracker.track return value (The real Tracker calls Odometry.track)
mock_pose_estimate = MagicMock()
mock_pose_estimate.world_from_rig.pose = MagicMock()
# The real Tracker.track returns (pose_estimate, slam_pose)
# But Odometry.track returns just pose_estimate
mock_bindings.Odometry.return_value.track.return_value = mock_pose_estimate
mock_bindings.Odometry.return_value.get_primary_cameras.return_value = [0]
mock_bindings.Odometry.return_value.get_state.return_value = MagicMock()


# Now import AeroSystem
try:
    from aero.interface import AeroSystem
    import aero.core
    import aero.interface
    
    # Inject attributes
    aero.interface.vslam.ImuMeasurement = mock_bindings.ImuMeasurement
    aero.interface.vslam.Pose = mock_bindings.Pose
    aero.interface.vslam.Rig = mock_bindings.Rig
    aero.interface.vslam.Tracker = aero.core.tracker.Tracker

except ImportError:
    # Flat structure support
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from interface import AeroSystem
    import core
    import interface
    
    # Inject attributes
    interface.vslam.ImuMeasurement = mock_bindings.ImuMeasurement
    interface.vslam.Pose = mock_bindings.Pose
    interface.vslam.Rig = mock_bindings.Rig
    interface.vslam.Tracker = core.tracker.Tracker


class TestAeroSystem(unittest.TestCase):
    def test_initialization(self):
        print("Testing AeroSystem initialization...")
        system = AeroSystem()
        
        # Run initialize
        system.initialize()
        
        # Verify tracker was created
        self.assertIsNotNone(system.tracker)
        self.assertIsNotNone(system.rig)
        
        # Verify RS pipeline was used to get params
        mock_pipeline.start.assert_called()
        mock_pipeline.stop.assert_called()
        
        print("AeroSystem initialized successfully with mocks.")

    def test_start_stop(self):
        print("Testing AeroSystem start/stop...")
        system = AeroSystem(config={"enable_visualization": False})
        
        # We need to initialize first
        system.initialize()
        
        # Start
        system.start()
        self.assertIsNotNone(system.ir_pipe)
        self.assertIsNotNone(system.motion_pipe)
        self.assertTrue(system.imu_thread_obj.is_alive())
        
        # Stop
        system.stop()
        self.assertFalse(system.imu_thread_obj.is_alive())
        print("AeroSystem start/stop cycle successful.")

if __name__ == "__main__":
    unittest.main()
