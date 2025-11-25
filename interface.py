
import queue
import threading
import time
from copy import deepcopy
from typing import List, Optional, Tuple, Dict, Any
import logging

import numpy as np
import pyrealsense2 as rs

try:
    from aero import core as vslam
    from aero.utils import camera as camera_utils
    from aero.utils import visualization as viz_utils
except ImportError:
    import core as vslam
    from utils import camera as camera_utils
    from utils import visualization as viz_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_RESOLUTION = (640, 360)
DEFAULT_FPS = 30
DEFAULT_IMU_FREQ_ACCEL = 200
DEFAULT_IMU_FREQ_GYRO = 200
IMAGE_JITTER_THRESHOLD_MS = 35 * 1e6
IMU_JITTER_THRESHOLD_MS = 6 * 1e6


class ThreadWithTimestamp:
    """Helper class to manage timestamps between camera and IMU threads."""
    
    def __init__(
        self,
        low_rate_threshold_ns: int,
        high_rate_threshold_ns: int
    ) -> None:
        self.prev_low_rate_timestamp: Optional[int] = None
        self.prev_high_rate_timestamp: Optional[int] = None
        self.low_rate_threshold_ns = low_rate_threshold_ns
        self.high_rate_threshold_ns = high_rate_threshold_ns
        self.last_low_rate_timestamp: Optional[int] = None


class AeroSystem:
    """
    Main interface for the Aero tracking system.
    Encapsulates the VIO tracker, camera/IMU pipelines, and processing threads.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.tracker: Optional[vslam.Tracker] = None
        self.rig: Optional[vslam.Rig] = None
        
        # Pipelines
        self.ir_pipe: Optional[rs.pipeline] = None
        self.motion_pipe: Optional[rs.pipeline] = None
        
        # Threading
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.imu_thread_obj: Optional[threading.Thread] = None
        self.camera_thread_obj: Optional[threading.Thread] = None
        self.timestamp_helper = ThreadWithTimestamp(
            IMAGE_JITTER_THRESHOLD_MS, IMU_JITTER_THRESHOLD_MS
        )
        
        # State
        self.latest_pose: Optional[vslam.Pose] = None
        self.trajectory: List[np.ndarray] = []
        self.frame_id = 0
        
        # Visualization
        self.visualizer: Optional[viz_utils.RerunVisualizer] = None
        self.enable_viz = self.config.get("enable_visualization", True)

    def initialize(self) -> None:
        """Initialize the system, setup cameras, and create the tracker."""
        logger.info("Initializing AeroSystem...")
        
        # 1. Setup camera parameters (requires brief pipeline start)
        camera_params = self._setup_camera_parameters()
        
        # 2. Create Rig
        self.rig = camera_utils.get_rs_vio_rig(camera_params)
        
        # 3. Configure Tracker
        tracker_config = vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            debug_imu_mode=False,
            odometry_mode=vslam.Tracker.OdometryMode.Inertial,
            horizontal_stereo_camera=True
        )
        
        # 4. Initialize Tracker
        self.tracker = vslam.Tracker(self.rig, tracker_config)
        
        if self.enable_viz:
            self.visualizer = viz_utils.RerunVisualizer()
            
        logger.info("AeroSystem initialized successfully.")

    def start(self) -> None:
        """Start the camera/IMU pipelines and processing threads."""
        if not self.tracker:
            raise RuntimeError("System not initialized. Call initialize() first.")
            
        logger.info("Starting AeroSystem pipelines...")
        self.stop_event.clear()
        
        # Setup IR Pipeline
        self.ir_pipe = rs.pipeline()
        ir_config = rs.config()
        ir_config.enable_stream(
            rs.stream.infrared, 1, DEFAULT_RESOLUTION[0], DEFAULT_RESOLUTION[1], rs.format.y8, DEFAULT_FPS
        )
        ir_config.enable_stream(
            rs.stream.infrared, 2, DEFAULT_RESOLUTION[0], DEFAULT_RESOLUTION[1], rs.format.y8, DEFAULT_FPS
        )
        
        # Configure device (emitter off)
        self._configure_device(self.ir_pipe, ir_config)
        
        # Setup Motion Pipeline
        self.motion_pipe = rs.pipeline()
        motion_config = rs.config()
        motion_config.enable_stream(
            rs.stream.accel, rs.format.motion_xyz32f, DEFAULT_IMU_FREQ_ACCEL
        )
        motion_config.enable_stream(
            rs.stream.gyro, rs.format.motion_xyz32f, DEFAULT_IMU_FREQ_GYRO
        )
        
        # Start pipelines
        self.motion_pipe.start(motion_config)
        self.ir_pipe.start(ir_config)
        
        # Start threads
        self.imu_thread_obj = threading.Thread(
            target=self._imu_worker,
            daemon=True
        )
        self.camera_thread_obj = threading.Thread(
            target=self._camera_worker,
            daemon=True
        )
        
        self.imu_thread_obj.start()
        self.camera_thread_obj.start()
        logger.info("AeroSystem started.")

    def stop(self) -> None:
        """Stop the system and cleanup resources."""
        logger.info("Stopping AeroSystem...")
        self.stop_event.set()
        
        if self.imu_thread_obj:
            self.imu_thread_obj.join(timeout=2.0)
        if self.camera_thread_obj:
            self.camera_thread_obj.join(timeout=2.0)
            
        if self.motion_pipe:
            self.motion_pipe.stop()
        if self.ir_pipe:
            self.ir_pipe.stop()
            
        logger.info("AeroSystem stopped.")

    def process_next_frame(self, timeout: float = 1.0) -> Optional[vslam.Pose]:
        """
        Process the next available frame from the queue.
        Returns the latest pose if available, or None.
        """
        try:
            timestamp, odom_pose, images = self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

        if odom_pose is None:
            return None

        self.frame_id += 1
        self.latest_pose = odom_pose
        self.trajectory.append(odom_pose.translation)

        gravity = None
        # Assuming Inertial mode for now, as hardcoded in initialize
        gravity = self.tracker.get_last_gravity()

        if self.enable_viz and self.visualizer:
            self.visualizer.visualize_frame(
                frame_id=self.frame_id,
                images=[images[0]],
                pose=odom_pose,
                observations_main_cam=[self.tracker.get_last_observations(0)],
                trajectory=self.trajectory,
                timestamp=timestamp,
                gravity=gravity
            )
            
        return odom_pose

    def _setup_camera_parameters(self) -> dict:
        """Internal method to fetch camera parameters."""
        config = rs.config()
        pipeline = rs.pipeline()

        config.enable_stream(
            rs.stream.infrared, 1, DEFAULT_RESOLUTION[0], DEFAULT_RESOLUTION[1], rs.format.y8, DEFAULT_FPS
        )
        config.enable_stream(
            rs.stream.infrared, 2, DEFAULT_RESOLUTION[0], DEFAULT_RESOLUTION[1], rs.format.y8, DEFAULT_FPS
        )
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, DEFAULT_IMU_FREQ_ACCEL)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, DEFAULT_IMU_FREQ_GYRO)

        profile = pipeline.start(config)
        frames = pipeline.wait_for_frames()
        pipeline.stop()

        camera_params = {'left': {}, 'right': {}, 'imu': {}}
        camera_params['right']['extrinsics'] = frames[1].profile.get_extrinsics_to(
            frames[0].profile
        )
        camera_params['imu']['cam_from_imu'] = frames[2].profile.get_extrinsics_to(
            frames[0].profile
        )
        camera_params['left']['intrinsics'] = (
            frames[0].profile.as_video_stream_profile().intrinsics
        )
        camera_params['right']['intrinsics'] = (
            frames[1].profile.as_video_stream_profile().intrinsics
        )

        return camera_params

    def _configure_device(self, pipeline: rs.pipeline, config: rs.config) -> None:
        config_temp = rs.config()
        ir_wrapper = rs.pipeline_wrapper(pipeline)
        ir_profile = config_temp.resolve(ir_wrapper)
        device = ir_profile.get_device()
        depth_sensor = device.query_sensors()[0]
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)

    def _imu_worker(self) -> None:
        """Worker thread for IMU data."""
        try:
            while not self.stop_event.is_set():
                imu_measurement = vslam.ImuMeasurement()
                # Note: wait_for_frames might block, so we check stop_event after
                # Ideally we should use a timeout or poll
                try:
                    imu_frames = self.motion_pipe.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    continue
                    
                current_timestamp = int(imu_frames[0].timestamp * 1e6)

                # Timestamp checks (simplified from original)
                if (self.timestamp_helper.last_low_rate_timestamp is not None and
                        current_timestamp < self.timestamp_helper.last_low_rate_timestamp):
                    continue

                timestamp_diff = 0
                if self.timestamp_helper.prev_high_rate_timestamp is not None:
                    timestamp_diff = (
                        current_timestamp - self.timestamp_helper.prev_high_rate_timestamp
                    )
                    if timestamp_diff < 0:
                        continue

                self.timestamp_helper.prev_high_rate_timestamp = deepcopy(current_timestamp)
                
                imu_measurement.timestamp_ns = current_timestamp
                accel_data = np.frombuffer(imu_frames[0].get_data(), dtype=np.float32)
                gyro_data = np.frombuffer(imu_frames[1].get_data(), dtype=np.float32)
                imu_measurement.linear_accelerations = accel_data[:3]
                imu_measurement.angular_velocities = gyro_data[:3]

                if timestamp_diff > 0:
                    self.tracker.register_imu_measurement(0, imu_measurement)
        except Exception as e:
            logger.error(f"IMU thread error: {e}")

    def _camera_worker(self) -> None:
        """Worker thread for Camera data."""
        try:
            while not self.stop_event.is_set():
                try:
                    ir_frames = self.ir_pipe.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    continue
                    
                ir_left_frame = ir_frames.get_infrared_frame(1)
                ir_right_frame = ir_frames.get_infrared_frame(2)
                current_timestamp = int(ir_left_frame.timestamp * 1e6)

                self.timestamp_helper.prev_low_rate_timestamp = deepcopy(current_timestamp)
                
                images = (
                    np.asanyarray(ir_left_frame.get_data()),
                    np.asanyarray(ir_right_frame.get_data())
                )

                odom_pose_estimate, _ = self.tracker.track(current_timestamp, images)
                odom_pose = odom_pose_estimate.world_from_rig.pose

                self.queue.put([current_timestamp, odom_pose, images])
                self.timestamp_helper.last_low_rate_timestamp = current_timestamp
        except Exception as e:
            logger.error(f"Camera thread error: {e}")
