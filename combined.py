"""
Combined Aero System - VIO Tracker with Camera/IMU Integration
Combines all modules: Core Tracker, Camera Utilities, and Visualization
"""

import queue
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Callable
import logging
import sys
import importlib.util

import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation
import rerun as rr
import rerun.blueprint as rrb

# ============================================================================
# BINDINGS MODULE - pycuvslam binding wrapper
# ============================================================================

# Try to import the extension module
try:
    from aero.core import pycuvslam
except ImportError:
    try:
        import pycuvslam
    except ImportError:
        # Check if the .so file exists in this directory
        _current_dir = Path(__file__).parent / "core"
        _so_path = _current_dir / "pycuvslam.so"
        
        if _so_path.exists():
            spec = importlib.util.spec_from_file_location("pycuvslam", _so_path)
            if spec and spec.loader:
                pycuvslam = importlib.util.module_from_spec(spec)
                sys.modules["pycuvslam"] = pycuvslam
                spec.loader.exec_module(pycuvslam)
            else:
                raise ImportError(f"Could not load extension from {_so_path}")
        else:
            raise ImportError("pycuvslam module not found. Ensure it is installed or the .so file is present.")

# Re-export everything from pycuvslam
for _name in dir(pycuvslam):
    if not _name.startswith("__"):
        globals()[_name] = getattr(pycuvslam, _name)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Interface constants
DEFAULT_RESOLUTION = (640, 360)
DEFAULT_FPS = 30
DEFAULT_IMU_FREQ_ACCEL = 200
DEFAULT_IMU_FREQ_GYRO = 200
IMAGE_JITTER_THRESHOLD_MS = 35 * 1e6
IMU_JITTER_THRESHOLD_MS = 6 * 1e6

# Camera constants
DEFAULT_IMU_FREQUENCY = 200

# IMU noise parameters for RealSense
IMU_GYROSCOPE_NOISE_DENSITY = 6.0673370376614875e-03
IMU_GYROSCOPE_RANDOM_WALK = 3.6211951458325785e-05
IMU_ACCELEROMETER_NOISE_DENSITY = 3.3621979208052800e-02
IMU_ACCELEROMETER_RANDOM_WALK = 9.8256589971851467e-04

# Visualization constants
DEFAULT_NUM_VIZ_CAMERAS = 1
POINT_RADIUS = 5.0
ARROW_SCALE = 0.1
GRAVITY_ARROW_SCALE = 0.02

# ============================================================================
# TRACKER MODULE - cuVSLAM Tracker wrapper
# ============================================================================

class Tracker:
    """
    A wrapper that combines cuVSLAM Odometry and SLAM functionality.
    
    This class automatically manages both the Odometry and SLAM instances,
    providing a simplified interface for common use cases.
    """

    # Expose inner classes of Odometry & Slam
    # We use getattr to avoid errors if bindings are mocked or incomplete
    OdometryMode = getattr(pycuvslam.Odometry, "OdometryMode", None)
    MulticameraMode = getattr(pycuvslam.Odometry, "MulticameraMode", None)
    OdometryConfig = getattr(pycuvslam.Odometry, "Config", None)
    OdometryRGBDSettings = getattr(pycuvslam.Odometry, "RGBDSettings", None)

    SlamConfig = getattr(pycuvslam.Slam, "Config", None) if hasattr(pycuvslam, "Slam") else None
    SlamMetrics = getattr(pycuvslam.Slam, "Metrics", None) if hasattr(pycuvslam, "Slam") else None
    SlamLocalizationSettings = getattr(pycuvslam.Slam, "LocalizationSettings", None) if hasattr(pycuvslam, "Slam") else None

    def __init__(self, rig: pycuvslam.Rig, odom_config: Optional[Any] = None, slam_config: Optional[Any] = None) -> None:
        """
        Initialize the cuVSLAM system.

        Args:
            rig: Camera rig configuration
            odom_config: Optional odometry configuration (uses defaults if omitted)
            slam_config: Optional SLAM configuration (disables SLAM if None)
        """
        if odom_config is None:
            odom_config = self.OdometryConfig()

        # need to export observations/landmarks for Slam
        if slam_config is not None:
            odom_config.enable_observations_export = True
            odom_config.enable_landmarks_export = True
        # Create odometry
        self.odom = pycuvslam.Odometry(rig, odom_config)
        # Create SLAM
        primary_cams = self.odom.get_primary_cameras()
        self.slam = pycuvslam.Slam(rig, primary_cams, slam_config) if slam_config is not None and hasattr(pycuvslam, "Slam") else None

    def track(self, timestamp: int, images: List[Any], masks: Optional[List[Any]] = None, depths: Optional[List[Any]] = None) -> Tuple[pycuvslam.PoseEstimate, Optional[pycuvslam.Pose]]:
        """
        Track a rig pose using current image frame.

        This method combines tracking and SLAM processing in a single call.

        Args:
            timestamp: Images timestamp in nanoseconds
            images: List of numpy arrays containing the camera images
            masks: Optional list of numpy arrays containing masks for the images
            depths: Optional list of numpy arrays containing depth images

        Returns:
            PoseEstimate: The computed pose estimate from Odometry
            Pose: The computed pose estimate from SLAM
        """
        pose_estimate = self.odom.track(timestamp, images, masks, depths)

        slam_pose = None
        if self.slam:
            state = self.odom.get_state()
            slam_pose = self.slam.track(state)

        return pose_estimate, slam_pose

    def register_imu_measurement(self, sensor_index: int, imu_measurement: pycuvslam.ImuMeasurement) -> None:
        """Register an IMU measurement with the tracker."""
        self.odom.register_imu_measurement(sensor_index, imu_measurement)

    def get_last_observations(self, camera_index: int) -> List[Any]:
        """Get observations from the last frame."""
        return self.odom.get_last_observations(camera_index)

    def get_last_landmarks(self) -> List[Any]:
        """Get landmarks from the last frame."""
        return self.odom.get_last_landmarks()

    def get_last_gravity(self) -> Optional[Any]:
        """Get gravity vector from the last frame."""
        return self.odom.get_last_gravity()

    def get_final_landmarks(self) -> dict[int, Any]:
        """Get all final landmarks from all frames."""
        return self.odom.get_final_landmarks()

    def get_all_slam_poses(self, max_poses_count: int = 0) -> Optional[List[Any]]:
        """Get all SLAM poses for each frame."""
        return self.slam.get_all_slam_poses(max_poses_count) if self.slam else None

    def set_slam_pose(self, pose: pycuvslam.Pose) -> None:
        """Set the rig SLAM pose to a value provided by user."""
        if self.slam:
            self.slam.set_slam_pose(pose)

    def save_map(self, folder_name: str, callback: Callable[[bool], None]) -> None:
        """Save SLAM database (map) to folder asynchronously."""
        if self.slam:
            self.slam.save_map(folder_name, callback)

    def localize_in_map(self, folder_name: str, guess_pose: pycuvslam.Pose, images: List[Any], settings: Any, callback: Callable[[Optional[pycuvslam.Pose], str], None]) -> None:
        """Localize in the existing database (map) asynchronously."""
        if self.slam:
            self.slam.localize_in_map(folder_name, guess_pose, images, settings, callback)

    def get_pose_graph(self) -> Optional[Any]:
        """Get pose graph."""
        return self.slam.get_pose_graph() if self.slam else None

    def get_slam_metrics(self) -> Optional[Any]:
        """Get SLAM metrics."""
        return self.slam.get_slam_metrics() if self.slam else None

    def get_loop_closure_poses(self) -> Optional[List[Any]]:
        """Get list of last 10 loop closure poses with timestamps."""
        return self.slam.get_loop_closure_poses() if self.slam else None

    @staticmethod
    def merge_maps(rig: pycuvslam.Rig, databases: List[str], output_folder: str) -> None:
        """Merge existing maps into one map."""
        if hasattr(pycuvslam, "Slam"):
            pycuvslam.Slam.merge_maps(rig, databases, output_folder)

# ============================================================================
# CAMERA UTILITIES MODULE
# ============================================================================

def opengl_to_opencv_transform(
    rotation: np.ndarray, translation: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert from OpenGL coordinate system to OpenCV coordinate system.
    
    Args:
        rotation: 3x3 rotation matrix in OpenGL coordinates
        translation: 3x1 translation vector in OpenGL coordinates
        
    Returns:
        Tuple of (rotation_opencv, translation_opencv)
    """
    transform_matrix = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    rotation_opencv = transform_matrix @ rotation @ transform_matrix.T
    translation_opencv = transform_matrix @ translation
    return rotation_opencv, translation_opencv


def transform_to_pose(transform_matrix=None) -> pycuvslam.Pose:
    """Convert a transformation matrix to a pycuvslam.Pose object.
    
    Args:
        transform_matrix: Either a RealSense transform object or a list of lists
                         representing the transformation matrix
                         
    Returns:
        pycuvslam.Pose object
    """
    if isinstance(transform_matrix, List):
        # Handle list of lists format from YAML
        rotation = np.array([row[:3] for row in transform_matrix])
        translation = np.array([row[3] for row in transform_matrix])
        rotation_opencv, translation_vec = opengl_to_opencv_transform(
            rotation, translation
        )
        rotation_quat = Rotation.from_matrix(rotation_opencv).as_quat()
    elif transform_matrix:
        # Handle RealSense transform object
        rotation_matrix = np.array(transform_matrix.rotation).reshape([3, 3])
        translation_vec = transform_matrix.translation
        rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()
        return pycuvslam.Pose(rotation=rotation_quat, translation=translation_vec)
    else:
        # Default identity transform
        rotation_matrix = np.eye(3)
        translation_vec = [0] * 3
        rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()
    
    return pycuvslam.Pose(rotation=rotation_quat, translation=translation_vec)


def rig_from_imu_pose(rs_transform=None) -> pycuvslam.Pose:
    """Convert IMU pose from OpenGL to OpenCV coordinate system.
    
    Args:
        rs_transform: RealSense transform object
        
    Returns:
        pycuvslam.Pose object in OpenCV coordinates
    """
    rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()
    translation_vec = rotation_matrix @ rs_transform.translation
    return pycuvslam.Pose(rotation=rotation_quat, translation=translation_vec)


def get_rs_camera(
    rs_intrinsics, transform_matrix: Optional[Any] = None
) -> pycuvslam.Camera:
    """Create a Camera object from RealSense intrinsics.
    
    Args:
        rs_intrinsics: RealSense intrinsics object
        transform_matrix: Optional transformation matrix for camera pose
        
    Returns:
        pycuvslam.Camera object
    """
    cam = pycuvslam.Camera()
    cam.distortion = pycuvslam.Distortion(pycuvslam.Distortion.Model.Pinhole)
    cam.focal = rs_intrinsics.fx, rs_intrinsics.fy
    cam.principal = rs_intrinsics.ppx, rs_intrinsics.ppy
    cam.size = rs_intrinsics.width, rs_intrinsics.height
    
    if transform_matrix is not None:
        cam.rig_from_camera = transform_to_pose(transform_matrix)
    
    return cam


def get_rs_imu(
    imu_extrinsics, frequency: int = DEFAULT_IMU_FREQUENCY
) -> pycuvslam.ImuCalibration:
    """Create an IMU calibration object from RealSense extrinsics.
    
    Args:
        imu_extrinsics: RealSense IMU extrinsics
        frequency: IMU sampling frequency in Hz
        
    Returns:
        pycuvslam.ImuCalibration object
    """
    imu = pycuvslam.ImuCalibration()
    imu.rig_from_imu = rig_from_imu_pose(imu_extrinsics)
    imu.gyroscope_noise_density = IMU_GYROSCOPE_NOISE_DENSITY
    imu.gyroscope_random_walk = IMU_GYROSCOPE_RANDOM_WALK
    imu.accelerometer_noise_density = IMU_ACCELEROMETER_NOISE_DENSITY
    imu.accelerometer_random_walk = IMU_ACCELEROMETER_RANDOM_WALK
    imu.frequency = frequency
    return imu


def setup_pipeline(
    serial_number: str,
    resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
    fps: int = DEFAULT_FPS
) -> Tuple[rs.pipeline, rs.config]:
    """Set up and configure a RealSense pipeline.
    
    Args:
        serial_number: Camera serial number
        resolution: Camera resolution as (width, height)
        fps: Frames per second
        
    Returns:
        Tuple of (pipeline, config)
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(
        rs.stream.infrared, 1, resolution[0], resolution[1], rs.format.y8, fps
    )
    config.enable_stream(
        rs.stream.infrared, 2, resolution[0], resolution[1], rs.format.y8, fps
    )
    return pipeline, config


def get_camera_intrinsics(
    pipeline: rs.pipeline, config: rs.config
) -> Tuple[Any, Any]:
    """Get camera intrinsics from a RealSense pipeline.
    
    Args:
        pipeline: RealSense pipeline
        config: RealSense config
        
    Returns:
        Tuple of (left_intrinsics, right_intrinsics)
    """
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    left_intrinsics = frames[0].profile.as_video_stream_profile().intrinsics
    right_intrinsics = frames[1].profile.as_video_stream_profile().intrinsics
    pipeline.stop()
    return left_intrinsics, right_intrinsics


def configure_device(
    pipeline: rs.pipeline, config: rs.config, is_master: bool = False
) -> None:
    """Configure device settings like IR emitter and sync mode.
    
    Args:
        pipeline: RealSense pipeline
        config: RealSense config
        is_master: Whether this device is the master for synchronization
    """
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    depth_sensor = device.query_sensors()[0]
    
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)
        # First camera is master, others are slave
        sync_mode = 1 if is_master else 2
        depth_sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)


def get_rs_stereo_rig(
    camera_params: Dict[str, Dict[str, Any]]
) -> pycuvslam.Rig:
    """Create a stereo Rig object from RealSense parameters.
    
    Args:
        camera_params: Dictionary containing camera parameters
        
    Returns:
        pycuvslam.Rig object
    """
    rig = pycuvslam.Rig()
    
    cameras = [get_rs_camera(camera_params['left']['intrinsics'])]
    
    if 'right' in camera_params:
        cameras.append(
            get_rs_camera(
                camera_params['right']['intrinsics'],
                camera_params['right']['extrinsics']
            )
        )
    
    rig.cameras = cameras
    return rig


def get_rs_multi_rig(
    camera_params: Dict[str, Dict[str, Dict[str, Any]]]
) -> pycuvslam.Rig:
    """Create a multi-camera Rig object from RealSense parameters.
    
    Args:
        camera_params: Dictionary containing parameters for multiple cameras
        
    Returns:
        pycuvslam.Rig object with multiple stereo cameras
    """
    rig = pycuvslam.Rig()
    cameras_list = []
    
    for i in range(1, len(camera_params) + 1):
        camera_idx = f"camera_{i}"
        
        # Add left camera
        cameras_list.append(
            get_rs_camera(
                camera_params[camera_idx]['left']['intrinsics'],
                camera_params[camera_idx]['left']['extrinsics']
            )
        )
        
        # Add right camera
        cameras_list.append(
            get_rs_camera(
                camera_params[camera_idx]['right']['intrinsics'],
                camera_params[camera_idx]['right']['extrinsics']
            )
        )
    
    rig.cameras = cameras_list
    return rig


def get_rs_vio_rig(
    camera_params: Dict[str, Dict[str, Any]]
) -> pycuvslam.Rig:
    """Create a VIO Rig object with cameras and IMU from RealSense parameters.
    
    Args:
        camera_params: Dictionary containing camera and IMU parameters
        
    Returns:
        pycuvslam.Rig object with cameras and IMU
    """
    rig = pycuvslam.Rig()
    rig.cameras = [
        get_rs_camera(camera_params['left']['intrinsics']),
        get_rs_camera(
            camera_params['right']['intrinsics'],
            camera_params['right']['extrinsics']
        )
    ]
    rig.imus = [get_rs_imu(camera_params['imu']['cam_from_imu'])]
    return rig

# ============================================================================
# VISUALIZATION MODULE - Rerun-based visualizer
# ============================================================================

class RerunVisualizer:
    """Rerun-based visualizer for cuVSLAM tracking results."""
    
    def __init__(self, num_viz_cameras: int = DEFAULT_NUM_VIZ_CAMERAS) -> None:
        """Initialize rerun visualizer.
        
        Args:
            num_viz_cameras: Number of cameras to visualize
        """
        self.num_viz_cameras = num_viz_cameras
        rr.init("cuVSLAM Visualizer", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        
        # Set up the visualization layout
        self._setup_blueprint()
        self.track_colors = {}

    def _setup_blueprint(self) -> None:
        """Set up the Rerun blueprint for visualization layout."""
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Horizontal(
                    column_shares=[0.5, 0.5],
                    contents=[
                        rrb.Vertical(contents=[
                            rrb.Spatial2DView(origin=f'world/camera_{i}')
                            for i in range(self.num_viz_cameras)
                        ]),
                        rrb.Spatial3DView(origin='world')
                    ]
                )
            ),
            make_active=True
        )

    def _log_rig_pose(
        self, rotation_quat: np.ndarray, translation: np.ndarray
    ) -> None:
        """Log rig pose to Rerun.
        
        Args:
            rotation_quat: Rotation quaternion
            translation: Translation vector
        """
        rr.log(
            "world/camera_0",
            rr.Transform3D(translation=translation, quaternion=rotation_quat),
            rr.Arrows3D(
                vectors=np.eye(3) * ARROW_SCALE,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ
            )
        )

    def _log_observations(
        self,
        observations_main_cam: List[pycuvslam.Observation],
        image: np.ndarray,
        camera_name: str
    ) -> None:
        """Log 2D observations for a specific camera with consistent colors.
        
        Args:
            observations_main_cam: List of observations
            image: Camera image
            camera_name: Name of the camera for logging
        """
        if not observations_main_cam:
            return

        # Assign random color to new tracks
        for obs in observations_main_cam:
            if obs.id not in self.track_colors:
                self.track_colors[obs.id] = np.random.randint(0, 256, size=3)

        points = np.array([[obs.u, obs.v] for obs in observations_main_cam])
        colors = np.array([
            self.track_colors[obs.id] for obs in observations_main_cam
        ])

        # Handle different image datatypes for compression
        if image.dtype == np.uint8:
            image_log = rr.Image(image).compress()
        else:
            # For other datatypes, don't compress to avoid issues
            image_log = rr.Image(image)

        rr.log(
            f"world/{camera_name}/observations",
            rr.Points2D(positions=points, colors=colors, radii=POINT_RADIUS),
            image_log
        )

    def _log_gravity(self, gravity: np.ndarray) -> None:
        """Log gravity vector to Rerun.
        
        Args:
            gravity: Gravity vector
        """
        rr.log(
            "world/camera_0/gravity",
            rr.Arrows3D(
                vectors=gravity,
                colors=[[255, 0, 0]],
                radii=GRAVITY_ARROW_SCALE
            )
        )

    def visualize_frame(
        self,
        frame_id: int,
        images: List[np.ndarray],
        pose: pycuvslam.Pose,
        observations_main_cam: List[List[pycuvslam.Observation]],
        trajectory: List[np.ndarray],
        timestamp: int,
        gravity: Optional[np.ndarray] = None
    ) -> None:
        """Visualize current frame state using Rerun.
        
        Args:
            frame_id: Current frame ID
            images: List of camera images
            pose: Current pose estimate
            observations_main_cam: List of observations for each camera
            trajectory: List of trajectory points
            timestamp: Current timestamp
            gravity: Optional gravity vector
        """
        rr.set_time_sequence("frame", frame_id)
        rr.log("world/trajectory", rr.LineStrips3D(trajectory), static=True)

        self._log_rig_pose(pose.rotation, pose.translation)
        
        for i in range(self.num_viz_cameras):
            self._log_observations(
                observations_main_cam[i], images[i], f"camera_{i}"
            )
            
        if gravity is not None:
            self._log_gravity(gravity)
            
        rr.log("world/timestamp", rr.TextLog(str(timestamp)))

# ============================================================================
# INTERFACE MODULE - Main AeroSystem class
# ============================================================================

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
        self.tracker: Optional[Tracker] = None
        self.rig: Optional[pycuvslam.Rig] = None
        
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
        self.latest_pose: Optional[pycuvslam.Pose] = None
        self.trajectory: List[np.ndarray] = []
        self.frame_id = 0
        
        # Visualization
        self.visualizer: Optional[RerunVisualizer] = None
        self.enable_viz = self.config.get("enable_visualization", True)

    def initialize(self) -> None:
        """Initialize the system, setup cameras, and create the tracker."""
        logger.info("Initializing AeroSystem...")
        
        # 1. Setup camera parameters (requires brief pipeline start)
        camera_params = self._setup_camera_parameters()
        
        # 2. Create Rig
        self.rig = get_rs_vio_rig(camera_params)
        
        # 3. Configure Tracker
        tracker_config = Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            debug_imu_mode=False,
            odometry_mode=Tracker.OdometryMode.Inertial,
            horizontal_stereo_camera=True
        )
        
        # 4. Initialize Tracker
        self.tracker = Tracker(self.rig, tracker_config)
        
        if self.enable_viz:
            self.visualizer = RerunVisualizer()
            
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

    def process_next_frame(self, timeout: float = 1.0) -> Optional[pycuvslam.Pose]:
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
                imu_measurement = pycuvslam.ImuMeasurement()
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


# ============================================================================
# MODULE EXPORTS - Make main classes available at module level
# ============================================================================

__all__ = [
    # Tracker
    'Tracker',
    # Camera utilities
    'opengl_to_opencv_transform',
    'transform_to_pose',
    'rig_from_imu_pose',
    'get_rs_camera',
    'get_rs_imu',
    'setup_pipeline',
    'get_camera_intrinsics',
    'configure_device',
    'get_rs_stereo_rig',
    'get_rs_multi_rig',
    'get_rs_vio_rig',
    # Visualization
    'RerunVisualizer',
    # Interface
    'AeroSystem',
    'ThreadWithTimestamp',
]
