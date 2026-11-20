"""RCS environment for a DROID setup (FR3 + Robotiq 2F-85 + ZED cameras) and MolmoBot obs/action conversion."""

import cv2
import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
from rcs._core.common import BaseCameraConfig
from rcs.envs.base import ControlMode, RelativeTo
from rcs_fr3.configs import DefaultFR3HardwareEnv
from rcs_fr3.creators import HardwareCameraCreatorConfig, RCSFR3ConfigEnvCreator
from rcs_robotiq2f85.hw import RobotiQ2F85GripperConfig

import rcs
from rcs import common

# MolmoBot is trained on MolmoSpaces sim, where the gripper joint ranges 0 (open) .. 0.824 (closed).
MLSPACES_MAX_GRIP = 0.824033


def _vec7(values, name: str) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float64)
    if vec.shape != (7,) or np.any(vec < 0):
        raise ValueError(f"robot.{name} must be 7 non-negative values, got {list(values)}")
    return vec


def make_env(cfg: DictConfig) -> gym.Env:
    rcfg = cfg.robot
    base = DefaultFR3HardwareEnv()
    base.ip = rcfg.ip
    env_cfg = base.config()

    robot_cfg = env_cfg.robot_cfg
    robot_cfg.async_control = True  # stream joint targets to RCS's background joint impedance controller
    robot_cfg.ignore_realtime = rcfg.ignore_realtime
    robot_cfg.policy_rate = int(rcfg.control_hz)
    robot_cfg.speed_factor = rcfg.home_speed_factor
    robot_cfg.q_home = np.asarray(rcfg.home_jointpos, dtype=np.float64)
    robot_cfg.tcp_offset = rcs.GRIPPER_TCP_OFFSETS[common.GripperType("Robotiq2F85")]
    robot_cfg.kp = _vec7(rcfg.kp, "kp")
    robot_cfg.kd = _vec7(rcfg.kd, "kd")
    robot_cfg.torque_limit = _vec7(rcfg.effort_limit, "effort_limit")
    robot_cfg.allow_high_collision = bool(rcfg.allow_high_collision)

    env_cfg.gripper_cfg = RobotiQ2F85GripperConfig(
        serial_number=str(rcfg.gripper.serial),
        speed=rcfg.gripper.speed,
        force=rcfg.gripper.force,
        async_control=True,
    )
    env_cfg.camera_cfgs = {
        "zed": HardwareCameraCreatorConfig(
            camera_type_id="zed",
            camera_cfgs={
                name: BaseCameraConfig(
                    identifier=str(serial),
                    resolution_width=rcfg.camera_width,
                    resolution_height=rcfg.camera_height,
                    frame_rate=rcfg.camera_fps,
                )
                for name, serial in rcfg.cameras.items()
            },
            kwargs={"enable_depth": False, "enable_imu": False},
        )
    }
    env_cfg.control_mode = ControlMode.JOINTS
    env_cfg.relative_to = RelativeTo.NONE  # MolmoBot outputs absolute joint positions
    env_cfg.max_relative_movement = float(rcfg.max_joint_delta_per_step)
    env_cfg.frequency = float(rcfg.control_hz)  # env.step() is rate limited to this
    env_cfg.wrapper_cfg.binary_gripper = True  # the server already binarizes the gripper to {0, 255}
    env_cfg.wrapper_cfg.home_on_reset = True
    env_cfg.wrapper_cfg.include_depth = False

    return RCSFR3ConfigEnvCreator().create_env(env_cfg)


def close_env(env: gym.Env) -> None:
    """env.close() with two RCS shutdown races worked around."""
    # HardwareCameraSet.close() closes the ZEDs before joining its polling thread, which can segfault mid-grab.
    # Stopping (joining) the thread first makes close() only close the cameras.
    try:
        env.get_wrapper_attr("camera_set").stop()
    except AttributeError:
        pass
    # robotiq2f's async status thread polls forever and has no stop; once RCS closes the serial port under it,
    # it dies with "Bad file descriptor". Make it exit quietly first (SystemExit ends a thread silently).
    try:
        driver = env.get_wrapper_attr("gripper").gripper
    except AttributeError:
        driver = None
    thread = getattr(driver, "_async_thread", None)
    if thread is not None:

        def _stop_polling():
            raise SystemExit

        driver.read_status_sync = _stop_polling
        thread.join(timeout=1.0)
    env.close()


def camera_frames(obs: dict) -> dict[str, np.ndarray]:
    """Full-resolution RGB frames from an RCS observation."""
    return {name: data["rgb"]["data"] for name, data in obs["frames"].items()}


def to_policy_obs(obs: dict, task: str, image_width: int, image_height: int, camera_names: list[str]) -> dict:
    """RCS observation -> MolmoBot server observation."""
    width_open = float(obs["gripper"][0])  # RCS: 1 = fully open, 0 = closed
    grip = (1.0 - np.clip(width_open, 0.0, 1.0)) * MLSPACES_MAX_GRIP
    policy_obs = {
        "task": task,
        "qpos": {
            "arm": np.asarray(obs["joints"], dtype=np.float64),
            "gripper": np.full(2, grip),
        },
    }
    frames = camera_frames(obs) if "frames" in obs else {}
    for name in camera_names:
        if name not in frames:
            raise CameraFailure(f"No frame from camera {name}")
        frame = frames[name]
        if np.std(frame.mean(axis=-1)) < 1.0:
            # ZEDs sometimes fail silently and return single-color frames.
            raise CameraFailure(f"Camera {name} returned a constant-color frame")
        policy_obs[name] = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
    return policy_obs


def to_rcs_action(action: dict, fixed_gripper: str | None = None) -> dict:
    """MolmoBot server action -> RCS action. Gripper: server 0 = open, 255 = closed.

    fixed_gripper ("open"/"closed") overrides the gripper for arm-only policies, which return no gripper action.
    """
    if fixed_gripper is not None:
        if fixed_gripper not in ("open", "closed"):
            raise ValueError(f"robot.gripper.fixed_command must be null, 'open' or 'closed', got {fixed_gripper!r}")
        gripper_open = 1.0 if fixed_gripper == "open" else 0.0
    else:
        gripper = np.asarray(action.get("gripper", [])).reshape(-1)
        if gripper.size == 0:
            raise ValueError(
                "The policy returned no gripper action (arm-only policy). "
                "Set robot.gripper.fixed_command to 'open' or 'closed'."
            )
        gripper_open = 1.0 - float(gripper[0]) / 255.0
    return {
        "joints": np.asarray(action["arm"][:7], dtype=np.float64),
        "gripper": np.array([gripper_open], dtype=np.float32),
    }


class CameraFailure(RuntimeError):
    pass
