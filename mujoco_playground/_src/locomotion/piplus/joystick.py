# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for PiPlus humanoid."""

_MAX_SYM_DELAY = 25  # steps; covers half-period at lowest gait freq (1.3 Hz, dt=0.02 → ~19 steps)

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src.locomotion.piplus import base as piplus_base
from mujoco_playground._src.locomotion.piplus import piplus_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.5,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,  # rad
              joint_vel=0.2,   # rad/s
              gravity=0.03,
              linvel=0.1,
              gyro=0.05,
              last_act=0.01,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking rewards.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Base rewards.
              lin_vel_z=0.0,
              ang_vel_xy=-0.15,
              orientation=-0.8,
              base_height=0.0,
              # Energy rewards.
              torques=-2.5e-4,
              action_rate=-0.01,
              energy=-1.0e-4,
              # Feet rewards.
              feet_clearance=0.0,
              feet_air_time=2.0,
              feet_slip=-0.25,
              feet_height=0.0,
              feet_phase=1.5,
              feet_level=-5.0,
              #feet_contact_force=-0.001,
              # Other rewards.
              stand_still=0.0,
              alive=0.0,
              termination=-10.0,
              symmetry=-0.02,
              feet_x_symmetry=-0.05,
              foot_impact=-0.05,
              # Pose rewards.
              joint_deviation_hip=-0.0,
              joint_deviation_knee=0.0,
              dof_pos_limits=-1.0,
              pose=-1.0,
          ),
          tracking_sigma=0.5,
          max_foot_height=0.06,
          base_height_target=0.35,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.05, 1.0],
      ),
      lin_vel_x=[-1.0, 1.0],
      lin_vel_y=[-0.5, 0.5],
      ang_vel_yaw=[-1.0, 1.0],
  )


class Joystick(piplus_base.PiplusEnv):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    # Hip indices (hip_pitch + hip_roll per side).
    # Joint order in qpos[7:]: r_hip_pitch(0), r_hip_roll(1), r_thigh(2),
    #   r_calf(3), r_ankle_pitch(4), r_ankle_roll(5),
    #   l_hip_pitch(6), l_hip_roll(7), l_thigh(8), l_calf(9),
    #   l_ankle_pitch(10), l_ankle_roll(11).
    self._hip_indices = jp.array([
        self._mj_model.joint("r_hip_pitch_joint").qposadr - 7,
        self._mj_model.joint("r_hip_roll_joint").qposadr - 7,
        self._mj_model.joint("l_hip_pitch_joint").qposadr - 7,
        self._mj_model.joint("l_hip_roll_joint").qposadr - 7,
    ])

    # Knee equivalent: calf joints.
    self._knee_indices = jp.array([
        self._mj_model.joint("r_calf_joint").qposadr - 7,
        self._mj_model.joint("l_calf_joint").qposadr - 7,
    ])

    # Pose weights: lower weight for thigh (twist) joints.
    # Order matches qpos[7:]: r_leg(6), l_leg(6).
    self._weights = jp.array([
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # r leg
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # l leg
    ])

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._site_id = self._mj_model.site("imu").id

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )
    self._feet_body_id = np.array(
        [self._mj_model.geom_bodyid[gid] for gid in self._feet_geom_id]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._double_support_phase = jp.array([-jp.pi / 2, jp.pi / 2])

    # Action mirror symmetry: swap right/left halves, negate roll and twist joints.
    # Joint order per leg: hip_pitch, hip_roll, thigh(twist), calf, ankle_pitch, ankle_roll
    self._mirror_indices = jp.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])
    self._mirror_signs = jp.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # Randomize initial xy position and yaw.
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # Randomize joint positions.
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
        qpos[7:] * jax.random.uniform(key, (12,), minval=0.5, maxval=1.5)
    )

    # Randomize initial velocity.
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

    # Gait phase.
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.3, maxval=1.6)
    phase_dt = 2 * jp.pi * self.dt * gait_freq

    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng)

    # Per-episode joint calibration bias (2° std), added to motor reference each step.
    rng, key = jax.random.split(rng)
    joint_bias = jax.random.normal(key, (12,)) * (2.0 * jp.pi / 180.0)

    # Per-episode IMU mounting bias in roll and pitch (0.06 rad std).
    rng, key = jax.random.split(rng)
    imu_bias_rp = jax.random.normal(key, (2,)) * 0.01

    phase = jp.array([0.0, jp.pi])

    # Push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    # IMU and action delays.
    rng, imu_rng = jax.random.split(rng)
    max_imu_delay = 2
    min_imu_delay = 0
    max_action_delay = 3
    min_action_delay = 1

    imu_delay = jax.random.randint(
        imu_rng, minval=min_imu_delay, maxval=max_imu_delay, shape=()
    )
    action_delay = jax.random.randint(
        imu_rng, minval=min_action_delay, maxval=max_action_delay, shape=()
    )

    half_period_steps = jp.round(jp.pi / jp.squeeze(phase_dt)).astype(jp.int32)

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        "phase_dt": phase_dt,
        "phase": phase,
        "phase_frozen": jp.zeros((), dtype=bool),
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
        "imu_delay": imu_delay,
        "imu_buffer": jp.broadcast_to(jp.eye(3), (max_imu_delay, 3, 3)),
        "gyro_buffer": jp.zeros((max_imu_delay, 3)),
        "action_delay": action_delay,
        "action_buffer": jp.zeros((max_action_delay, self.mjx_model.nu)),
        "sym_buffer": jp.zeros((_MAX_SYM_DELAY, self.mjx_model.nu)),
        "sym_cmd_buffer": jp.zeros((_MAX_SYM_DELAY, 4)),
        "half_period_steps": half_period_steps,
        "joint_bias": joint_bias,
        "imu_bias_rp": imu_bias_rp,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # Delay action.
    state.info["action_buffer"] = jp.concatenate(
        [state.info["action_buffer"][1:], action[None, ...]], axis=0
    )
    delayed_action = state.info["action_buffer"][state.info["action_delay"]]

    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
        == 0
    )
    push *= self._config.push_config.enable
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    data = state.data.replace(qvel=qvel)
    state = state.replace(data=data)

    motor_targets = self._default_pose + state.info["joint_bias"] + delayed_action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    # Update IMU buffer.
    state.info["imu_buffer"] = jp.concatenate(
        [
            state.info["imu_buffer"][1:],
            data.site_xmat[self._site_id][None, ...],
        ],
        axis=0,
    )
    state.info["gyro_buffer"] = jp.concatenate(
        [state.info["gyro_buffer"][1:], self.get_gyro(data)[None]],
        axis=0,
    )

    # Update symmetry buffers: newest entry at end, oldest at start.
    state.info["sym_buffer"] = jp.concatenate(
        [state.info["sym_buffer"][1:], action[None, ...]], axis=0
    )
    state.info["sym_cmd_buffer"] = jp.concatenate(
        [state.info["sym_cmd_buffer"][1:], state.info["command"][None, ...]], axis=0
    )

    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["push"] = push
    state.info["step"] += 1
    state.info["push_step"] += 1

    # Phase update.
    is_stop = state.info["command"][3] > 0.5

    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    phase_tp1 = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi

    # Freeze phase at nearest symmetric point when stop is commanded.
    # Uses phase proximity (not contact) to avoid coupling reward/state.
    ds1 = jp.array([-jp.pi / 2, jp.pi / 2])
    ds2 = jp.array([jp.pi / 2, -jp.pi / 2])
    tol = jp.squeeze(state.info["phase_dt"]) * 3

    def _max_circ_dist(a, b):
      d = jp.abs(a - b)
      return jp.max(jp.minimum(d, 2 * jp.pi - d))

    d1 = _max_circ_dist(phase_tp1, ds1)
    d2 = _max_circ_dist(phase_tp1, ds2)
    near_ds = (d1 < tol) | (d2 < tol)
    should_freeze = is_stop & (near_ds | state.info["phase_frozen"])
    nearest_ds = jp.where(d1 <= d2, ds1, ds2)
    state.info["phase"] = jp.where(should_freeze, nearest_ds, phase_tp1)
    state.info["phase_frozen"] = should_freeze

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    state.info["command"] = jp.where(
        state.info["step"] > 500,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500), 0, state.info["step"]
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_gravity(data)[-1] < 0.0
    return (
        fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    # IMU mounting bias rotation: Ry(pitch) @ Rx(roll).
    roll, pitch = info["imu_bias_rp"]
    cr, sr = jp.cos(roll), jp.sin(roll)
    cp, sp = jp.cos(pitch), jp.sin(pitch)
    imu_bias_rot = jp.array([
        [cp,  sp * sr,  sp * cr],
        [0.0, cr,      -sr     ],
        [-sp, cp * sr,  cp * cr],
    ])

    true_gyro = info["gyro_buffer"][info["imu_delay"]]
    gyro = imu_bias_rot @ true_gyro
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + jax.random.normal(noise_rng, shape=gyro.shape)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    true_gravity = info["imu_buffer"][info["imu_delay"]].T @ jp.array([0, 0, -1])
    gravity = imu_bias_rot @ true_gravity
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + jax.random.normal(noise_rng, shape=gravity.shape)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    noisy_joint_vel = jp.zeros_like(joint_vel)

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    linvel = self.get_local_linvel(data)

    noisy_last_act = (
        info["last_act"]
        + jax.random.normal(noise_rng, shape=info["last_act"].shape)
        * self._config.noise_config.level
        * self._config.noise_config.scales.last_act
    )

    state = jp.hstack([
        noisy_gyro,                                    # 3
        noisy_gravity,                                 # 3
        info["command"],                               # 4
        noisy_joint_angles - self._default_pose - info["joint_bias"],  # 12
        noisy_joint_vel,                               # 12
        noisy_last_act,                                # 12
        phase,                                         # 4
    ])

    accelerometer = self.get_accelerometer(data)
    global_angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]

    privileged_state = jp.hstack([
        state,
        true_gyro,                                     # 3 (true, no bias)
        accelerometer,                                 # 3
        true_gravity,                                  # 3 (true, no bias)
        linvel,                                        # 3
        global_angvel,                                 # 3
        joint_angles - self._default_pose,                               # 12 (true, no bias/noise)
        joint_vel,                                     # 12
        root_height,                                   # 1
        data.actuator_force,                           # 12
        contact,                                       # 2
        feet_vel,                                      # 2*3
        info["feet_air_time"],                         # 2
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics
    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_gravity(data)),
        "base_height": self._cost_base_height(data.qpos[2]),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data, info),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "feet_phase": self._reward_feet_phase(
            data,
            info["phase"],
            self._config.reward_config.max_foot_height,
            info["command"],
        ),
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            data.qpos[7:], info["command"]
        ),
        "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "pose": self._cost_pose(data.qpos[7:]),
        "feet_level": self._cost_feet_level(data),
        #"feet_contact_force": self._cost_feet_contact_force(data, first_contact),
        "symmetry": self._cost_action_symmetry(action, info),
        "feet_x_symmetry": self._cost_feet_x_symmetry(data),
        "foot_impact": self._cost_foot_impact(
            data.sensordata[self._foot_linvel_sensor_adr], first_contact
        ),
    }

  def _reward_tracking_lin_vel(
      self, commands: jax.Array, local_vel: jax.Array
  ) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self, commands: jax.Array, ang_vel: jax.Array
  ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    return jp.square(
        base_height - self._config.reward_config.base_height_target
    )

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act
    return jp.sum(jp.square(act - last_act))

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_stand_still(
      self, commands: jax.Array, qpos: jax.Array
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands[:3])
    return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.1)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _reward_alive(self) -> jax.Array:
    return jp.array(1.0)

  def _cost_joint_deviation_hip(
      self, qpos: jax.Array, cmd: jax.Array
  ) -> jax.Array:
    cost = jp.sum(
        jp.abs(qpos[self._hip_indices] - self._default_pose[self._hip_indices])
    )
    cost *= jp.abs(cmd[1]) > 0.1
    return cost

  def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(
        jp.abs(
            qpos[self._knee_indices] - self._default_pose[self._knee_indices]
        )
    )

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos - self._default_pose) * self._weights)

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    del info
    body_vel = self.get_global_linvel(data)[:2]
    return jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)

  def _cost_feet_clearance(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    del info
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    del info
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact)

  def _reward_feet_air_time(
      self,
      air_time: jax.Array,
      first_contact: jax.Array,
      commands: jax.Array,
      threshold_min: float = 0.2,
      threshold_max: float = 0.5,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands[:3])
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    reward *= cmd_norm > 0.1
    return reward

  def _reward_feet_phase(
      self,
      data: mjx.Data,
      phase: jax.Array,
      foot_height: jax.Array,
      commands: jax.Array,
  ) -> jax.Array:
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(jp.clip(rz - foot_z, a_min=0.0)))
    walk_in_place = jp.linalg.norm(commands[:3]) < 0.05
    reward = jp.exp(-error / 0.01) * (1.0 + 0.25 * walk_in_place)
    reward *= commands[3] < 0.5
    return reward

  def _cost_feet_level(self, data: mjx.Data) -> jax.Array:
    # Foot pitch penalty: penalize non-level feet when under the body.
    # When the foot is far forward/backward the penalty is relaxed.
    foot_xmat = data.site_xmat[self._feet_site_id].reshape(-1, 3, 3)
    # Third column of rotation matrix = local Z axis (foot normal) in world frame.
    foot_normal_world = foot_xmat[:, :, 2]  # shape (2, 3)
    # Pitch error only (X component); ignore roll (Y component).
    pitch_error = jp.square(foot_normal_world[:, 0])  # shape (2,)

    # Foot position relative to body in body frame.
    body_pos = data.qpos[:3]
    body_xmat = data.xmat[self._torso_body_id].reshape(3, 3)
    foot_pos = data.site_xpos[self._feet_site_id]  # shape (2, 3)
    foot_rel_body = (body_xmat.T @ (foot_pos - body_pos).T).T  # shape (2, 3)
    foot_forward = foot_rel_body[:, 0]  # forward/backward distance in body frame

    # Penalty weight: high when foot is under the body, low when foot is far forward/backward.
    weight = jp.exp(-jp.square(foot_forward) / (0.07 ** 2))

    return jp.sum(pitch_error * weight)

  def _cost_feet_contact_force(
      self, data: mjx.Data, first_contact: jax.Array
  ) -> jax.Array:
    # cfrc_ext layout per body: [torque(3), force(3)] in world frame.
    forces = data.cfrc_ext[self._feet_body_id, 3:]  # (2, 3)
    force_norms = jp.linalg.norm(forces, axis=-1)    # (2,)
    return jp.sum(jp.square(force_norms) * first_contact)

  def _cost_action_symmetry(
      self, action: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    half_steps = info["half_period_steps"]
    read_idx = _MAX_SYM_DELAY - 1 - half_steps
    delayed_action = info["sym_buffer"][read_idx]
    delayed_cmd = info["sym_cmd_buffer"][read_idx]
    mirrored = action[self._mirror_indices] * self._mirror_signs
    error = jp.sum(jp.square(mirrored - delayed_action))
    # Suppress penalty until the buffer has accumulated enough history.
    has_data = info["step"] >= half_steps
    # Suppress penalty if the command changed between the delayed and current step.
    same_cmd = jp.all(jp.abs(delayed_cmd - info["command"]) < 1e-6)
    return error * has_data * same_cmd

  def _cost_feet_x_symmetry(self, data: mjx.Data) -> jax.Array:
    # In a symmetric gait, right foot x + left foot x ≈ 0 in the local body frame.
    body_pos = data.qpos[:3]
    body_xmat = data.xmat[self._torso_body_id].reshape(3, 3)
    foot_pos = data.site_xpos[self._feet_site_id]  # (2, 3) world frame
    foot_pos_local = (body_xmat.T @ (foot_pos - body_pos).T).T  # (2, 3)
    foot_x = foot_pos_local[:, 0]  # forward/backward per foot
    return jp.square(foot_x[0] + foot_x[1])

  def _cost_foot_impact(
      self, feet_vel: jax.Array, first_contact: jax.Array
  ) -> jax.Array:
    # Penalize downward foot velocity at the moment of first ground contact.
    foot_z_vel = feet_vel[:, 2]  # (2,); negative when foot is moving down
    return jp.sum(jp.square(foot_z_vel) * first_contact)

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    lin_vel_x = jax.random.uniform(
        rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.ang_vel_yaw[0],
        maxval=self._config.ang_vel_yaw[1],
    )

    normal_cmd = jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw, jp.zeros(1)])
    zero_walk_cmd = jp.zeros(4)
    stop_cmd = jp.array([0.0, 0.0, 0.0, 1.0])

    # 10% stop, 10% walk-in-place, 80% normal.
    u = jax.random.uniform(rng4)
    cmd = jp.where(u < 0.1, stop_cmd, jp.where(u < 0.2, zero_walk_cmd, normal_cmd))
    return cmd
