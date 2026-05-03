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
"""Kick task for PiPlus humanoid."""

_MAX_ACTION_DELAY = 2
_MAX_SYM_DELAY = 25  # covers half-period at lowest gait freq (1.25 Hz, dt=0.02 → ~20 steps)

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
      episode_length=500,
      action_repeat=1,
      action_scale=0.5,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,   # rad
              joint_vel=0.2,    # rad/s
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,
              last_act=0.05,
              ball_pos=0.1,     # m
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Kick rewards.
              lin_vel_x=0.7,
              stop_for_kick=0.0,
              orient_to_ball=0.5,
              ball_proximity=0.0,
              ball_height=0.3,
              ball_travel=0.0,
              ball_speed=0.0,
              kick_foot_velocity=0.0,
              kick_motion=0.0,
              kick_direction=1.0,
              orient_to_kick_dir=0.8,
              wrong_approach=-0.5,
              symmetry=-0.05,
              # Base rewards.
              lin_vel_z=0.0,
              ang_vel_xy=-0.15,
              orientation=-1.0,
              base_height=0.0,
              # Energy rewards.
              torques=-2.5e-3,
              action_rate=-0.01,
              energy=-1.0e-3,
              # Feet rewards.
              feet_clearance=0.0,
              feet_air_time=2.0,
              feet_slip=-0.25,
              feet_height=0.0,
              feet_phase=1.0,
              feet_level=-5.0,
              # Other rewards.
              alive=0.0,
              termination=-1.0,
              stand_still=0.0,
              joint_deviation_hip=0.0,
              joint_deviation_knee=0.0,
              dof_pos_limits=-1.0,
              pose=-1.0,
          ),
          tracking_sigma=0.5,
          max_foot_height=0.08,
          base_height_target=0.35,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.05, 2.5],
      ),
      ball_distance=[0.25, 0.4],
  )


class Kick(piplus_base.PiplusEnv):
  """Walk to ball and kick it."""

  def __init__(
      self,
      task: str = "kick_flat_terrain",
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
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:19])

    self._lowers, self._uppers = self.mj_model.jnt_range[1:13].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    # Joint order in qpos[7:19]: r_hip_pitch(0), r_hip_roll(1), r_thigh(2),
    #   r_calf(3), r_ankle_pitch(4), r_ankle_roll(5),
    #   l_hip_pitch(6), l_hip_roll(7), l_thigh(8), l_calf(9),
    #   l_ankle_pitch(10), l_ankle_roll(11).
    self._hip_indices = jp.array([
        self._mj_model.joint("r_hip_pitch_joint").qposadr - 7,
        self._mj_model.joint("r_hip_roll_joint").qposadr - 7,
        self._mj_model.joint("l_hip_pitch_joint").qposadr - 7,
        self._mj_model.joint("l_hip_roll_joint").qposadr - 7,
    ])
    self._knee_indices = jp.array([
        self._mj_model.joint("r_calf_joint").qposadr - 7,
        self._mj_model.joint("l_calf_joint").qposadr - 7,
    ])
    # Kick leg: left. Indices into qpos[7:19].
    self._l_hip_pitch_idx = int(
        self._mj_model.joint("l_hip_pitch_joint").qposadr - 7
    )
    self._l_calf_idx = int(self._mj_model.joint("l_calf_joint").qposadr - 7)

    self._weights = jp.array([
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # r leg
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # l leg
    ])

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._site_id = self._mj_model.site("imu").id
    self._ball_site_id = self._mj_model.site("ball_center").id
    self._ball_body_id = self._mj_model.body("ball").id
    self._ball_geom_id = self._mj_model.geom("ball_geom").id
    self._ball_mass_base = float(self._mj_model.body_mass[self._ball_body_id])
    self._ball_radius_base = float(self._mj_model.geom_size[self._ball_geom_id, 0])
    self._ball_friction_base = jp.array(self._mj_model.geom_friction[self._ball_geom_id])
    self._kick_dir_marker_mocap_id = self._mj_model.body_mocapid[
        self._mj_model.body("kick_dir_marker").id
    ]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
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

    sensor_id = self._mj_model.sensor("ball_global_linvel").id
    sensor_adr = self._mj_model.sensor_adr[sensor_id]
    sensor_dim = self._mj_model.sensor_dim[sensor_id]
    self._ball_linvel_sensor_adr = jp.array(
        list(range(sensor_adr, sensor_adr + sensor_dim))
    )

    self._double_support_phase = jp.array([-jp.pi / 2, jp.pi / 2])

    # Mirror symmetry: swap legs, negate roll/twist joints.
    # Per-leg order: hip_pitch, hip_roll, thigh(twist), calf, ankle_pitch, ankle_roll
    self._mirror_indices = jp.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])
    self._mirror_signs = jp.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # Randomize robot xy and yaw.
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # Sample kick direction in world frame.
    rng, key = jax.random.split(rng)
    kick_angle = jax.random.uniform(key, minval=0.0, maxval=2 * jp.pi)
    kick_dir_world = jp.array([jp.cos(kick_angle), jp.sin(kick_angle)])

    # Randomize ball properties ±25%.
    rng, key1, key2, key3 = jax.random.split(rng, 4)
    ball_mass = self._ball_mass_base * jax.random.uniform(key1, minval=0.75, maxval=1.25)
    ball_radius = self._ball_radius_base * jax.random.uniform(key2, minval=0.75, maxval=1.25)
    ball_friction = self._ball_friction_base * jax.random.uniform(key3, (3,), minval=0.75, maxval=1.25)

    # Ball position relative to robot.
    ball_pos = self._sample_ball_position(rng, yaw) + qpos[0:2]
    qpos = qpos.at[19:21].set(ball_pos)
    qpos = qpos.at[21].set(ball_radius)

    # Randomize joint positions.
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:19].set(
        qpos[7:19] * jax.random.uniform(key, (12,), minval=0.5, maxval=1.5)
    )

    # Randomize initial velocity.
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    ball_mjx_model = self._randomize_ball_model(ball_mass, ball_radius, ball_friction)
    data = mjx_env.init(ball_mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:19])
    torso_pos = data.site_xpos[self._site_id]
    _s = jp.sqrt(jp.array(0.5))
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._kick_dir_marker_mocap_id].set(
            jp.array([torso_pos[0], torso_pos[1], 0.6])
        ),
        mocap_quat=data.mocap_quat.at[self._kick_dir_marker_mocap_id].set(
            jp.array([_s, -kick_dir_world[1] * _s, kick_dir_world[0] * _s, 0.0])
        ),
    )

    # Gait phase.
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    phase = jp.array([0, jp.pi])

    # Push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    # Kick direction resample interval (3–8 s).
    rng, resample_rng = jax.random.split(rng)
    kick_resample_interval_steps = jp.round(
        jax.random.uniform(resample_rng, minval=3.0, maxval=8.0) / self.dt
    ).astype(jp.int32)

    # Ball push interval (3–8 s).
    rng, ball_push_rng = jax.random.split(rng)
    ball_push_interval_steps = jp.round(
        jax.random.uniform(ball_push_rng, minval=3.0, maxval=8.0) / self.dt
    ).astype(jp.int32)

    # IMU and action delays.
    rng, imu_rng = jax.random.split(rng)
    max_imu_delay = 1
    imu_delay = jax.random.randint(
        imu_rng, minval=0, maxval=max_imu_delay, shape=()
    )
    rng, act_rng = jax.random.split(rng)
    action_delay = jax.random.randint(
        act_rng, minval=1, maxval=_MAX_ACTION_DELAY, shape=()
    )

    half_period_steps = jp.round(jp.pi / jp.squeeze(phase_dt)).astype(jp.int32)

    info = {
        "rng": rng,
        "step": 0,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        "kick_dir_world": kick_dir_world,
        "kick_resample_step": jp.zeros((), dtype=jp.int32),
        "kick_resample_interval_steps": kick_resample_interval_steps,
        "ball_mass": ball_mass,
        "ball_radius": ball_radius,
        "ball_friction": ball_friction,
        "ball_push_step": jp.zeros((), dtype=jp.int32),
        "ball_push_interval_steps": ball_push_interval_steps,
        "initial_ball_pos": jp.array(qpos[19:22]),
        "reached_ball": jp.zeros((), dtype=bool),
        "steps_since_reached_ball": jp.zeros((), dtype=jp.int32),
        "phase_dt": phase_dt,
        "phase": phase,
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
        "imu_delay": imu_delay,
        "imu_buffer": jp.broadcast_to(jp.eye(3), (max_imu_delay, 3, 3)),
        "gyro_buffer": jp.zeros((max_imu_delay, 3)),
        "action_delay": action_delay,
        "action_buffer": jp.zeros((_MAX_ACTION_DELAY, self.mjx_model.nu)),
        "sym_buffer": jp.zeros((_MAX_SYM_DELAY, self.mjx_model.nu)),
        "half_period_steps": half_period_steps,
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

    # Push.
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

    # Ball push.
    state.info["rng"], bp_rng1, bp_rng2 = jax.random.split(state.info["rng"], 3)
    ball_push_theta = jax.random.uniform(bp_rng1, maxval=2 * jp.pi)
    ball_push_speed = jax.random.uniform(bp_rng2, minval=0.1, maxval=2.5)
    ball_push = (
        jp.array([jp.cos(ball_push_theta), jp.sin(ball_push_theta), 0.0])
        * ball_push_speed
        * (jp.mod(state.info["ball_push_step"] + 1,
                  state.info["ball_push_interval_steps"]) == 0)
    )
    qvel = qvel.at[18:21].set(qvel[18:21] + ball_push)

    data = state.data.replace(qvel=qvel)
    state = state.replace(data=data)

    motor_targets = self._default_pose + delayed_action * self._config.action_scale
    ball_mjx_model = self._randomize_ball_model(
        state.info["ball_mass"], state.info["ball_radius"], state.info["ball_friction"]
    )
    data = mjx_env.step(ball_mjx_model, state.data, motor_targets, self.n_substeps)
    state.info["motor_targets"] = motor_targets

    # Resample kick direction at random intervals.
    should_resample = (
        jp.mod(state.info["kick_resample_step"] + 1,
               state.info["kick_resample_interval_steps"]) == 0
    )
    state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    new_angle = jax.random.uniform(key1, minval=0.0, maxval=2 * jp.pi)
    new_dir = jp.array([jp.cos(new_angle), jp.sin(new_angle)])
    new_interval = jp.round(
        jax.random.uniform(key2, minval=3.0, maxval=8.0) / self.dt
    ).astype(jp.int32)
    state.info["kick_dir_world"] = jp.where(
        should_resample, new_dir, state.info["kick_dir_world"]
    )
    state.info["kick_resample_interval_steps"] = jp.where(
        should_resample, new_interval, state.info["kick_resample_interval_steps"]
    )
    state.info["kick_resample_step"] = jp.where(
        should_resample, 0, state.info["kick_resample_step"] + 1
    )

    # Update arrow marker: attached to robot, pointing in kick direction.
    torso_pos = data.site_xpos[self._site_id]
    kd = state.info["kick_dir_world"]
    _s = jp.sqrt(jp.array(0.5))
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._kick_dir_marker_mocap_id].set(
            jp.array([torso_pos[0], torso_pos[1], 0.6])
        ),
        mocap_quat=data.mocap_quat.at[self._kick_dir_marker_mocap_id].set(
            jp.array([_s, -kd[1] * _s, kd[0] * _s, 0.0])
        ),
    )

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_f[..., -1])

    # Update IMU buffer.
    state.info["imu_buffer"] = jp.concatenate(
        [state.info["imu_buffer"][1:], data.site_xmat[self._site_id][None, ...]],
        axis=0,
    )
    state.info["gyro_buffer"] = jp.concatenate(
        [state.info["gyro_buffer"][1:], self.get_gyro(data)[None]], axis=0
    )

    # Update symmetry buffer.
    state.info["sym_buffer"] = jp.concatenate(
        [state.info["sym_buffer"][1:], action[None, ...]], axis=0
    )

    # Update reached_ball (sticky: once reached stays reached).
    state.info["reached_ball"] = jp.logical_or(
        state.info["reached_ball"],
        self._reached_ball(self._get_ball_pos_local(data)),
    )
    state.info["steps_since_reached_ball"] = (
        state.info["steps_since_reached_ball"] + 1
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
    state.info["ball_push_step"] = jp.where(
        jp.mod(state.info["ball_push_step"] + 1,
               state.info["ball_push_interval_steps"]) == 0,
        0, state.info["ball_push_step"] + 1,
    )

    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
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

  def _randomize_ball_model(
      self, mass: jax.Array, radius: jax.Array, friction: jax.Array
  ) -> mjx.Model:
    inertia = jp.full(3, 0.4 * mass * jp.square(radius))
    return self.mjx_model.replace(
        body_mass=self.mjx_model.body_mass.at[self._ball_body_id].set(mass),
        body_inertia=self.mjx_model.body_inertia.at[self._ball_body_id].set(inertia),
        geom_size=self.mjx_model.geom_size.at[self._ball_geom_id, 0].set(radius),
        geom_friction=self.mjx_model.geom_friction.at[self._ball_geom_id].set(friction),
    )

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_gravity(data)[-1] < 0.0
    return (
        fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    )

  def _get_ball_pos_local(self, data: mjx.Data) -> jax.Array:
    """Ball position in robot's local XY frame (via IMU site)."""
    ball_pos = data.site_xpos[self._ball_site_id]
    torso_pos = data.site_xpos[self._site_id]
    torso_mat = data.site_xmat[self._site_id]
    ball_local = jp.dot(torso_mat.T, ball_pos - torso_pos)
    return ball_local[:2]

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    gyro = info["gyro_buffer"][info["imu_delay"]]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + jax.random.normal(noise_rng, shape=gyro.shape)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = info["imu_buffer"][info["imu_delay"]].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + jax.random.normal(noise_rng, shape=gravity.shape)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:19]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:18]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + jax.random.normal(noise_rng, shape=joint_vel.shape)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    ball_pos_local = self._get_ball_pos_local(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_ball_pos = (
        ball_pos_local
        + jax.random.normal(noise_rng, shape=ball_pos_local.shape)
        * self._config.noise_config.level
        * self._config.noise_config.scales.ball_pos
    )

    # Kick direction in robot local frame (command, no noise).
    torso_mat = data.site_xmat[self._site_id].reshape(3, 3)
    kick_dir_world_3d = jp.concatenate([info["kick_dir_world"], jp.zeros(1)])
    kick_dir_local = (torso_mat.T @ kick_dir_world_3d)[:2]

    noisy_last_act = (
        info["last_act"]
        + jax.random.normal(noise_rng, shape=info["last_act"].shape)
        * self._config.noise_config.level
        * self._config.noise_config.scales.last_act
    )

    state = jp.hstack([
        noisy_ball_pos,                                # 2
        noisy_gyro,                                    # 3
        noisy_gravity,                                 # 3
        noisy_joint_angles - self._default_pose,       # 12
        noisy_joint_vel,                               # 12
        noisy_last_act,                                # 12
        phase,                                         # 4
        kick_dir_local,                                # 2
    ])

    accelerometer = self.get_accelerometer(data)
    global_angvel = self.get_global_angvel(data)
    linvel = self.get_local_linvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]
    ball_vel = jp.linalg.norm(data.sensordata[self._ball_linvel_sensor_adr])

    privileged_state = jp.hstack([
        state,
        ball_vel,                                      # 1
        ball_pos_local,                                # 2
        gyro,                                          # 3
        accelerometer,                                 # 3
        gravity,                                       # 3
        linvel,                                        # 3
        global_angvel,                                 # 3
        joint_angles - self._default_pose,             # 12
        joint_vel,                                     # 12
        root_height,                                   # 1
        data.actuator_force,                           # 12
        contact,                                       # 2
        feet_vel,                                      # 2*3
        info["feet_air_time"],                         # 2
        info["reached_ball"],                          # 1
    ])

    return {"state": state, "privileged_state": privileged_state}

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
    ball_pos_local = self._get_ball_pos_local(data)
    return {
        "lin_vel_x": jp.linalg.norm(self.get_local_linvel(data)[:2]),
        "stop_for_kick": self._reward_stop_for_kick(
            info["reached_ball"], self.get_local_linvel(data)
        ),
        "orient_to_ball": self._reward_orient_to_ball(ball_pos_local),
        "ball_proximity": self._reward_ball_proximity(
            ball_pos_local, info["reached_ball"]
        ),
        "ball_height": self._reward_ball_height(
            data.site_xpos[self._ball_site_id]
        ),
        "ball_travel": self._reward_ball_travel(
            data.site_xpos[self._ball_site_id], info["initial_ball_pos"]
        ),
        "ball_speed": self._reward_ball_speed(
            data.sensordata[self._ball_linvel_sensor_adr]
        ),
        "kick_foot_velocity": self._reward_kick_foot_velocity(
            data.sensordata[self._foot_linvel_sensor_adr].ravel(),
            info["reached_ball"],
        ),
        "kick_direction": self._reward_kick_direction(data, info),
        "orient_to_kick_dir": self._reward_orient_to_kick_dir(data, info),
        "wrong_approach": self._cost_wrong_approach(data, info),
        "kick_motion": self._reward_kick_motion(
            info["reached_ball"], info["phase"], data.qpos[7:19]
        ),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_gravity(data)),
        "base_height": self._cost_base_height(data.qpos[2]),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:18], data.actuator_force),
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data, info),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact
        ),
        "feet_phase": self._reward_feet_phase(
            data,
            info["phase"],
            self._config.reward_config.max_foot_height,
        ),
        "feet_level": self._cost_feet_level(data),
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        "stand_still": self._cost_stand_still(jp.zeros(3), data.qpos[7:19]),
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            data.qpos[7:19], jp.zeros(3)
        ),
        "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:19]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:19]),
        "pose": self._cost_pose(data.qpos[7:19]),
        "symmetry": self._cost_action_symmetry(action, info, ball_pos_local),
    }

  # --- Kick rewards ---

  def _reward_orient_to_ball(self, ball_pos_local: jax.Array) -> jax.Array:
    ball_angle = jp.arctan2(ball_pos_local[1], ball_pos_local[0])
    return jp.exp(-jp.abs(ball_angle))

  def _reward_ball_proximity(
      self, ball_pos_local: jax.Array, reached_ball: jax.Array
  ) -> jax.Array:
    dist = jp.linalg.norm(ball_pos_local)
    return jp.where(reached_ball, 1.0, jp.exp(-dist))

  def _reward_ball_height(self, ball_pos: jax.Array) -> jax.Array:
    return ball_pos[2]

  def _reward_ball_travel(
      self, ball_pos: jax.Array, initial_ball_pos: jax.Array
  ) -> jax.Array:
    return jp.linalg.norm(ball_pos - initial_ball_pos)

  def _reward_ball_speed(self, ball_linvel: jax.Array) -> jax.Array:
    return jp.square(jp.linalg.norm(ball_linvel))

  def _reward_kick_foot_velocity(
      self, foot_vels: jax.Array, reached_ball: jax.Array
  ) -> jax.Array:
    del reached_ball
    # Reward forward (x) velocity of left foot (index 3: l_foot is second site,
    # sensor order [l_foot, r_foot] matching FEET_SITES = ["l_foot", "r_foot"]).
    return jp.abs(foot_vels[3])

  def _reward_kick_direction(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    """Ball speed squared in the commanded direction; zero if ball moves the wrong way."""
    ball_vel = data.sensordata[self._ball_linvel_sensor_adr]
    kick_dir = jp.concatenate([info["kick_dir_world"], jp.zeros(1)])
    proj = jp.clip(jp.dot(ball_vel, kick_dir), 0.0, None)
    return jp.square(proj)

  def _reward_orient_to_kick_dir(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    """Reward robot forward axis aligning with kick direction, scaled by proximity."""
    dist = jp.linalg.norm(self._get_ball_pos_local(data))
    weight = jp.clip(1.0 - dist / 1.0, 0.0, 1.0)
    torso_mat = data.site_xmat[self._site_id].reshape(3, 3)
    fwd_xy = torso_mat[:2, 0]
    fwd_xy = fwd_xy / (jp.linalg.norm(fwd_xy) + 1e-6)
    return jp.clip(jp.dot(fwd_xy, info["kick_dir_world"]), 0.0, None) * weight

  def _cost_wrong_approach(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    """Penalty for being outside the 0.4–0.6 m approach ring while not aligned.
    Inside ring (<0.4 m): 1x penalty. Outside ring (>0.6 m): 2x penalty."""
    dist = jp.linalg.norm(self._get_ball_pos_local(data))
    torso_mat = data.site_xmat[self._site_id].reshape(3, 3)
    fwd_xy = torso_mat[:2, 0]
    fwd_xy = fwd_xy / (jp.linalg.norm(fwd_xy) + 1e-6)
    cos_angle = jp.dot(fwd_xy, info["kick_dir_world"])
    cos_threshold = jp.cos(jp.array(40.0 * jp.pi / 180.0))
    not_facing = cos_angle < cos_threshold
    too_close = dist < 0.4
    too_far = dist > 0.6
    return ((too_close | too_far) & not_facing).astype(jp.float32)

  def _reward_stop_for_kick(
      self, reached_ball: jax.Array, lin_vel: jax.Array
  ) -> jax.Array:
    return jp.where(
        reached_ball,
        self._reward_tracking_lin_vel(jp.array([0.0, 0.0, 0.0]), lin_vel),
        0.0,
    )

  def _reward_kick_motion(
      self,
      reached_ball: jax.Array,
      phase: jax.Array,
      qpos: jax.Array,
  ) -> jax.Array:
    """Sinusoidal left hip pitch trajectory when ball is reached."""
    l_hip_pitch = qpos[self._l_hip_pitch_idx]
    x = phase[0] + jp.pi  # map [-pi, pi] -> [0, 2pi]
    target = (
        jp.sin(x * 2) * jp.pi / 4 + self._default_pose[self._l_hip_pitch_idx]
    )
    error = jp.square(l_hip_pitch - target)
    reward = jp.exp(-error / 0.01)
    return jp.where(reached_ball & (x > jp.pi), reward, 0.0)

  def _cost_action_symmetry(
      self,
      action: jax.Array,
      info: dict[str, Any],
      ball_pos_local: jax.Array,
  ) -> jax.Array:
    half_steps = info["half_period_steps"]
    read_idx = _MAX_SYM_DELAY - 1 - half_steps
    delayed_action = info["sym_buffer"][read_idx]
    mirrored = action[self._mirror_indices] * self._mirror_signs
    error = jp.sum(jp.square(mirrored - delayed_action))
    has_data = info["step"] >= half_steps
    far_from_ball = jp.linalg.norm(ball_pos_local) > 0.3
    return error * has_data * far_from_ball

  # --- Tracking ---

  def _reward_tracking_lin_vel(
      self, commands: jax.Array, local_vel: jax.Array
  ) -> jax.Array:
    error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-error / self._config.reward_config.tracking_sigma)

  # --- Base ---

  def _cost_lin_vel_z(self, global_linvel: jax.Array) -> jax.Array:
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel: jax.Array) -> jax.Array:
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    return jp.square(base_height - self._config.reward_config.base_height_target)

  # --- Energy ---

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(torques))

  def _cost_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act
    return jp.sum(jp.square(act - last_act))

  # --- Limits / pose ---

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out)

  def _cost_stand_still(
      self, commands: jax.Array, qpos: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qpos - self._default_pose)) * (
        jp.linalg.norm(commands) < 0.1
    )

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
        jp.abs(qpos[self._knee_indices] - self._default_pose[self._knee_indices])
    )

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos - self._default_pose) * self._weights)

  # --- Feet ---

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
    vel_norm = jp.sqrt(jp.linalg.norm(feet_vel[..., :2], axis=-1))
    foot_z = data.site_xpos[self._feet_site_id][..., -1]
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
      threshold_min: float = 0.2,
      threshold_max: float = 0.5,
  ) -> jax.Array:
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    return jp.sum(air_time)

  def _reward_feet_phase(
      self,
      data: mjx.Data,
      phase: jax.Array,
      foot_height: jax.Array,
  ) -> jax.Array:
    foot_z = data.site_xpos[self._feet_site_id][..., -1]
    rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(jp.clip(rz - foot_z, a_min=0.0)))
    return jp.exp(-error / 0.01)

  def _cost_feet_level(self, data: mjx.Data) -> jax.Array:
    foot_xmat = data.site_xmat[self._feet_site_id].reshape(-1, 3, 3)
    foot_normal_world = foot_xmat[:, :, 2]
    pitch_error = jp.square(foot_normal_world[:, 0])

    body_pos = data.qpos[:3]
    body_xmat = data.xmat[self._torso_body_id].reshape(3, 3)
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_rel_body = (body_xmat.T @ (foot_pos - body_pos).T).T
    foot_forward = foot_rel_body[:, 0]
    weight = jp.exp(-jp.square(foot_forward) / (0.033 ** 2))
    return jp.sum(pitch_error * weight)

  # --- Ball helpers ---

  def _reached_ball(self, ball_pos_local: jax.Array) -> jax.Array:
    dist = jp.linalg.norm(ball_pos_local)
    angle = jp.arctan2(ball_pos_local[1], ball_pos_local[0])
    return jp.logical_and(dist < 0.3, jp.abs(angle) < jp.pi / 4)

  def _sample_ball_position(
      self, rng: jax.Array, yaw: jax.Array
  ) -> jax.Array:
    rng1, rng2 = jax.random.split(rng, 2)
    dist = jax.random.uniform(
        rng1,
        minval=self._config.ball_distance[0],
        maxval=self._config.ball_distance[1],
    )
    angle = jax.random.uniform(
        rng2, minval=yaw[0] - jp.pi / 4, maxval=yaw[0] + jp.pi / 4
    )
    return jp.array([jp.cos(angle) * dist, jp.sin(angle) * dist])
