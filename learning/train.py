"""Train PPO on a locomotion environment, then export ONNX and render a video."""

import argparse
import functools
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.acme import running_statistics

from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="PiplusJoystickFlatTerrain")
    p.add_argument("--timesteps", type=int, default=None,
                   help="Override num_timesteps from default config")
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--checkpoint", default=None, help="Resume from checkpoint dir")
    p.add_argument("--outdir", default="runs", help="Base output directory")
    p.add_argument("--no_onnx", action="store_true", help="Skip ONNX export")
    p.add_argument("--no_video", action="store_true", help="Skip video render")
    p.add_argument("--no_domain_rand", action="store_true", help="Disable domain randomization")
    return p.parse_args()


def export_onnx(params, ppo_params, env, out_path: Path):
    import tensorflow as tf
    import tf2onnx
    import onnxruntime as rt
    from tensorflow.keras import layers

    obs_size = env.observation_size["state"][0]
    act_size = env.action_size
    hidden_sizes = list(ppo_params.network_factory.policy_hidden_layer_sizes)

    mean = np.array(params[0].mean["state"])
    std = np.array(params[0].std["state"])

    class PolicyNetwork(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.mean = tf.Variable(mean, trainable=False, dtype=tf.float32)
            self.std = tf.Variable(std, trainable=False, dtype=tf.float32)
            self.mlp = tf.keras.Sequential(name="MLP_0")
            for i, size in enumerate(hidden_sizes):
                self.mlp.add(layers.Dense(size, activation=tf.nn.swish,
                                          kernel_initializer="lecun_uniform",
                                          name=f"hidden_{i}"))
            self.mlp.add(layers.Dense(act_size * 2, kernel_initializer="lecun_uniform",
                                      name=f"hidden_{len(hidden_sizes)}"))

        def call(self, inputs):
            x = (inputs - self.mean) / self.std
            logits = self.mlp(x)
            loc, _ = tf.split(logits, 2, axis=-1)
            return tf.tanh(loc)

    model = PolicyNetwork()
    model.output_names = ["continuous_actions"]

    # Build model by running a forward pass.
    model(np.ones((1, obs_size), dtype=np.float32))

    # Transfer weights from JAX params.
    jax_layers = params[1]["params"]
    for i, (name, layer_params) in enumerate(jax_layers.items()):
        tf_layer = model.mlp.get_layer(name=f"hidden_{i}")
        tf_layer.set_weights([np.array(layer_params["kernel"]),
                               np.array(layer_params["bias"])])

    # Export.
    spec = [tf.TensorSpec(shape=(1, obs_size), dtype=tf.float32, name="obs")]
    out_path = str(out_path)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=11, output_path=out_path)

    # Verify with OnnxRuntime.
    sess = rt.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    dummy = np.ones((1, obs_size), dtype=np.float32)
    onnx_out = sess.run(["continuous_actions"], {"obs": dummy})[0]
    print(f"ONNX export OK — output shape {onnx_out.shape} → {out_path}")


def render_video(make_inference_fn, params, env_name, env_cfg, out_path: Path, seed: int = 0):
    eval_env = registry.load(env_name, config=env_cfg)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_infer = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)
    rollout = [state]

    for _ in range(env_cfg.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_infer(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)
        if state.done:
            break

    render_every = 2
    fps = 1.0 / eval_env.dt / render_every
    traj = rollout[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    frames = eval_env.render(
        traj, camera="track", height=480, width=640, scene_option=scene_option
    )
    media.write_video(str(out_path), frames, fps=fps)
    print(f"Video saved → {out_path}  ({len(frames)} frames @ {fps:.0f} fps)")


def main():
    args = parse_args()

    env_cfg = registry.get_default_config(args.env)
    ppo_params = locomotion_params.brax_ppo_config(args.env)

    if args.timesteps is not None:
        ppo_params.num_timesteps = args.timesteps
    if args.num_envs is not None:
        ppo_params.num_envs = args.num_envs

    # Output directory.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir).resolve() / f"{args.env}-{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {outdir}")

    env = registry.load(args.env, config=env_cfg)
    eval_env = registry.load(args.env, config=env_cfg)

    training_params = dict(ppo_params)
    del training_params["network_factory"]

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )

    if not args.no_domain_rand:
        try:
            training_params["randomization_fn"] = registry.get_domain_randomizer(args.env)
        except Exception:
            pass  # env has no randomizer registered

    restore_path = None
    if args.checkpoint:
        ckpts = sorted(Path(args.checkpoint).glob("*/"), key=lambda p: int(p.name))
        restore_path = ckpts[-1] if ckpts else Path(args.checkpoint)
        print(f"Restoring from {restore_path}")

    times = [time.monotonic()]

    def progress(num_steps, metrics):
        times.append(time.monotonic())
        reward = metrics.get("eval/episode_reward", float("nan"))
        std = metrics.get("eval/episode_reward_std", float("nan"))
        elapsed = times[-1] - times[0]
        print(
            f"[{elapsed:7.0f}s] steps={num_steps:>12,}  "
            f"reward={reward:7.3f} ± {std:.3f}"
        )

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=args.seed,
        restore_checkpoint_path=restore_path,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=ppo_params.get("num_eval_envs", 128),
        progress_fn=progress,
    )

    print(f"Training {args.env} for {ppo_params.num_timesteps:,} steps …")
    make_inference_fn, params, metrics = train_fn(
        environment=env, eval_env=eval_env
    )
    print(f"Training done. JIT: {times[1]-times[0]:.1f}s  "
          f"Train: {times[-1]-times[1]:.1f}s")

    if not args.no_onnx:
        try:
            export_onnx(params, ppo_params, registry.load(args.env, config=env_cfg), outdir / "policy.onnx")
        except ImportError as e:
            print(f"ONNX export skipped (missing dep): {e}")

    if not args.no_video:
        for i in range(5):
            render_video(make_inference_fn, params, args.env, env_cfg,
                         outdir / f"rollout_{i}.mp4", seed=i)


if __name__ == "__main__":
    main()
