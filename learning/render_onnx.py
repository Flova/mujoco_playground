"""Render rollout videos from a saved ONNX policy."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np
import onnxruntime as rt

from mujoco_playground import registry


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--onnx", required=True, help="Path to policy.onnx")
    p.add_argument("--num_videos", type=int, default=5)
    p.add_argument("--outdir", default=None, help="Output dir (defaults to onnx dir)")
    return p.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    outdir = Path(args.outdir) if args.outdir else onnx_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    def infer(obs_state: np.ndarray) -> np.ndarray:
        out = sess.run([output_name], {input_name: obs_state[None].astype(np.float32)})[0]
        return out[0]

    env_cfg = registry.get_default_config(args.env)
    env = registry.load(args.env, config=env_cfg)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    render_every = 2
    fps = 1.0 / env.dt / render_every

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    for i in range(args.num_videos):
        rng = jax.random.PRNGKey(i)
        state = jit_reset(rng)
        rollout = [state]

        for _ in range(env_cfg.episode_length):
            obs = np.array(state.obs["state"])
            ctrl = infer(obs)
            rng, step_rng = jax.random.split(rng)
            state = jit_step(state, jp.array(ctrl))
            rollout.append(state)
            if state.done:
                break

        traj = rollout[::render_every]
        frames = env.render(traj, camera="track", height=480, width=640,
                            scene_option=scene_option)

        if traj and "max_ball_speed" in traj[0].info:
            from PIL import Image, ImageDraw, ImageFont
            try:
                font = ImageFont.load_default(size=20)
            except TypeError:
                font = ImageFont.load_default()
            annotated = []
            for frame, s in zip(frames, traj):
                speed = float(s.info["max_ball_speed"])
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), f"max kick speed: {speed:.2f} m/s",
                          fill=(0, 0, 0), font=font)
                annotated.append(np.array(img))
            frames = annotated

        out_path = outdir / f"rollout_{i}.mp4"
        media.write_video(str(out_path), frames, fps=fps)
        print(f"Video {i} saved → {out_path}  ({len(frames)} frames @ {fps:.0f} fps)")


if __name__ == "__main__":
    main()
