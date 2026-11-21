#!/usr/bin/env python3
"""Keyboard teleoperation of the differential-drive robot in MuJoCo (no ROS).

    python teleop.py robot_scene.xml

Keys (click the MuJoCo window first so it has keyboard focus):
    Up / Down      increase / decrease forward speed v   (hold for auto-repeat)
    Left / Right   turn left / right (yaw rate w, counter-clockwise positive)
    Space or End   stop (v = w = 0)
With no arrow key pressed for a moment, v and w decay smoothly back to zero.

Differential drive (r = wheel radius, L = track width, both read from the scene file):
    w_left  = (v - w * L / 2) / r          w_right = (v + w * L / 2) / r     [rad/s]
Positive wheel commands drive the robot forward (+X): build_scene.py chose the gear signs.
"""
from __future__ import annotations

import math
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

# ---- command settings -------------------------------------------------------------------
V_STEP, W_STEP = 0.05, 0.3        # change per key press / auto-repeat    [m/s], [rad/s]
V_MAX, W_MAX = 0.5, 3.0           # command limits                          [m/s], [rad/s]
DECAY_AFTER = 0.3                 # [s] without a key press before commands start to decay
DECAY_TAU = 0.8                   # [s] exponential decay time constant
RENDER_HZ = 60.0                  # viewer sync rate (physics runs at the model timestep)

# GLFW key codes (the MuJoCo viewer passes these to key_callback)
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_SPACE, KEY_END = 32, 269


class Command:
    """Forward speed v and yaw rate w, changed by the keyboard thread, read by the sim loop."""

    def __init__(self) -> None:
        self.v = 0.0
        self.w = 0.0
        self.last_key = 0.0
        self.lock = threading.Lock()

    def on_key(self, keycode: int) -> None:
        with self.lock:
            if keycode == KEY_UP:
                self.v += V_STEP
            elif keycode == KEY_DOWN:
                self.v -= V_STEP
            elif keycode == KEY_LEFT:
                self.w += W_STEP
            elif keycode == KEY_RIGHT:
                self.w -= W_STEP
            elif keycode in (KEY_SPACE, KEY_END):
                self.v = self.w = 0.0
            else:
                return
            self.v = float(np.clip(self.v, -V_MAX, V_MAX))
            self.w = float(np.clip(self.w, -W_MAX, W_MAX))
            self.last_key = time.monotonic()

    def update(self, dt: float) -> tuple:
        """Decay towards zero when no key has been pressed recently; return (v, w)."""
        with self.lock:
            if time.monotonic() - self.last_key > DECAY_AFTER:
                k = math.exp(-dt / DECAY_TAU)
                self.v *= k
                self.w *= k
                if abs(self.v) < 1e-3:
                    self.v = 0.0
                if abs(self.w) < 1e-3:
                    self.w = 0.0
            return self.v, self.w


def wheel_params(m: mujoco.MjModel, d: mujoco.MjData) -> tuple:
    """Wheel radius r and track width L from the wheel collision cylinders in the scene."""
    left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'left_wheel_collision')
    right = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'right_wheel_collision')
    mujoco.mj_forward(m, d)
    r = 0.5 * (m.geom_size[left][0] + m.geom_size[right][0])
    L = float(np.linalg.norm(d.geom_xpos[left] - d.geom_xpos[right]))
    return float(r), L


def main() -> None:
    scene = sys.argv[1] if len(sys.argv) > 1 else 'robot_scene.xml'
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    r, L = wheel_params(m, d)
    act_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_motor')
    act_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_motor')
    wheel_max = float(m.actuator_ctrlrange[act_l][1])
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')

    cmd = Command()
    print(f'{scene}: r = {r:.4f} m, L = {L:.4f} m, wheel limit +-{wheel_max:.0f} rad/s, dt = {m.opt.timestep} s')
    print('Click the MuJoCo window, then: Up/Down = speed, Left/Right = turn, Space/End = stop. '
          'Close the window to quit.')

    viewer = mujoco.viewer.launch_passive(m, d, key_callback=cmd.on_key)
    try:
        with viewer.lock():                       # direct state change: hold the viewer lock
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING      # follow the robot
            viewer.cam.trackbodyid = base
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.6, -25.0, 180.0

        sim_start, wall_start = d.time, time.monotonic()
        last_render = 0.0
        while viewer.is_running():
            now = time.monotonic()
            frame_dt = 1.0 / RENDER_HZ
            v, w = cmd.update(frame_dt)

            # Differential drive -> wheel speeds, clamped to the motor range
            w_left = float(np.clip((v - w * L / 2) / r, -wheel_max, wheel_max))
            w_right = float(np.clip((v + w * L / 2) / r, -wheel_max, wheel_max))
            d.ctrl[act_l], d.ctrl[act_r] = w_left, w_right

            # Real time: step the physics until sim time catches up with wall-clock time
            target = sim_start + (now - wall_start)
            steps = 0
            while d.time < target and steps < 200:      # cap: never spiral if rendering is slow
                mujoco.mj_step(m, d)
                steps += 1
            if steps == 200:                            # fell behind: resynchronise the clocks
                sim_start, wall_start = d.time, time.monotonic()

            if now - last_render >= frame_dt:
                # set_texts() and sync() take the viewer lock themselves: calling them while
                # holding viewer.lock() deadlocks (the sim loop froze in testing).
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  'v  [m/s]\nw  [rad/s]\nwheels L/R [rad/s]',
                                  f'{v:+.2f}\n{w:+.2f}\n{w_left:+.1f} / {w_right:+.1f}'))
                viewer.sync()
                last_render = now
            time.sleep(0.001)
    except KeyboardInterrupt:
        print('\nCtrl+C: quitting')
    finally:
        # Close only if the window is still open (closing it with its X button already shut the
        # viewer down), then give the render thread a moment to finish its last frame, so Python
        # does not exit while it is still drawing into a destroyed window (X error GLXBadDrawable).
        if viewer.is_running():
            viewer.close()
        time.sleep(0.3)


if __name__ == '__main__':
    main()
