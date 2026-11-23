#!/usr/bin/env python3
"""Headless drive test for robot_scene.xml (no window).

    python test_drive.py [robot_scene.xml]

Phases: settle 1 s -> forward 3 s (both wheels +10 rad/s) -> hard stop 1 s -> turn in place 3 s
(left -5, right +5 rad/s). Checks: moves +X with little yaw, turns, all four contacts loaded,
no tipping / sinking / bouncing / NaN. Exit code 0 = all checks passed.
"""
from __future__ import annotations

import os
import sys

import mujoco
import numpy as np

# default: the scene next to this script, so it runs from any folder
SCENE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robot_scene.xml')
FWD, TURN = 10.0, 5.0          # [rad/s] wheel commands


def yaw_roll_pitch(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return yaw, roll, pitch


def main() -> int:
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_motor')
    right = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_motor')
    wl, wr = (m.geom(f'{s}_wheel_collision') for s in ('left', 'right'))
    r = float(wl.size[0])
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    ok = True

    def run(seconds, ul, ur, log):
        d.ctrl[left], d.ctrl[right] = ul, ur
        for _ in range(int(round(seconds / m.opt.timestep))):
            mujoco.mj_step(m, d)
            yaw, roll, pitch = yaw_roll_pitch(d.qpos[3:7])
            log.append((d.time, *d.qpos[0:3], yaw, roll, pitch, d.qvel[2]))
        return np.array(log)

    def contact_loads():
        loads = {}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            other = c.geom2 if c.geom1 == floor else c.geom1 if c.geom2 == floor else None
            if other is None:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            name = m.geom(other).name
            loads[name] = loads.get(name, 0.0) + f6[0]
        return loads

    # ---- settle --------------------------------------------------------------------------
    s = run(1.0, 0, 0, [])
    if not np.all(np.isfinite(d.qpos)):
        print('FAIL: NaN while settling'); return 1
    z0, pitch0 = s[-1, 3], s[-1, 6]
    loads = contact_loads()
    print(f'settled: base z = {z0 * 1000:.2f} mm, pitch {np.degrees(pitch0):+.2f} deg, '
          f'roll {np.degrees(s[-1, 5]):+.2f} deg')
    print('  floor contact normal forces [N]: ' + ', '.join(f'{k} {v:.2f}' for k, v in sorted(loads.items()))
          + f'  (sum {sum(loads.values()):.2f}, weight {m.body_subtreemass[1] * 9.81:.2f})')
    for name in ('left_wheel_collision', 'right_wheel_collision', 'caster_ball_1_collision',
                 'caster_ball_2_collision'):
        if loads.get(name, 0.0) <= 0.05:
            print(f'  FAIL: {name} carries no load'); ok = False

    # ---- forward ------------------------------------------------------------------------
    p0, yaw0 = d.qpos[0:3].copy(), yaw_roll_pitch(d.qpos[3:7])[0]
    f = run(3.0, FWD, FWD, [])
    dx, dy = d.qpos[0] - p0[0], d.qpos[1] - p0[1]
    dyaw = np.degrees(np.unwrap(f[:, 4])[-1] - yaw0)
    v_meas = np.hypot(*np.diff(f[-200:, 1:3], axis=0).sum(0)) / (f[-1, 0] - f[-200, 0])
    print(f'forward 3 s @ {FWD} rad/s: dx = {dx:+.3f} m, dy = {dy:+.3f} m, yaw change {dyaw:+.2f} deg, '
          f'steady speed {v_meas:.3f} m/s (ideal r*w = {r * FWD:.3f})')
    if not (dx > 1.0 and abs(dy) < 0.1 * dx and abs(dyaw) < 5):
        print('  FAIL: not driving straight forward'); ok = False

    st = run(1.0, 0, 0, [])                              # hard stop (worst case for pitching)

    # ---- turn in place ------------------------------------------------------------------
    p1, yaw1 = d.qpos[0:2].copy(), yaw_roll_pitch(d.qpos[3:7])[0]
    t = run(3.0, -TURN, TURN, [])
    yaw_un = np.unwrap(np.r_[yaw1, t[:, 4]])
    rate = (yaw_un[-1] - yaw_un[-300]) / (t[-1, 0] - t[-300, 0])
    L = float(np.linalg.norm(d.geom_xpos[wl.id] - d.geom_xpos[wr.id]))   # wheel centre distance
    drift = np.linalg.norm(d.qpos[0:2] - p1)
    print(f'turn in place 3 s (L {-TURN:+}, R {TURN:+} rad/s): yaw change {np.degrees(yaw_un[-1] - yaw1):+.1f} deg, '
          f'steady yaw rate {rate:+.2f} rad/s (ideal r*(wR-wL)/L = {r * 2 * TURN / L:+.2f}, L = {L:.4f} m), '
          f'centre drift {drift * 1000:.1f} mm')
    if not rate > 1.0:
        print('  FAIL: not turning left (counter-clockwise)'); ok = False

    # ---- stability while driving (settle excluded: it is the planned 2 mm drop from spawn) --
    # Judged on what matters: no tipping (tilt < 3 deg), no sinking or hopping (base height within
    # -1..+2 mm of its settled height), no launch (|vz| < 0.2 m/s), everything finite. Step commands
    # cause short sub-millimetre transients (e.g. a hard stop dips the nose by < 0.5 mm); those are fine.
    allp = np.vstack([f, st, t])
    max_roll = np.degrees(np.max(np.abs(allp[:, 5])))
    max_dpitch = np.degrees(np.max(np.abs(allp[:, 6] - pitch0)))
    zr = (allp[:, 3].min() - z0) * 1000, (allp[:, 3].max() - z0) * 1000
    vz = np.max(np.abs(allp[:, 7]))
    finite = bool(np.all(np.isfinite(allp)))
    print(f'stability while driving: max roll {max_roll:.2f} deg, max pitch change {max_dpitch:.2f} deg, '
          f'base z {zr[0]:+.2f}..{zr[1]:+.2f} mm from settled, max |vz| {vz * 1000:.1f} mm/s, finite: {finite}')
    if max(max_roll, max_dpitch) > 3 or zr[0] < -1.0 or zr[1] > 2.0 or vz > 0.2 or not finite:
        print('  FAIL: tipping / sinking / hopping / instability'); ok = False

    print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
