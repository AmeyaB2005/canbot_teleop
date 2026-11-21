#!/usr/bin/env python3
"""Build robot_scene.xml: the sw2robot MJCF export of assem_mobile, made drivable in MuJoCo.

    python build_scene.py            # writes robot_scene.xml next to this script

Source: mjcf/assem_mobile.xml, the MuJoCo export from sw2robot (read only, never modified),
with only the 16 visual meshes it needs in mjcf/assets/.
It already has a free base, a fused base_link (all fixed parts) and correct inertias, so we
start from it and replace only what is not usable for a wheeled robot:

  * collision: the export collides with the CONVEX HULLS of every mesh (shell, lid, motors,
    the wheels themselves) plus 12.5 mm "toe" spheres (it treated the wheels like feet).
    A wheel's hull is a faceted polygon -> bumpy rolling. All meshes become VISUAL ONLY, and
    collision is done with primitives:
      - one cylinder per wheel, sized from the wheel mesh, centred on the wheel and aligned
        with the wheel's joint axis (computed in the world frame, the local frames are odd),
        friction 1.2;
      - one sphere per caster ball at the ball centre, condim=1 + priority=1 = frictionless.
        (In a contact pair MuJoCo uses the MAX condim/friction of the two geoms unless one has
        a higher priority; without priority=1 the floor's friction would make the casters drag.)
  * actuators: the export has position servos; we use velocity motors "left_motor" /
    "right_motor" (kv=2, ctrl +-20 rad/s, torque limit +-0.5 N m) with the gear sign chosen so
    that POSITIVE ctrl on both wheels drives the robot FORWARD (+X).
  * scene: checker floor, lights, timestep 0.002 s, implicitfast integrator (explicit Euler
    blows up with light wheels + velocity actuators), robot spawned 2 mm above the floor.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'mjcf', 'assem_mobile.xml')        # sw2robot MJCF export (+ mjcf/assets/)
OUT = os.path.join(HERE, 'robot_scene.xml')

WHEELS = {  # side -> (wheel body, wheel joint, visual mesh geom of the wheel)
    'left': ('left_wheel', 'left_wheel_joint', 'left_wheel_visual0'),
    'right': ('right_wheel', 'right_wheel_joint', 'right_wheel_visual0'),
}
CASTER_BALL_GEOMS = ('base_link_visual11', 'base_link_visual12')   # castor_ball_1, castor_ball_1_2
WHEEL_FRICTION = (1.2, 0.005, 0.0001)      # rubber-like tyre: sliding, torsional, rolling
                                           # (casters: frictionless; floor: MuJoCo default 1.0)

# ---- materials (assigned in SolidWorks) ----------------------------------------------------
# Masses and inertias come from the CAD export and are NOT changed here. The materials are used
# to (a) colour the visual meshes and (b) sanity-check the exported masses.
MATERIAL_RGBA = {
    'acrylic':   (0.80, 0.85, 0.90, 0.6),     # base plate, light grey, semi-clear
    'abs_white': (0.95, 0.95, 0.95, 1.0),     # shell, lid, caster housings
    'abs_dark':  (0.15, 0.15, 0.17, 1.0),     # lidar puck
    'tyre':      (0.10, 0.10, 0.10, 1.0),     # drive wheels (ABS, black tyre look)
    'aluminium': (0.70, 0.72, 0.75, 1.0),     # motors (1060) and motor brackets (6061)
    'steel':     (0.80, 0.80, 0.82, 1.0),     # caster balls (plain carbon steel)
}
# mesh-name prefix -> material, checked in order: longer names first, because 'lidar' starts
# with 'lid' and 'motor_clamp' with 'motor'
PART_MATERIAL = (
    ('base_plate', 'acrylic'), ('shell', 'abs_white'), ('lidar', 'abs_dark'), ('lid', 'abs_white'),
    ('castor_housing', 'abs_white'), ('traction_wheel', 'tyre'),
    ('motor_clamp', 'aluminium'), ('motor', 'aluminium'), ('castor_ball', 'steel'),
)
# Expected masses from SolidWorks (kg): every part except the wheels is fixed to base_link.
EXPECTED_MASS = {
    'base_link': 0.502 + 0.335 + 0.285 + 0.123 + 2 * 0.087 + 4 * 0.018 + 2 * 0.010 + 2 * 0.064,
    'left_wheel': 0.180,
    'right_wheel': 0.180,
}
MASS_TOLERANCE = 0.05                      # report a body whose mass differs by more than 5 %
SPAWN_CLEARANCE = 0.002                    # [m] lowest collision point above the floor at spawn
MOTOR_KV = 2.0                             # velocity actuator gain [N m / (rad/s)]
MOTOR_CTRL = 20.0                          # [rad/s] ctrlrange +-
MOTOR_TORQUE = 0.5                         # [N m] forcerange +-: a real motor's torque limit.
# Without it a velocity motor applies kv*(target - actual) = 2*20 = 40 N m on a command jump.
# The tyres can only pass ~mu*N*r = 1.2*6.3 N*0.05 m = 0.38 N m to the floor; the rest comes back
# as reaction torque on the body: measured +67 mm hop and 22 deg nose-dive on a hard stop.
# 0.5 N m (just above the traction limit): <0.5 mm and <0.2 deg, still 0->0.45 m/s in 0.05 s.
TIMESTEP = 0.002                           # [s]


def mesh_vertices_world(m: mujoco.MjModel, d: mujoco.MjData, geom_name: str) -> np.ndarray:
    """(N, 3) world coordinates of the vertices of a mesh geom (after mj_forward)."""
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mid = m.geom_dataid[g]
    adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    return d.geom_xpos[g] + m.mesh_vert[adr:adr + num] @ d.geom_xmat[g].reshape(3, 3).T


def wheel_geometry(m, d, joint: str, geom: str):
    """Centre, unit axis (world), radius and width of a wheel, from its mesh vertices.

    The axis is the joint's WORLD axis (d.xaxis); the centre is on that axis, half-way between
    the wheel's two faces; radius = largest distance of a vertex from the axis.
    """
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
    axis = d.xaxis[j] / np.linalg.norm(d.xaxis[j])
    v = mesh_vertices_world(m, d, geom)
    along = v @ axis
    anchor = d.xanchor[j]
    centre = anchor + axis * (0.5 * (along.min() + along.max()) - anchor @ axis)
    rel = v - centre
    radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
    return centre, axis, float(radial.max()), float(along.max() - along.min())


def main() -> None:
    spec = mujoco.MjSpec.from_file(SRC)
    # Absolute mesh path, so the saved scene works from any folder.
    spec.meshdir = os.path.join(os.path.dirname(SRC), spec.meshdir or '')
    spec.texturedir = spec.meshdir

    # ---- 1. remove the export's collision geometry, servos and floor -------------------
    # Done first: the export's collision meshes are not shipped (only the visual ones are),
    # so they must be gone before anything is compiled.
    for g in list(spec.geoms):
        if g.name == 'ground' or g.name.endswith('_toe') or '_collision' in g.name:
            spec.delete(g)
    used = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
    for mesh in list(spec.meshes):
        if mesh.name not in used:
            spec.delete(mesh)
    for a in list(spec.actuators):
        spec.delete(a)
    for g in spec.geoms:                 # every remaining mesh is visual only, coloured by material
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.contype, g.conaffinity, g.group = 0, 0, 2
            material = next((mat for prefix, mat in PART_MATERIAL if g.meshname.startswith(prefix)), None)
            if material:
                g.rgba = MATERIAL_RGBA[material]

    # ---- 2. measure the robot (visual meshes) with the base at the origin ------------
    base = spec.body('base_link')
    base.pos = [0.0, 0.0, 0.0]
    m0 = spec.compile()
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    wheels = {side: wheel_geometry(m0, d0, jnt, geom) for side, (_, jnt, geom) in WHEELS.items()}
    wheel_body_pose = {}
    for side, (body, _, _) in WHEELS.items():
        b = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, body)
        wheel_body_pose[side] = (d0.xpos[b].copy(), d0.xmat[b].reshape(3, 3).copy())
    casters = []
    for g in CASTER_BALL_GEOMS:
        v = mesh_vertices_world(m0, d0, g)
        c = 0.5 * (v.min(axis=0) + v.max(axis=0))
        casters.append((c, float(np.linalg.norm(v - c, axis=1).max())))

    # ---- 3. primitive collision geometry ----------------------------------------------
    for side, (body_name, _, _) in WHEELS.items():
        centre, axis, radius, width = wheels[side]
        xpos, xmat = wheel_body_pose[side]
        c_local = xmat.T @ (centre - xpos)          # world -> wheel body frame
        a_local = xmat.T @ axis
        body = spec.body(body_name)
        g = body.add_geom()
        g.name = f'{side}_wheel_collision'
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.fromto = np.concatenate([c_local - a_local * width / 2, c_local + a_local * width / 2])
        g.size = [radius, 0, 0]
        g.friction = WHEEL_FRICTION
        g.condim = 3
        g.contype, g.conaffinity, g.group = 1, 1, 3
        g.rgba = [0.15, 0.15, 0.15, 1]
        g.mass = 0                                  # keep the exported wheel inertia
    for i, (c, r) in enumerate(casters, start=1):
        g = base.add_geom()
        g.name = f'caster_ball_{i}_collision'
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.pos = c                                   # base frame == world frame (base at origin)
        g.size = [r, 0, 0]
        g.condim, g.priority = 1, 1                 # frictionless: this geom decides the contact
        g.contype, g.conaffinity, g.group = 1, 1, 3
        g.rgba = [0.6, 0.6, 0.6, 1]
        g.mass = 0

    # ---- 4. velocity motors: positive ctrl = forward (+X) -------------------------------
    # A wheel spinning at rate w about the world axis a, touching the floor at -r*z from its
    # centre, rolls the robot with velocity  v = w * r * (a x z).  (a x z)_x = a_y, so a
    # positive joint velocity drives +X iff a_y > 0. The gear sign makes ctrl>0 mean forward.
    gear_sign = {}
    for side, (_, joint, _) in WHEELS.items():
        axis = wheels[side][1]
        gear_sign[side] = 1.0 if np.cross(axis, [0, 0, 1])[0] > 0 else -1.0
        act = spec.add_actuator()
        act.name = f'{side}_motor'
        act.target = joint
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_velocity(kv=MOTOR_KV)
        act.gear = [gear_sign[side], 0, 0, 0, 0, 0]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [-MOTOR_CTRL, MOTOR_CTRL]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-MOTOR_TORQUE, MOTOR_TORQUE]

    # ---- 5. scene: floor, lights, physics options, spawn pose --------------------------
    tex = spec.add_texture()
    tex.name, tex.type, tex.builtin = 'checker', mujoco.mjtTexture.mjTEXTURE_2D, mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1, tex.rgb2, tex.width, tex.height = [0.2, 0.3, 0.4], [0.1, 0.15, 0.2], 300, 300
    tex.mark, tex.markrgb = mujoco.mjtMark.mjMARK_EDGE, [0.8, 0.8, 0.8]
    mat = spec.add_material()
    mat.name, mat.texrepeat, mat.texuniform, mat.reflectance = 'floor', [10, 10], True, 0.1
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'checker'
    floor = spec.worldbody.add_geom()
    floor.name, floor.type, floor.size, floor.material = 'floor', mujoco.mjtGeom.mjGEOM_PLANE, [10, 10, 0.05], 'floor'
    floor.contype, floor.conaffinity = 1, 1
    light = spec.worldbody.add_light()
    light.name, light.pos, light.dir, light.diffuse = 'key_light', [1, -1, 3], [-0.3, 0.3, -1], [0.6, 0.6, 0.6]

    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    lowest = min(min(c[2] - c_r for c, c_r in casters),
                 min(w[0][2] - w[2] for w in wheels.values()))      # lowest collision point
    spawn_z = SPAWN_CLEARANCE - lowest
    base.pos = [0.0, 0.0, spawn_z]
    for key in spec.keys:            # keep the exported "home" keyframe consistent with the spawn
        q = np.array(key.qpos, dtype=float)
        if q.size >= 3:
            q[2] = spawn_z
            key.qpos = q

    # ---- 6. compile, report, save ------------------------------------------------------
    model = spec.compile()
    # Save with the mesh folder RELATIVE to robot_scene.xml (MuJoCo resolves meshdir relative to
    # the model file), so the scene works wherever the repository is cloned. to_xml() needs the
    # absolute path (it recompiles), so the path is swapped in the XML text afterwards.
    xml = spec.to_xml()
    rel = os.path.relpath(spec.meshdir, os.path.dirname(OUT)).replace(os.sep, '/') + '/'
    xml = xml.replace(f'"{spec.meshdir}"', f'"{rel}"')
    with open(OUT, 'w') as f:
        f.write(xml)

    r = float(np.mean([w[2] for w in wheels.values()]))
    track = float(abs(wheels['left'][0][1] - wheels['right'][0][1]))
    print(f'wrote {OUT}')
    for side, (c, a, rad, wid) in wheels.items():
        print(f'  {side:5s} wheel: centre {np.round(c, 4)}  world axis {np.round(a, 3)}  '
              f'radius {rad:.4f} m  width {wid:.4f} m  gear {gear_sign[side]:+.0f}')
    for i, (c, rad) in enumerate(casters, start=1):
        print(f'  caster {i}: centre {np.round(c, 4)}  radius {rad:.4f} m (frictionless)')
    print(f'wheel radius r = {r:.4f} m')
    print(f'track width  L = {track:.4f} m')
    print(f'total mass     = {model.body_subtreemass[1]:.3f} kg '
          f'(SolidWorks materials: {sum(EXPECTED_MASS.values()):.3f} kg)')
    for body, expected in EXPECTED_MASS.items():      # report only; masses are never changed
        actual = float(model.body(body).mass[0])
        ok = abs(actual - expected) <= MASS_TOLERANCE * expected
        print(f'  {body:11s} {actual:.3f} kg vs {expected:.3f} kg expected  '
              f'{"OK" if ok else "<-- CHECK THIS BODY (mass differs > 5 %)"}')
    print(f'motors         = velocity, kv {MOTOR_KV}, ctrl +-{MOTOR_CTRL} rad/s, torque limit +-{MOTOR_TORQUE} N m')
    print(f'spawn height   = {spawn_z:.4f} m (lowest collision point {SPAWN_CLEARANCE * 1000:.0f} mm above floor)')


if __name__ == '__main__':
    main()
