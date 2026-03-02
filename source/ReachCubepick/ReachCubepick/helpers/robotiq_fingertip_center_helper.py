# robotiq_fingertip_center_helpers.py
# ------------------------------------------------------------
# Robust fingertip-center computation for Robotiq 2F-140 in Isaac Sim 5.1 + Isaac Lab.
#
# Your confirmed structure:
#   parent_link_path (world): /World/.../left_inner_finger
#   tiproot_path     (world): /World/.../left_inner_finger/Fingertip_01
#   tiproot is an INSTANCE ROOT -> its geometry lives under an implicit PROTOTYPE: /__Prototype*
#
# Pipeline:
#   (A) Init once (USD):
#       1) Find tiproot's prototype root prim.
#       2) Traverse prototype and pick a Fingertip mesh.
#       3) Compute fingertip "center" in TIPROOT local frame (front-face centroid).
#       4) Compute transform TIPROOT -> PARENT(link) in world namespace and transform the point.
#          => tip_center_parent (constant offset in parent link frame).
#
#   (B) Runtime each step (PhysX):
#       5) Get parent link actor-frame pose from Isaac Lab (not COM).
#       6) tip_pos_w = parent_pos_w + quat_apply(parent_quat_w, tip_center_parent)
#
# Notes:
#   - Instancing: descendants under instanceable prims are shared in implicit prototypes (instance proxies are read-only). [1](https://blog.csdn.net/qq_50990388/article/details/156274566)[2](https://blog.csdn.net/Clam_dw/article/details/145577007)
#   - Relative transforms: use UsdGeom.XformCache / ComputeRelativeTransform. [3](https://blog.csdn.net/Jerris_Gigl/article/details/131124499)[4](https://forum.aousd.org/t/material-overrides-on-instanceable-prims-in-maya/986)
#   - Isaac Lab frames: actor(link) frame vs center-of-mass frame may not coincide. [5](https://www.cnblogs.com/myleaf/p/18843334)
#   - Isaac Lab quats are (w, x, y, z). [6](https://docs.unity3d.com/Packages/com.unity.formats.usd@3.0/api/pxr.GfMatrix4d.html)
# ------------------------------------------------------------

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import omni.usd
from pxr import Gf, Usd, UsdGeom

import torch
from isaaclab.utils.math import quat_apply, quat_mul, quat_inv  # wxyz [6](https://docs.unity3d.com/Packages/com.unity.formats.usd@3.0/api/pxr.GfMatrix4d.html)


# -------------------------
# USD helpers (instancing-aware)
# -------------------------

def usd_stage() -> Usd.Stage:
    return omni.usd.get_context().get_stage()


def usd_prim(path: str) -> Usd.Prim:
    p = usd_stage().GetPrimAtPath(path)
    if not p.IsValid():
        raise RuntimeError(f"Prim not found: {path}")
    return p


def usd_find_instance_root(prim: Usd.Prim) -> Optional[Usd.Prim]:
    """Walk up to find the nearest ancestor that is an instance root."""
    p = prim
    while p and p.IsValid():
        if p.IsInstance():
            return p
        p = p.GetParent()
    return None


def usd_tiproot_prototype_root_path(tiproot_path_world: str) -> str:
    """
    For your case: tiproot is inside an instance, and typically tiproot itself is the instance root.
    Return its prototype root path, e.g. '/__Prototype1'. [1](https://blog.csdn.net/qq_50990388/article/details/156274566)[2](https://blog.csdn.net/Clam_dw/article/details/145577007)
    """
    tiproot = usd_prim(tiproot_path_world)
    inst_root = usd_find_instance_root(tiproot)
    if inst_root is None:
        raise RuntimeError(f"Tiproot is not under an instance root (unexpected): {tiproot_path_world}")

    proto = inst_root.GetPrototype()
    if not proto.IsValid():
        raise RuntimeError(f"Prototype not found for instance root: {inst_root.GetPath()}")
    return str(proto.GetPath())


def usd_list_meshes_under(root_path: str) -> List[str]:
    """List Mesh prims under a given root (assumes root is a real prim, e.g., prototype root)."""
    root = usd_prim(root_path)
    meshes: List[str] = []
    for p in Usd.PrimRange(root):
        if UsdGeom.Mesh(p):
            meshes.append(str(p.GetPath()))
    return meshes


# -------------------------
# Matrix -> point (robust, no Transform()/operator* dependency)
# -------------------------

def gf_transform_point_affine(M: Gf.Matrix4d, p3) -> Gf.Vec3d:
    """
    Apply affine Matrix4d to point using ExtractRotationQuat + ExtractTranslation:
      p' = R(q) p + t
    This avoids relying on Matrix4d.Transform()/operator* bindings. [3](https://blog.csdn.net/Jerris_Gigl/article/details/131124499)[4](https://forum.aousd.org/t/material-overrides-on-instanceable-prims-in-maya/986)
    """
    if not isinstance(M, Gf.Matrix4d):
        M = Gf.Matrix4d(M)

    t = M.ExtractTranslation()          # Vec3d [4](https://forum.aousd.org/t/material-overrides-on-instanceable-prims-in-maya/986)
    q = M.ExtractRotationQuat()         # Quatd [4](https://forum.aousd.org/t/material-overrides-on-instanceable-prims-in-maya/986)
    w = float(q.GetReal())
    u = q.GetImaginary()                # Vec3d

    p = Gf.Vec3d(float(p3[0]), float(p3[1]), float(p3[2]))

    # quaternion rotate: p_rot = p + 2*w*(u x p) + 2*(u x (u x p))
    uxp = Gf.Cross(u, p)
    p_rot = p + (2.0 * w) * uxp + 2.0 * Gf.Cross(u, uxp)

    return p_rot + t


# -------------------------
# Compute fingertip center in TIPROOT local frame (from prototype geometry)
# -------------------------

_AXIS_CANDIDATES: List[Tuple[float, float, float]] = [
    ( 1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
    ( 0.0, 1.0, 0.0), ( 0.0,-1.0, 0.0),
    ( 0.0, 0.0, 1.0), ( 0.0, 0.0,-1.0),
]


def _proj(p: Gf.Vec3d, axis: Tuple[float, float, float]) -> float:
    return float(p[0]*axis[0] + p[1]*axis[1] + p[2]*axis[2])

def _slice_center_at_fraction(points: Sequence[Gf.Vec3d],
                              axis: Tuple[float, float, float],
                              fraction: float = 0.5,
                              slice_eps: float = 0.002) -> Tuple[float, float, float]:
    """
    Center of a slice plane located at a given fraction along the axis.
    - fraction=0.0 -> back-most (min projection)
    - fraction=1.0 -> front-most (max projection, i.e.,尖端)
    - fraction=0.5 -> half-way (你要的“一半处”)
    slice_eps in meters (e.g., 0.001~0.003).
    """
    proj = [_proj(p, axis) for p in points]
    s_min = min(proj)
    s_max = max(proj)
    s_target = s_min + (s_max - s_min) * float(fraction)

    slab = [points[i] for i, s in enumerate(proj) if abs(s - s_target) <= slice_eps]

    if not slab:
        # fallback: pick the closest single point to s_target
        i = min(range(len(proj)), key=lambda k: abs(proj[k] - s_target))
        p = points[i]
        return (float(p[0]), float(p[1]), float(p[2]))

    cx = sum(p[0] for p in slab) / len(slab)
    cy = sum(p[1] for p in slab) / len(slab)
    cz = sum(p[2] for p in slab) / len(slab)
    return (float(cx), float(cy), float(cz))


def _front_face_center(points: Sequence[Gf.Vec3d],
                       axis: Tuple[float, float, float],
                       face_eps: float) -> Tuple[float, float, float]:
    proj = [_proj(p, axis) for p in points]
    max_s = max(proj)
    front = [points[i] for i, s in enumerate(proj) if s >= max_s - face_eps]
    if not front:
        i = proj.index(max_s)
        p = points[i]
        return (float(p[0]), float(p[1]), float(p[2]))
    cx = sum(p[0] for p in front) / len(front)
    cy = sum(p[1] for p in front) / len(front)
    cz = sum(p[2] for p in front) / len(front)
    return (float(cx), float(cy), float(cz))


def compute_tip_center_in_tiproot_frame_from_prototype(
    tiproot_path_world: str,
    face_eps: float = 1e-4,
) -> Tuple[Tuple[float, float, float], str, Tuple[float, float, float], float]:
    """
    Returns:
      tip_center_tiproot: (x,y,z) in TIPROOT local frame
      mesh_path_proto: chosen mesh path in prototype namespace
      axis_tiproot: chosen forward axis in tiproot frame
      score_span: span along axis for chosen mesh
    """
    stage = usd_stage()

    proto_root_path = usd_tiproot_prototype_root_path(tiproot_path_world)  # e.g. '/__Prototype1'
    meshes = usd_list_meshes_under(proto_root_path)
    if not meshes:
        raise RuntimeError(f"No meshes found under prototype root: {proto_root_path}")

    cache = UsdGeom.XformCache()  # default time code for authored geometry transforms [3](https://blog.csdn.net/Jerris_Gigl/article/details/131124499)[4](https://forum.aousd.org/t/material-overrides-on-instanceable-prims-in-maya/986)

    proto_root_prim = stage.GetPrimAtPath(proto_root_path)

    best_mesh = None
    best_axis = None
    best_span = -1e30
    best_pts_tiproot: Optional[List[Gf.Vec3d]] = None
    best_proj: Optional[List[float]] = None

    for mesh_path in meshes:
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        mesh = UsdGeom.Mesh(mesh_prim)
        pts = mesh.GetPointsAttr().Get()
        if not pts:
            continue

        # mesh relative to tiproot frame (prototype space) [3](https://blog.csdn.net/Jerris_Gigl/article/details/131124499)
        M_mesh_in_tiproot, _ = cache.ComputeRelativeTransform(mesh_prim, proto_root_prim)

        pts_tiproot = [gf_transform_point_affine(M_mesh_in_tiproot, p) for p in pts]

        for ax in _AXIS_CANDIDATES:
            proj = [_proj(p, ax) for p in pts_tiproot]
            span = max(proj) - min(proj)
            if span > best_span:
                best_span = span
                best_mesh = mesh_path
                best_axis = ax
                best_pts_tiproot = pts_tiproot
                best_proj = proj

    if best_mesh is None or best_axis is None or best_pts_tiproot is None or best_proj is None:
        raise RuntimeError("Failed to select a mesh/axis under prototype.")

    # Use the front-face centroid as "fingertip center"
    tip_center = _slice_center_at_fraction(best_pts_tiproot, best_axis, fraction=0.5, slice_eps=face_eps)

    return tip_center, best_mesh, best_axis, float(best_span)


# -------------------------
# Transform TIPROOT-local point -> PARENT(link)-local point (world namespace)
# -------------------------

def tip_center_in_parent_link_frame(
    parent_link_path_world: str,
    tiproot_path_world: str,
    face_eps: float = 1e-4,
) -> Tuple[Tuple[float, float, float], Dict]:
    """
    Compute fingertip center offset in PARENT(link) frame.

    Steps:
      1) Compute tip_center in TIPROOT local frame from prototype geometry. [1](https://blog.csdn.net/qq_50990388/article/details/156274566)[2](https://blog.csdn.net/Clam_dw/article/details/145577007)
      2) Compute M_tiproot_in_parent via XformCache.ComputeRelativeTransform(tiproot, parent). [3](https://blog.csdn.net/Jerris_Gigl/article/details/131124499)[4](https://forum.aousd.org/t/material-overrides-on-instanceable-prims-in-maya/986)
      3) Apply M to point -> get point expressed in parent frame.
    """
    stage = usd_stage()

    # (1) tip center in tiproot frame
    tip_center_tiproot, mesh_path, axis_tiproot, span = compute_tip_center_in_tiproot_frame_from_prototype(
        tiproot_path_world, face_eps=face_eps
    )

    # (2) transform of tiproot relative to parent in WORLD namespace
    parent_prim = stage.GetPrimAtPath(parent_link_path_world)
    tiproot_prim = stage.GetPrimAtPath(tiproot_path_world)
    if not parent_prim.IsValid():
        raise RuntimeError(f"Parent link prim not found: {parent_link_path_world}")
    if not tiproot_prim.IsValid():
        raise RuntimeError(f"Tiproot prim not found: {tiproot_path_world}")

    cache = UsdGeom.XformCache()
    M_tiproot_in_parent, _ = cache.ComputeRelativeTransform(tiproot_prim, parent_prim)  # [3](https://blog.csdn.net/Jerris_Gigl/article/details/131124499)

    # (3) apply to get point in parent frame
    p_parent = gf_transform_point_affine(M_tiproot_in_parent, tip_center_tiproot)
    tip_center_parent = (float(p_parent[0]), float(p_parent[1]), float(p_parent[2]))

    debug = {
        "parent_link_path_world": parent_link_path_world,
        "tiproot_path_world": tiproot_path_world,
        "proto_root_path": usd_tiproot_prototype_root_path(tiproot_path_world),
        "mesh_path_proto": mesh_path,
        "axis_tiproot": axis_tiproot,
        "span": span,
        "tip_center_tiproot": tip_center_tiproot,
        "tip_center_parent": tip_center_parent,
    }
    return tip_center_parent, debug


# -------------------------
# Isaac Lab runtime helpers (actor/link pose)
# -------------------------

def lab_get_body_id(robot, body_name: str) -> int:
    ids, names = robot.find_bodies(body_name)
    if len(ids) != 1:
        raise RuntimeError(f"Expected 1 match for '{body_name}', got {len(ids)}: {names}")
    return int(ids[0])


def lab_get_body_link_pose_w(robot, body_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (pos_w, quat_w) of ACTOR/LINK frame.
    Isaac Lab distinguishes actor(link) frame and COM frame; they may not coincide. [5](https://www.cnblogs.com/myleaf/p/18843334)
    """
    data = robot.data

    if hasattr(data, "body_link_pose_w"):
        pose = data.body_link_pose_w[:, body_id, :]  # [N,7]
        return pose[:, 0:3], pose[:, 3:7]

    if hasattr(data, "body_link_pos_w") and hasattr(data, "body_link_quat_w"):
        return data.body_link_pos_w[:, body_id, :], data.body_link_quat_w[:, body_id, :]

    # Convert COM -> LINK if available
    if hasattr(data, "body_com_pose_w") and hasattr(data, "body_com_pose_b"):
        com_pose_w = data.body_com_pose_w[:, body_id, :]   # T_w_c
        p_wc = com_pose_w[:, 0:3]
        q_wc = com_pose_w[:, 3:7]

        com_pose_b = data.body_com_pose_b[:, body_id, :]   # T_l_c (COM w.r.t LINK) [5](https://www.cnblogs.com/myleaf/p/18843334)
        p_lc = com_pose_b[:, 0:3]
        q_lc = com_pose_b[:, 3:7]

        q_cl = quat_inv(q_lc)
        p_cl = quat_apply(q_cl, -p_lc)

        p_wl = p_wc + quat_apply(q_wc, p_cl)
        q_wl = quat_mul(q_wc, q_cl)
        return p_wl, q_wl

    # Fallback
    return data.body_pos_w[:, body_id, :], data.body_quat_w[:, body_id, :]


def lab_tip_center_pos_w(robot,
                         parent_body_id: int,
                         tip_center_parent_t: torch.Tensor) -> torch.Tensor:
    """
    Compute fingertip center world position:
      p_tip = p_parent + R_parent * p_tip_parent
    quat convention wxyz in Isaac Lab math. [6](https://docs.unity3d.com/Packages/com.unity.formats.usd@3.0/api/pxr.GfMatrix4d.html)
    """
    p_parent, q_parent = lab_get_body_link_pose_w(robot, parent_body_id)
    N = q_parent.shape[0]
    tip_p = tip_center_parent_t.to(device=q_parent.device, dtype=q_parent.dtype).view(1, 3).expand(N, 3)
    return p_parent + quat_apply(q_parent, tip_p)


# -------------------------
# End-to-end init (returns everything you need)
# -------------------------

def init_fingertip_center(
    env,
    robot_key: str,
    parent_body_name: str,
    parent_link_path_world: str,
    tiproot_path_world: str,
    face_eps: float = 1e-4,
) -> Tuple[int, torch.Tensor, Dict]:
    """
    One-time init for your exact paths:
      - Compute tip_center_parent (constant offset in parent link frame).
      - Convert to torch tensor on robot.device.
      - Find parent_body_id for runtime.
    Returns:
      parent_body_id,
      tip_center_parent_t (3,),
      debug dict
    """
    robot = env.scene.articulations[robot_key]

    parent_body_id = lab_get_body_id(robot, parent_body_name)

    tip_center_parent, debug = tip_center_in_parent_link_frame(
        parent_link_path_world=parent_link_path_world,
        tiproot_path_world=tiproot_path_world,
        face_eps=face_eps,
    )

    tip_center_parent_t = torch.tensor(tip_center_parent, dtype=torch.float32, device=robot.device)

    # extra sanity: tip offset should be cm-scale; print if suspicious
    debug["tip_center_parent_norm_m"] = float(torch.norm(tip_center_parent_t).item())

    return parent_body_id, tip_center_parent_t, debug

class FingertipInitCfg:
    ROBOT_KEY: str = "robot"
    LEFT_PARENT_BODY_NAME: str = "left_inner_finger"
    RIGHT_PARENT_BODY_NAME: str = "right_inner_finger"
    LEFT_PARENT_LINK_PATH_ENV0: str = "/World/envs/env_0/Robot/ee_link/left_inner_finger"
    RIGHT_PARENT_LINK_PATH_ENV0: str = "/World/envs/env_0/Robot/ee_link/right_inner_finger"
    LEFT_FINGERTIP_PATH_ENV0: str = "/World/envs/env_0/Robot/ee_link/left_inner_finger/Fingertip_01"
    RIGHT_FINGERTIP_PATH_ENV0: str = "/World/envs/env_0/Robot/ee_link/right_inner_finger/Fingertip_01"

def write_fingertip_offset_to_env(env):
    env_unwrapperd = env.unwrapped
    left_body_id, left_tip_center_parent_t, _ = init_fingertip_center(
        env=env_unwrapperd,
        robot_key=FingertipInitCfg.ROBOT_KEY,
        parent_body_name=FingertipInitCfg.LEFT_PARENT_BODY_NAME,
        parent_link_path_world=FingertipInitCfg.LEFT_PARENT_LINK_PATH_ENV0,
        tiproot_path_world=FingertipInitCfg.LEFT_FINGERTIP_PATH_ENV0,
        face_eps=1e-4,
    )
    right_body_id, right_tip_center_parent_t, _ = init_fingertip_center(
        env=env_unwrapperd,
        robot_key=FingertipInitCfg.ROBOT_KEY,
        parent_body_name=FingertipInitCfg.RIGHT_PARENT_BODY_NAME,
        parent_link_path_world=FingertipInitCfg.RIGHT_PARENT_LINK_PATH_ENV0,
        tiproot_path_world=FingertipInitCfg.RIGHT_FINGERTIP_PATH_ENV0,
        face_eps=1e-4,
    )
    env_unwrapperd._fingertip_center_info = {
        "left_fingertip_id": left_body_id,
        "left_tip_center_parent_t": left_tip_center_parent_t,
        "right_fingertip_id": right_body_id,
        "right_tip_center_parent_t": right_tip_center_parent_t,
    }

def get_left_right_fingertip_midpoint_pos_w(env) -> torch.Tensor:
    if hasattr(env.unwrapped, "_fingertip_center_info") is False and env.unwrapped.common_step_counter==0:
        # Not initialized; return dummy value. 
        # The shape should be N*3
        return torch.zeros((env.unwrapped.num_envs, 3), dtype=torch.float32, device=env.scene.articulations[FingertipInitCfg.ROBOT_KEY].device)
    info = env.unwrapped._fingertip_center_info
    robot = env.scene.articulations[FingertipInitCfg.ROBOT_KEY]

    left_tip_pos_w = lab_tip_center_pos_w(
        robot,
        parent_body_id=info["left_fingertip_id"],
        tip_center_parent_t=info["left_tip_center_parent_t"],
    )
    right_tip_pos_w = lab_tip_center_pos_w(
        robot,
        parent_body_id=info["right_fingertip_id"],
        tip_center_parent_t=info["right_tip_center_parent_t"],
    )
    return (left_tip_pos_w + right_tip_pos_w) / 2.0

def get_left_right_fingertip_gap(env) -> torch.Tensor:
    if hasattr(env.unwrapped, "_fingertip_center_info") is False and env.unwrapped.common_step_counter==0:
        # Not initialized; return dummy value. 
        # The shape should be N
        return torch.zeros((env.unwrapped.num_envs,1), dtype=torch.float32, device=env.scene.articulations[FingertipInitCfg.ROBOT_KEY].device)
    info = env.unwrapped._fingertip_center_info
    robot = env.scene.articulations[FingertipInitCfg.ROBOT_KEY]

    left_tip_pos_w = lab_tip_center_pos_w(
        robot,
        parent_body_id=info["left_fingertip_id"],
        tip_center_parent_t=info["left_tip_center_parent_t"],
    )
    right_tip_pos_w = lab_tip_center_pos_w(
        robot,
        parent_body_id=info["right_fingertip_id"],
        tip_center_parent_t=info["right_tip_center_parent_t"],
    )
    gap = torch.norm(left_tip_pos_w - right_tip_pos_w, dim=1)
    return gap.unsqueeze(dim=1)

def native_get_left_right_finger_midpoint(env):
    LEFT_INNER_FINGER_NAME = "left_inner_finger"
    RIGHT_INNER_FINGER_NAME = "right_inner_finger"
    ROBOT_NAME = "robot"
    robot = env.scene.articulations[ROBOT_NAME]
    left_finger_pos = robot.data.body_pos_w[:, lab_get_body_id(robot, LEFT_INNER_FINGER_NAME), :]
    right_finger_pos = robot.data.body_pos_w[:, lab_get_body_id(robot, RIGHT_INNER_FINGER_NAME), :]
    return (left_finger_pos + right_finger_pos) / 2.0