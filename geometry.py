import torch


def rot6d_to_matrix(r6):
    """Zhou et al. 2019 — continuous 6D rotation to 3x3 matrix.
       r6 shape (..., 6).  Returns (..., 3, 3)."""
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns = basis


def matrix_to_rot6d(R):
    """Take first two columns."""
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)


def euler_zyx_to_matrix(euler):
    """Blender XYZ Euler -> matrix (Rz @ Ry @ Rx)."""
    rx, ry, rz = euler.unbind(-1)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    Rx = torch.stack([
        torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], -1),
        torch.stack([torch.zeros_like(cx), cx, -sx], -1),
        torch.stack([torch.zeros_like(cx), sx,  cx], -1),
    ], -2)
    Ry = torch.stack([
        torch.stack([cy, torch.zeros_like(cy), sy], -1),
        torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], -1),
        torch.stack([-sy, torch.zeros_like(cy), cy], -1),
    ], -2)
    Rz = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)], -1),
        torch.stack([sz,  cz, torch.zeros_like(cz)], -1),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], -1),
    ], -2)
    return Rz @ Ry @ Rx


def lift_uv_depth_to_camera(uv_pixel, depth, K, blender_convention=True):
    """
    uv_pixel: (..., 2)  pixel coords in original image space
    depth:    (..., 1)  positive depth (meters, +z forward)
    K:        (..., 3, 3)  intrinsics
    Returns 3D point in camera frame.  If blender_convention, returns Blender axes (-z forward).
    """
    fx = K[..., 0, 0:1]
    fy = K[..., 1, 1:2]
    cx = K[..., 0, 2:3]
    cy = K[..., 1, 2:3]

    u = uv_pixel[..., 0:1]
    v = uv_pixel[..., 1:2]
    z = depth

    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    pt = torch.cat([X, Y, Z], dim=-1)

    if blender_convention:
        # OpenCV (x right, y down, +z fwd) -> Blender (x right, y up, -z fwd)
        pt = pt * torch.tensor([1.0, -1.0, -1.0], device=pt.device, dtype=pt.dtype)
    return pt


def build_corners_from_geometry(uv_pixel, depth, size, rot6d, K,
                                blender_convention=True):
    """
    Build the 8 corners of an oriented 3D bbox by lifting (uv, depth) -> 3D center
    then applying rotation + size.

    All shapes broadcast over (...,).
    Returns corners (..., 8, 3) in Blender camera frame.
    """
    center = lift_uv_depth_to_camera(uv_pixel, depth, K, blender_convention)  # (...,3)

    w, h, d = size.unbind(-1)
    sx = torch.stack([-w, -w, -w, -w,  w,  w,  w,  w], -1) / 2
    sy = torch.stack([ h,  h, -h, -h,  h,  h, -h, -h], -1) / 2
    sz = torch.stack([-d,  d,  d, -d, -d,  d,  d, -d], -1) / 2
    local = torch.stack([sx, sy, sz], -1)            # (...,8,3)

    R = rot6d_to_matrix(rot6d)                       # (...,3,3)
    rotated = torch.einsum('...ij,...kj->...ki', R, local)
    return rotated + center.unsqueeze(-2)

# ---------------------------------------------------------------------------
# Rotated-rectangle IoU (numpy; Sutherland-Hodgman clip, no shapely dependency)
# ---------------------------------------------------------------------------
# Lives here rather than in evaluate.py so that NMS (predict.py) and matching
# (evaluate.py) score overlap with the same code.
import math as _math

import numpy as _np


def rbox_corners(cx, cy, w, h, angle):
    """(cx, cy, w, h, angle_rad) -> (4, 2) corner array, counter-clockwise."""
    ca, sa = _math.cos(angle), _math.sin(angle)
    hw, hh = w / 2, h / 2
    local = _np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    R = _np.array([[ca, -sa], [sa, ca]])
    return local @ R.T + _np.array([cx, cy])


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(_np.dot(x, _np.roll(y, -1)) - _np.dot(y, _np.roll(x, -1)))


def _ensure_ccw(poly):
    x, y = poly[:, 0], poly[:, 1]
    signed_area2 = _np.dot(x, _np.roll(y, -1)) - _np.dot(y, _np.roll(x, -1))
    return poly if signed_area2 >= 0 else poly[::-1]


def _cross2(u, v):
    """Scalar z-component of the 2D cross product u x v."""
    return u[0] * v[1] - u[1] * v[0]


def _seg_intersect(p1, p2, p3, p4):
    d1, d2 = p2 - p1, p4 - p3
    denom = _cross2(d1, d2)
    if abs(denom) < 1e-12:
        return p2
    t = _cross2(p3 - p1, d2) / denom
    return p1 + t * d1


def _clip_poly(subject, a, b):
    """Keep the part of `subject` on the left of directed edge a->b."""
    out = []
    n = len(subject)
    if n == 0:
        return out
    edge = b - a
    for i in range(n):
        cur, prev = subject[i], subject[i - 1]
        cur_in = _cross2(edge, cur - a) >= 0
        prev_in = _cross2(edge, prev - a) >= 0
        if cur_in:
            if not prev_in:
                out.append(_seg_intersect(prev, cur, a, b))
            out.append(cur)
        elif prev_in:
            out.append(_seg_intersect(prev, cur, a, b))
    return out


def polygon_intersection_area(poly_a, poly_b):
    """poly_b must be convex; clipped against it edge by edge."""
    output = list(poly_a)
    n = len(poly_b)
    for i in range(n):
        output = _clip_poly(_np.array(output), poly_b[i - 1], poly_b[i])
        if not output:
            return 0.0
    return _poly_area(_np.array(output))


def rotated_iou(box1, box2):
    """box = (cx, cy, w, h, angle_rad). Returns scalar IoU in [0, 1]."""
    # cheap reject: two boxes further apart than their circumradii cannot overlap
    r1 = 0.5 * _math.hypot(box1[2], box1[3])
    r2 = 0.5 * _math.hypot(box2[2], box2[3])
    if _math.hypot(box1[0] - box2[0], box1[1] - box2[1]) >= r1 + r2:
        return 0.0
    c1 = rbox_corners(*box1)
    c2 = _ensure_ccw(rbox_corners(*box2))
    inter = polygon_intersection_area(c1, c2)
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return inter / union if union > 1e-9 else 0.0
