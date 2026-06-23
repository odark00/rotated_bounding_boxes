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