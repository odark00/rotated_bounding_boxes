"""Pinhole + Brown-Conrady camera model of the overhead camera (OpenCV conventions).

Camera frame: X right, Y down (image axes), Z along the optical axis. The camera looks
(almost) straight down, so the floor is the plane Z = mounting_height and a surface
at height h above the floor lies at Z = mounting_height - h.
"""
import glob
import json
import os

import cv2
import numpy as np


class Camera:
    def __init__(self, cam_dir, scale=1.0):
        ci = json.load(open(os.path.join(cam_dir, "cam_info.json")))["intrinsics"]
        raw = json.load(open(sorted(glob.glob(os.path.join(cam_dir, "raw_*.json")))[0]))
        info = raw["camera_specific_info"]
        self.full_size = (raw["camera_config"]["image_width_px"], raw["camera_config"]["image_height_px"])
        self.scale = scale
        self.size = (round(self.full_size[0] * scale), round(self.full_size[1] * scale))  # (w, h)
        self.fx, self.fy = ci["fx"] * scale, ci["fy"] * scale
        self.cx, self.cy = (ci["cx"] + 0.5) * scale - 0.5, (ci["cy"] + 0.5) * scale - 0.5
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.dist = np.array([ci["k1"], ci["k2"], ci["p1"], ci["p2"], ci["k3"]])
        self.mount_h = info["mounting_height"]

    # ---- 3D <-> pixels -------------------------------------------------------------
    def project(self, pts3d):
        """Camera-frame points (N,3) -> undistorted (pinhole) pixels (N,2)."""
        p = np.asarray(pts3d, np.float64)
        return np.stack([self.fx * p[:, 0] / p[:, 2] + self.cx, self.fy * p[:, 1] / p[:, 2] + self.cy], 1)

    def distort(self, px):
        """Undistorted pixels (N,2) -> raw (distorted) pixels (N,2)."""
        k1, k2, p1, p2, k3 = self.dist
        x = (px[:, 0] - self.cx) / self.fx
        y = (px[:, 1] - self.cy) / self.fy
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([xd * self.fx + self.cx, yd * self.fy + self.cy], 1)

    def undistort_points(self, px):
        """Raw pixels (N,2) -> undistorted pixels (N,2)."""
        px = np.asarray(px, np.float64).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-6)
        return cv2.undistortPoints(px, self.K, self.dist, R=None, P=self.K, criteria=crit).reshape(-1, 2)

    def backproject(self, px_undist, depth):
        """Undistorted pixels (N,2) at camera depth Z -> camera-frame points (N,3)."""
        px = np.asarray(px_undist, np.float64)
        return np.stack([(px[:, 0] - self.cx) / self.fx * depth, (px[:, 1] - self.cy) / self.fy * depth,
                         np.full(len(px), depth)], 1)

    # ---- whole images --------------------------------------------------------------
    def undistort_image(self, img):
        """Raw image at this camera's scale -> pinhole image with the same K."""
        return cv2.undistort(img, self.K, self.dist)

    def intrinsics_json(self, canvas):
        """`camera_intrinsics` block of the custom_3d_model schema for a square canvas."""
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy, "width": canvas, "height": canvas,
                "source_camera_type": "Dachser JAI GODA-5000M-PGE KOWA LM12HC (3,90m)",
                "note": "undistorted; frame placed top-left on a black square canvas"}


def pad_square(img, canvas):
    """Place img top-left on a black canvas x canvas image (pure translation of nothing: K unchanged)."""
    out = np.zeros((canvas, canvas), img.dtype)
    out[:img.shape[0], :img.shape[1]] = img
    return out
