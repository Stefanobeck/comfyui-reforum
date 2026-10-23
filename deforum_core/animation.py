"""Deforum camera motion: 2D (zoom/angle/translation/perspective flip) and 3D (depth warp).

Port of animation.py of the Deforum extension for AUTOMATIC1111 (MIT, commit a7b829e). The 3D warp is, as in Deforum,
"an adapted and optimized version of transform_image_3d from Disco Diffusion" (MIT, see LICENSE) and uses
py3d_tools.py (an extract of PyTorch3D, BSD, see LICENSE) unchanged.
OpenCV is not part of ComfyUI, so cv2.getRotationMatrix2D / warpPerspective / perspectiveTransform /
getPerspectiveTransform are rewritten with numpy and torch (same formulas, bilinear sampling with replicate or
wrap borders). Images are uint8 BGR (H, W, 3) numpy arrays, as in Deforum.
"""
import math
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F

from . import py3d_tools as p3d


# ---------------------------------------------------------------------------------------------------------------------
# OpenCV replacements
# ---------------------------------------------------------------------------------------------------------------------

def get_rotation_matrix_2d(center, angle, scale):
    """cv2.getRotationMatrix2D: angle in degrees, positive = counter-clockwise."""
    a = math.radians(angle)
    alpha, beta = scale * math.cos(a), scale * math.sin(a)
    cx, cy = center
    return np.array([[alpha, beta, (1 - alpha) * cx - beta * cy],
                     [-beta, alpha, beta * cx + (1 - alpha) * cy]], dtype=np.float64)


def get_perspective_transform(src, dst):
    """cv2.getPerspectiveTransform: the 3x3 homography that maps 4 src points to 4 dst points."""
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    h = np.linalg.solve(np.array(a, dtype=np.float64), np.array(b, dtype=np.float64))
    return np.append(h, 1.0).reshape(3, 3)


def perspective_transform_3d(points, m):
    """cv2.perspectiveTransform for 3D points (N, 3) with a 4x4 matrix."""
    p = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1) @ m.T
    w = p[:, 3:4]
    w = np.where(np.abs(w) > np.finfo(np.float64).eps, w, np.inf)
    return p[:, :3] / w


def warp_perspective(img, m, border="replicate", device="cpu"):
    """cv2.warpPerspective(img, m, (W, H), INTER_LINEAR, BORDER_REPLICATE or BORDER_WRAP) on a uint8 image."""
    h, w = img.shape[:2]
    minv = np.linalg.inv(m)
    t = torch.from_numpy(img).to(device=device, dtype=torch.float32)                     # (H, W, C)
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float64),
                            torch.arange(w, device=device, dtype=torch.float64), indexing="ij")
    mi = torch.from_numpy(minv).to(device)
    den = mi[2, 0] * xs + mi[2, 1] * ys + mi[2, 2]
    sx = (mi[0, 0] * xs + mi[0, 1] * ys + mi[0, 2]) / den
    sy = (mi[1, 0] * xs + mi[1, 1] * ys + mi[1, 2]) / den
    x0, y0 = torch.floor(sx), torch.floor(sy)
    fx, fy = (sx - x0).float()[..., None], (sy - y0).float()[..., None]
    x0, y0 = x0.long(), y0.long()
    x1, y1 = x0 + 1, y0 + 1
    if border == "wrap":
        fix_x, fix_y = (lambda v: v % w), (lambda v: v % h)
    else:
        fix_x, fix_y = (lambda v: v.clamp(0, w - 1)), (lambda v: v.clamp(0, h - 1))
    x0, x1, y0, y1 = fix_x(x0), fix_x(x1), fix_y(y0), fix_y(y1)
    top = t[y0, x0] * (1 - fx) + t[y0, x1] * fx
    bottom = t[y1, x0] * (1 - fx) + t[y1, x1] * fx
    out = top * (1 - fy) + bottom * fy
    return torch.round(out).clamp(0, 255).to(torch.uint8).cpu().numpy()


# ---------------------------------------------------------------------------------------------------------------------
# Perspective flip (Deforum)
# ---------------------------------------------------------------------------------------------------------------------

def get_rotation_matrix_manual(rotation_angles):
    phi, gamma, theta = [np.deg2rad(x) for x in rotation_angles]
    r_phi = np.eye(4)
    r_phi[1, 1] = r_phi[2, 2] = np.cos(phi)
    r_phi[1, 2], r_phi[2, 1] = -np.sin(phi), np.sin(phi)
    r_gamma = np.eye(4)
    r_gamma[0, 0] = r_gamma[2, 2] = np.cos(gamma)
    r_gamma[0, 2], r_gamma[2, 0] = np.sin(gamma), -np.sin(gamma)
    r_theta = np.eye(4)
    r_theta[0, 0] = r_theta[1, 1] = np.cos(theta)
    r_theta[0, 1], r_theta[1, 0] = -np.sin(theta), np.sin(theta)
    return reduce(np.matmul, [r_phi, r_gamma, r_theta])


def warp_matrix(w, h, theta, phi, gamma, scale, fv):
    fv_half = np.deg2rad(fv / 2.)
    d = np.sqrt(w * w + h * h)
    side_length = scale * d / np.cos(fv_half)
    hh = d / (2.0 * np.sin(fv_half))
    n = hh - (d / 2.0)
    f = hh + (d / 2.0)
    t = np.eye(4)
    t[2, 3] = -hh
    r = get_rotation_matrix_manual([phi, gamma, theta])
    p = np.eye(4)
    p[0, 0] = p[1, 1] = 1.0 / np.tan(fv_half)
    p[2, 2] = -(f + n) / (f - n)
    p[2, 3] = -(2.0 * f * n) / (f - n)
    p[3, 2] = -1.0
    m = reduce(np.matmul, [p, t, r])
    pts_in = np.array([[-w / 2., h / 2., 0.], [w / 2., h / 2., 0.], [w / 2., -h / 2., 0.], [-w / 2., -h / 2., 0.]])
    pts_out = perspective_transform_3d(pts_in, m)
    pin = (pts_in[:, :2] + [w / 2., h / 2.]).astype(np.float32)
    pout = ((pts_out[:, :2] + [1., 1.]) * (0.5 * side_length)).astype(np.float32)
    return get_perspective_transform(pin, pout), side_length


def get_flip_perspective_matrix(w, h, keys, frame_idx):
    m, sl = warp_matrix(w, h, keys.perspective_flip_theta_series[frame_idx], keys.perspective_flip_phi_series[frame_idx],
                        keys.perspective_flip_gamma_series[frame_idx], 1., keys.perspective_flip_fv_series[frame_idx])
    post = np.vstack([np.float32([[1, 0, (w - sl) / 2], [0, 1, (h - sl) / 2]]), [0, 0, 1]])
    return np.matmul(m, post)


# ---------------------------------------------------------------------------------------------------------------------
# Frame warp
# ---------------------------------------------------------------------------------------------------------------------

def anim_frame_warp_2d(prev_img, settings, keys, frame_idx, device="cpu"):
    w, h = settings["W"], settings["H"]
    center = (w * keys.transform_center_x_series[frame_idx], h * keys.transform_center_y_series[frame_idx])
    rot_mat = np.vstack([get_rotation_matrix_2d(center, keys.angle_series[frame_idx], keys.zoom_series[frame_idx]),
                         [0, 0, 1]])
    trans_mat = np.vstack([np.float32([[1, 0, keys.translation_x_series[frame_idx]],
                                       [0, 1, keys.translation_y_series[frame_idx]]]), [0, 0, 1]])
    if settings["enable_perspective_flip"]:
        # Deforum wrote np.matmul(bM, rot_mat, trans_mat): the third argument of np.matmul is 'out', so the
        # translation is NOT applied when the perspective flip is on. Kept as it was, it is part of the look.
        m = np.matmul(get_flip_perspective_matrix(w, h, keys, frame_idx), rot_mat)
    else:
        m = np.matmul(rot_mat, trans_mat)
    return warp_perspective(prev_img, m, settings["border"], device)


def prepare_depth_tensor(depth_tensor):
    """Normalisation to [0, 1] + histogram equalisation (Deforum's prepare_depth_tensor / depth_equalization)."""
    depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())
    arr = depth_tensor.cpu().numpy()
    hist, bin_edges = np.histogram(arr, bins=1024, range=(0, 1))
    cdf = hist.cumsum()
    cdf = cdf / float(cdf[-1])
    return torch.from_numpy(np.interp(arr, bin_edges[:-1], cdf)).to(depth_tensor.device)


# How the depth values become z (Deforum's depth_algorithm switch). "near_bright": maps where bright = near
# (MiDaS, Depth Anything renders, any depth IMAGE); "far_bright": distances (Zoe, AdaBins, LeReS, DA3/MoGe metric).
DEPTH_KINDS = {"near_bright": (1, -1, -2), "far_bright": (1, 1, 1)}


def transform_image_3d(prev_img, depth_tensor, rot_mat, translate, settings, keys, frame_idx, depth_kind, device):
    depth, depth_factor, depth_offset = DEPTH_KINDS[depth_kind]
    h, w = prev_img.shape[:2]
    aspect_ratio = float(w) / float(h) if settings["aspect_ratio_use_old_formula"] else keys.aspect_ratio_series[frame_idx]
    near, far, fov_deg = keys.near_series[frame_idx], keys.far_series[frame_idx], keys.fov_series[frame_idx]
    cam_old = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, device=device)
    cam_new = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, R=rot_mat,
                                        T=torch.tensor([translate]), device=device)
    y, x = torch.meshgrid(torch.linspace(-1., 1., h, dtype=torch.float32, device=device),
                          torch.linspace(-1., 1., w, dtype=torch.float32, device=device), indexing="ij")
    invalid = (depth_tensor is None or torch.isnan(depth_tensor).any() or torch.isinf(depth_tensor).any()
               or depth_tensor.min() == depth_tensor.max())
    if invalid:
        z = torch.ones_like(x)
    else:
        depth_final = prepare_depth_tensor(depth_tensor) * depth + depth_offset
        if depth_factor != 1:
            depth_final = depth_final * depth_factor
        z = torch.as_tensor(depth_final, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    old_xy = cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:, 0:2]
    new_xy = cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:, 0:2]
    offset_xy = new_xy - old_xy
    identity = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device).unsqueeze(0)
    coords = F.affine_grid(identity, [1, 1, h, w], align_corners=False)
    coords = coords - torch.reshape(offset_xy, (h, w, 2)).unsqueeze(0)
    image = torch.from_numpy(prev_img.astype(np.float32)).permute(2, 0, 1).to(device)
    new_image = F.grid_sample(image.unsqueeze(0), coords, mode=settings["sampling_mode"],
                              padding_mode=settings["padding_mode"], align_corners=False)
    return new_image.squeeze(0).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(prev_img.dtype)


def anim_frame_warp_3d(prev_img, depth, settings, keys, frame_idx, depth_kind, device):
    scale = 1.0 / 200.0                                                     # matches Disco
    translate = [-keys.translation_x_series[frame_idx] * scale, keys.translation_y_series[frame_idx] * scale,
                 -keys.translation_z_series[frame_idx] * scale]
    rotate = [math.radians(keys.rotation_3d_x_series[frame_idx]), math.radians(keys.rotation_3d_y_series[frame_idx]),
              math.radians(keys.rotation_3d_z_series[frame_idx])]
    if settings["enable_perspective_flip"]:
        h, w = prev_img.shape[:2]
        prev_img = warp_perspective(prev_img, get_flip_perspective_matrix(w, h, keys, frame_idx), settings["border"],
                                    device)
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate, device=device), "XYZ").unsqueeze(0)
    return transform_image_3d(prev_img, depth, rot_mat, translate, settings, keys, frame_idx, depth_kind, device)


def anim_frame_warp(prev_img, settings, keys, frame_idx, predict_depth=None, depth=None, depth_kind="near_bright",
                    device="cpu"):
    """Returns (warped image, depth used). predict_depth(img) -> depth tensor (H, W), called in 3D when no depth
    is given, exactly where Deforum called its depth model."""
    if settings["use_depth_warping"]:
        if depth is None and predict_depth is not None:
            depth = predict_depth(prev_img)
    else:
        depth = None
    if settings["animation_mode"] == "2D":
        return anim_frame_warp_2d(prev_img, settings, keys, frame_idx, device), depth
    return anim_frame_warp_3d(prev_img, depth, settings, keys, frame_idx, depth_kind, device), depth
