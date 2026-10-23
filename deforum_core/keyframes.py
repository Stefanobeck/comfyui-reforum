"""Deforum keyframe schedules: "0: (1.0), 30: (1.02+0.01*sin(t/5))" -> one value per frame.

Port of FrameInterpolater / DeformAnimKeys from the Deforum extension for AUTOMATIC1111 (MIT, commit a7b829e,
scripts/deforum_helpers/animation_key_frames.py), with the same behaviour:
- a number at a keyframe is interpolated linearly up to the next keyframe;
- an expression at a keyframe is evaluated at EVERY frame from there to the next keyframe (t = frame);
- before the first and after the last keyframe the value is held.
numexpr and pandas are replaced by expr.evaluate and numpy. Two harmless extensions: commas inside parentheses
(max(a, b)) no longer break the parsing, and a schedule that does not start at frame 0 no longer crashes.
"""
import re

import numpy as np

from .expr import evaluate

_FLOAT = re.compile(r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$')


def check_is_number(value):
    return _FLOAT.match(str(value)) is not None


def sanitize_value(value):
    return value.replace("'", "").replace('"', "").replace('(', "").replace(')', "")


def _split_top_level(text, sep=","):
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += (ch == "(") - (ch == ")")
        if ch == sep and depth <= 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


class FrameInterpolater:
    def __init__(self, max_frames=0, seed=-1):
        self.max_frames = int(max_frames)
        self.seed = seed

    def _names(self, t=None):
        names = {"max_f": self.max_frames - 1, "s": self.seed}
        if t is not None:
            names["t"] = t
        return names

    def parse_key_frames(self, string):
        frames = {}
        for chunk in _split_top_level(str(string)):
            if not chunk.strip():
                continue
            frame_part, _, value = chunk.partition(":")
            key = frame_part.strip()
            if check_is_number(sanitize_value(key)):
                frame = int(float(sanitize_value(key)))
            else:
                # "max_f-2" -> strip one pair of quotes, then evaluate
                k = key.replace("'", "", 1).replace('"', "", 1)[::-1].replace("'", "", 1).replace('"', "", 1)[::-1]
                frame = int(evaluate(k, **self._names()))
            frames[frame] = value.strip()
        if not frames and len(str(string).strip()) != 0:
            raise ValueError(f"Key frame string not correctly formatted: '{string}'")
        return frames

    def get_inbetweens(self, key_frames, integer=False, is_single_string=False):
        n = self.max_frames
        series = [None if is_single_string else np.nan] * n
        value, value_is_number = None, True
        for i in range(n):
            if i in key_frames:
                value = key_frames[i]
                value_is_number = check_is_number(sanitize_value(value))
                if value_is_number:
                    series[i] = sanitize_value(value) if is_single_string else float(sanitize_value(value))
            if value is None:
                continue
            if not value_is_number:
                series[i] = sanitize_value(value) if is_single_string else float(evaluate(value, **self._names(i)))
            elif is_single_string and i > 0:
                series[i] = series[i - 1]
        if is_single_string:
            last = next((v for v in series if v is not None), None)
            out = []
            for v in series:
                last = v if v is not None else last
                out.append(last)
            return out
        arr = np.array(series, dtype=np.float64)
        valid = np.flatnonzero(~np.isnan(arr))
        if valid.size == 0:
            raise ValueError("a schedule has no value")
        arr[0] = arr[valid[0]]
        arr[n - 1] = arr[valid[-1]]
        valid = np.flatnonzero(~np.isnan(arr))
        arr = np.interp(np.arange(n), valid, arr[valid])
        if integer:
            return arr.astype(int)
        return arr


# name of the schedule in the settings -> name of the series (as in DeformAnimKeys)
SCHEDULES = {
    "angle": "angle_series", "transform_center_x": "transform_center_x_series",
    "transform_center_y": "transform_center_y_series", "zoom": "zoom_series",
    "translation_x": "translation_x_series", "translation_y": "translation_y_series",
    "translation_z": "translation_z_series", "rotation_3d_x": "rotation_3d_x_series",
    "rotation_3d_y": "rotation_3d_y_series", "rotation_3d_z": "rotation_3d_z_series",
    "perspective_flip_theta": "perspective_flip_theta_series", "perspective_flip_phi": "perspective_flip_phi_series",
    "perspective_flip_gamma": "perspective_flip_gamma_series", "perspective_flip_fv": "perspective_flip_fv_series",
    "noise_schedule": "noise_schedule_series", "strength_schedule": "strength_schedule_series",
    "contrast_schedule": "contrast_schedule_series", "cfg_scale_schedule": "cfg_scale_schedule_series",
    "subseed_schedule": "subseed_schedule_series", "subseed_strength_schedule": "subseed_strength_schedule_series",
    "steps_schedule": "steps_schedule_series", "seed_schedule": "seed_schedule_series",
    "clipskip_schedule": "clipskip_schedule_series", "noise_multiplier_schedule": "noise_multiplier_schedule_series",
    "kernel_schedule": "kernel_schedule_series", "sigma_schedule": "sigma_schedule_series",
    "amount_schedule": "amount_schedule_series", "threshold_schedule": "threshold_schedule_series",
    "aspect_ratio_schedule": "aspect_ratio_series", "fov_schedule": "fov_series",
    "near_schedule": "near_series", "far_schedule": "far_series",
}
STRING_SCHEDULES = {"sampler_schedule": "sampler_schedule_series"}


class DeformAnimKeys:
    """All the schedules of an animation, expanded to one value per frame (same attribute names as Deforum)."""

    def __init__(self, anim_args, seed=-1):
        self.fi = FrameInterpolater(anim_args["max_frames"], seed)
        for name, attr in SCHEDULES.items():
            try:
                setattr(self, attr, self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args[name])))
            except Exception as e:
                raise ValueError(f"Deforum schedule '{name}': {e}") from None
        for name, attr in STRING_SCHEDULES.items():
            setattr(self, attr, self.fi.get_inbetweens(self.fi.parse_key_frames(anim_args[name]), is_single_string=True))
