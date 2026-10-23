"""Deforum's "Guided Images" mode (the looper): a few keyframed images pull the animation towards them.

Port of the looper of the Deforum extension for AUTOMATIC1111 (MIT, commit a7b829e: LoopArgs in args.py,
LooperAnimKeys in animation_key_frames.py, the 'use_looper' branch of generate_inner in generate.py), with the same
setting names. With the same image on the first and on the last keyframe the animation closes on itself: a loop.

Every frame (2D and 3D only):
- the strength is image_strength_schedule (it replaces strength_schedule), seed_behavior becomes "schedule" and
  strength_0_no_init False, as Deforum did;
- the guide image is the one of the last keyframe reached; the warped previous frame is blended towards it by
  blendFactor = blendFactorMax - blendFactorSlope * cos((frame % tweening) / (tweening / 2)) while
  frame % (length of the keyframe interval) <= tweening, else 0.07;
- the result of the model is colour-corrected towards (that blend, blended again towards the guide by
  color_correction_factor). AUTOMATIC1111 did it with its img2img colour correction: histogram match in LAB, then a
  "luminosity" blend that keeps the brightness of the generated image. Here it is rewritten from that description and
  the W3C compositing spec (non-separable luminosity), so it is close to A1111 but not bit-identical.
The first frame starts from the first guide image (img2img with the image strength).
Difference from Deforum: init_images values are an index in the node's 'guided_images' batch or a file path (Deforum
took paths or URLs), and the default is a loop on one image ({"0": 0, "max_f": 0}).
Images are uint8 BGR numpy arrays (H, W, 3), like the rest of the port.
"""
import json
import math

import numpy as np
import torch

from .colors import _lab2rgb, _match, _rgb2lab, _t
from .expr import evaluate
from .keyframes import FrameInterpolater, check_is_number

GUIDED_DEFAULTS = {
    "use_looper": False,
    "init_images": '{\n    "0": 0,\n    "max_f": 0\n}',
    "image_strength_schedule": "0:(0.75)",
    "blendFactorMax": "0:(0.35)",
    "blendFactorSlope": "0:(0.25)",
    "tweening_frames_schedule": "0:(20)",
    "color_correction_factor": "0:(0.075)",
}


def parse_guide_keys(init_images, max_frames):
    """init_images (JSON text or dict) -> [(frame, value), ...] in the order written. Keys are frame numbers or
    expressions with max_f (= max_frames - 1), e.g. "max_f/2"."""
    data = json.loads(init_images) if isinstance(init_images, str) else dict(init_images)
    if not data:
        raise ValueError("init_images: at least one guide image is needed")
    keys = []
    for key, value in data.items():
        key = str(key)
        frame = int(float(key)) if check_is_number(key) else int(evaluate(key, max_f=max_frames - 1))
        keys.append((frame, value))
    frames = [f for f, _ in keys]
    if len(set(frames)) != len(frames):
        raise ValueError(f"init_images: two guide images on the same frame ({frames})")
    return keys


def blend_factor(frame, key_frames, tweening, factor_max, factor_slope):
    """The looper's blend of the previous frame towards the guide image (generate.py, use_looper branch)."""
    skip = 25
    for start, end in zip(key_frames, key_frames[1:]):
        if start <= frame <= end:
            skip = end - start
    if frame % skip <= tweening:
        return factor_max - factor_slope * math.cos((frame % tweening) / (tweening / 2))
    return 0.07


def guide_index(frame, key_frames):
    """Which guide image is on: the number of keyframes after the first one that this frame has reached."""
    return sum(frame >= k for k in key_frames[1:])


def pil_blend(a, b, alpha):
    """PIL Image.blend for uint8 images and 0 <= alpha <= 1: in1 + alpha * (in2 - in1) in float32, truncated."""
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    return (a32 + np.float32(alpha) * (b32 - a32)).astype(np.uint8)


def _set_lum(c, lum):
    """W3C compositing SetLum + ClipColor; c (..., 3) floats 0..1 in RGB, lum (...)."""
    d = lum - (c[..., 0] * 0.3 + c[..., 1] * 0.59 + c[..., 2] * 0.11)
    c = c + d[..., None]
    l = (c[..., 0] * 0.3 + c[..., 1] * 0.59 + c[..., 2] * 0.11)[..., None]
    n = c.min(-1, keepdim=True).values
    x = c.max(-1, keepdim=True).values
    c = torch.where(n < 0, l + (c - l) * l / (l - n).clamp_min(1e-12), c)
    c = torch.where(x > 1, l + (c - l) * (1 - l) / (x - l).clamp_min(1e-12), c)
    return c


def color_correct(image, correction, device="cpu"):
    """A1111-style img2img colour correction: the colours of 'correction', the brightness of 'image' (BGR uint8)."""
    rgb = _t(np.ascontiguousarray(image[..., ::-1]), device).long()
    ref = _t(np.ascontiguousarray(correction[..., ::-1]), device).long()
    matched = _lab2rgb(_match(_rgb2lab(rgb), _rgb2lab(ref))).double() / 255.0
    orig = rgb.double() / 255.0
    lum = orig[..., 0] * 0.3 + orig[..., 1] * 0.59 + orig[..., 2] * 0.11
    out = (_set_lum(matched, lum).clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    return np.ascontiguousarray(out[..., ::-1])


class GuidedImages:
    """The looper's schedules and guide images. load(value) -> uint8 BGR image of the right size (value = what is
    written in init_images: an index in the guided_images batch, or a file)."""

    def __init__(self, settings, seed, load):
        s = settings
        n = int(s["max_frames"])
        self.keys = parse_guide_keys(s["init_images"], n)
        self.key_frames = [f for f, _ in self.keys]
        fi = FrameInterpolater(n, seed)

        def series(name):
            return fi.get_inbetweens(fi.parse_key_frames(s[name]))

        self.image_strength = series("image_strength_schedule")
        self.factor_max = series("blendFactorMax")
        self.factor_slope = series("blendFactorSlope")
        self.tweening = series("tweening_frames_schedule")
        self.correction = series("color_correction_factor")
        self._cache, self._load = {}, load

    def image(self, i):
        value = self.keys[i][1]
        key = json.dumps(value)                       # the same image on several keyframes is loaded once
        if key not in self._cache:
            self._cache[key] = self._load(value)
        return self._cache[key]

    def first(self):
        return self.image(0)

    def guide(self, frame):
        return self.image(min(guide_index(frame, self.key_frames), len(self.keys) - 1))

    def factor(self, frame):
        return blend_factor(frame, self.key_frames, float(self.tweening[frame]), float(self.factor_max[frame]),
                            float(self.factor_slope[frame]))
