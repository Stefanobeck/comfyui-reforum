"""Tests of the render loop with a fake engine (no ComfyUI, no model, CPU only).

    python -m pytest tests        (or: python tests/test_render.py)
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deforum_core import settings as ds  # noqa: E402
from deforum_core.render import render_animation  # noqa: E402


class FakeEngine:
    """img2img = the init pulled towards a flat colour that depends on the seed, by 'denoise'."""

    def __init__(self, w, h, depth=False):
        self.w, self.h, self.depth = w, h, depth
        self.calls, self.done, self.depth_calls = [], [], 0

    def generate(self, init, denoise, positive, negative, seed, cfg, steps, sampler, clipskip, noise_multiplier,
                 subseed, subseed_strength):
        self.calls.append(dict(init=init is not None, denoise=denoise, positive=positive, negative=negative,
                               seed=seed, cfg=cfg, steps=steps, noise_multiplier=noise_multiplier))
        colour = np.array([seed * 37 % 256, seed * 91 % 256, seed * 53 % 256], np.float64)
        if init is None:
            y, x = np.mgrid[0:self.h, 0:self.w]
            img = np.stack([x * 255 / self.w, y * 255 / self.h, np.full_like(x, 128.0)], -1)
        else:
            img = init.astype(np.float64) * (1 - denoise) + colour * denoise
        return np.clip(img, 0, 255).astype(np.uint8)

    def predict_depth(self, img, frame_idx):
        if not self.depth:
            return None
        self.depth_calls += 1
        y, x = np.mgrid[0:self.h, 0:self.w]
        depth = np.exp(-((x - self.w / 2) ** 2 + (y - self.h / 2) ** 2) / 800.0)
        return torch.tensor(depth, dtype=torch.float32), "near_bright"

    def frame_done(self, idx, image, total):
        self.done.append(idx)

    def check_interrupt(self):
        pass


def make(**over):
    s = ds.default_settings()
    s.update(dict(W=128, H=96, max_frames=10, seed=5), **over)
    return ds.normalize(s)


PROMPTS = {"0": "a forest", "5": "a city --neg blurry"}


def test_2d_basic():
    s = make()
    e = FakeEngine(128, 96)
    frames, _ = render_animation(s, PROMPTS, e, negative="nsfw")
    assert sorted(frames) == list(range(10))
    assert all(f.shape == (96, 128, 3) and f.dtype == np.uint8 for f in frames.values())
    assert not e.calls[0]["init"] and e.calls[0]["denoise"] == 1.0                 # frame 0 = txt2img
    assert all(c["init"] and abs(c["denoise"] - 0.35) < 1e-9 for c in e.calls[1:])  # strength 0.65 -> denoise 0.35
    assert [c["seed"] for c in e.calls] == list(range(5, 15))                       # seed_behavior iter
    assert e.calls[0]["positive"] == "a forest" and e.calls[0]["negative"] == "nsfw"
    assert e.calls[6]["positive"] == "a city" and e.calls[6]["negative"] == "blurry  nsfw"
    assert e.calls[0]["noise_multiplier"] is None and e.calls[1]["noise_multiplier"] == 1.05
    assert e.calls[0]["cfg"] == 7 and e.calls[0]["steps"] == 25


def test_cadence():
    s = make(diffusion_cadence=3)
    e = FakeEngine(128, 96)
    frames, _ = render_animation(s, PROMPTS, e)
    assert len(e.calls) == 4                                   # diffusion at 0, 3, 6, 9
    assert sorted(frames) == list(range(10))                   # the last generated frame is kept (Deforum dropped it)
    assert e.done == list(range(10))
    short = make(diffusion_cadence=2, max_frames=2)            # used to give no frames at all
    frames, _ = render_animation(short, PROMPTS, FakeEngine(128, 96))
    assert sorted(frames) == [0, 1]


def test_3d_first_frame_uses_the_depth_kind_of_its_depth():
    class Metric(FakeEngine):
        def predict_depth(self, img, frame_idx):
            d, _ = super().predict_depth(img, frame_idx)
            self.kinds.append("far_bright")
            return d, "far_bright"
    import deforum_core.render as r
    seen = []
    original = r.anim_frame_warp
    r.anim_frame_warp = lambda *a, **k: (seen.append(k["depth_kind"]), original(*a, **k))[1]
    try:
        e = Metric(128, 96, depth=True)
        e.kinds = []
        render_animation(make(animation_mode="3D", translation_z="0: (10)"), PROMPTS, e)
    finally:
        r.anim_frame_warp = original
    assert seen and all(k == "far_bright" for k in seen)


def test_3d_with_depth():
    s = make(animation_mode="3D", translation_z="0: (10)", rotation_3d_y="0: (1)")
    e = FakeEngine(128, 96, depth=True)
    frames, depths = render_animation(s, PROMPTS, e)
    assert sorted(frames) == list(range(10))
    assert e.depth_calls == 9 and len(depths) == 9


def test_3d_without_depth_model():
    s = make(animation_mode="3D", translation_z="0: (10)")
    frames, depths = render_animation(s, PROMPTS, FakeEngine(128, 96))
    assert len(frames) == 10 and not depths


def test_video_input():
    s = make(animation_mode="Video Input")
    video = [np.full((96, 128, 3), 20 * i, np.uint8) for i in range(10)]
    e = FakeEngine(128, 96)
    render_animation(s, PROMPTS, e, video_frames=lambda i: video[i])
    assert all(c["init"] for c in e.calls) and abs(e.calls[0]["denoise"] - 0.35) < 1e-9


def test_init_image_first_frame():
    s = make(use_init=True)
    e = FakeEngine(128, 96)
    render_animation(s, PROMPTS, e, init_image=np.zeros((96, 128, 3), np.uint8))
    assert e.calls[0]["init"] and abs(e.calls[0]["denoise"] - 0.2) < 1e-9        # Init tab strength 0.8


def test_seed_schedule_and_colour_modes():
    for mode in ("LAB", "HSV", "RGB", "None"):
        s = make(seed_behavior="schedule", color_coherence=mode)
        e = FakeEngine(128, 96)
        render_animation(s, PROMPTS, e)
        assert [c["seed"] for c in e.calls][:2] == [5, -1]   # '0:(s), 1:(-1), ...'


def test_load_settings_file():
    data = {"W": 640, "H": 360, "max_frames": 48, "animation_mode": "3D", "translation_z": "0:(2)",
            "use_init": False, "enable_perspective_flip": "false", "hybrid_motion": "Optical Flow",
            "prompts": {"0": "x", "24": "y"}, "animation_prompts_positive": "cinematic",
            "animation_prompts_negative": "text", "some_ui_thing": 3}
    s, prompts, pos, neg, warnings = ds.load_settings_text(json.dumps(data))
    s = ds.normalize(s)
    assert s["W"] == 640 and s["H"] == 360 and s["animation_mode"] == "3D" and s["enable_perspective_flip"] is False
    assert prompts == {"0": "x", "24": "y"} and pos == "cinematic" and neg == "text"
    assert any("hybrid_motion" in w for w in warnings)


def test_colour_anchor_per_scene():
    class Tinted(FakeEngine):
        """Each scene pulls the image towards its colour: green for the forest, blue for the city (BGR)."""
        def generate(self, init, denoise, positive, negative, seed, *rest):
            self.calls.append(positive)
            target = np.array([200, 40, 30] if "city" in positive else [30, 180, 40], np.float64)
            base = init.astype(np.float64) if init is not None else np.full((self.h, self.w, 3), 90.0)
            return np.clip(base * 0.6 + target * 0.4, 0, 255).astype(np.uint8)

    def last_blue(anchor):
        s = make(max_frames=40, color_coherence="LAB", color_coherence_anchor=anchor, noise_schedule="0: (0)")
        frames, _ = render_animation(s, {"0": "a forest", "15": "a city"}, Tinted(128, 96))
        return frames[39][..., 0].mean() - frames[39][..., 1].mean()          # blue minus green

    assert last_blue("scene") > last_blue("first_frame") + 40, (last_blue("scene"), last_blue("first_frame"))


def test_guided_images_loop():
    from deforum_core.guided import blend_factor, guide_index, parse_guide_keys
    red = np.zeros((96, 128, 3), np.uint8)
    red[..., 2] = 220
    loaded = []

    def load(value):
        loaded.append(value)
        return red

    s = make(use_looper=True, max_frames=12, image_strength_schedule="0:(0.6)", seed_behavior="iter")
    e = FakeEngine(128, 96)
    frames, _ = render_animation(s, {"0": "a forest"}, e, guide_loader=load)
    assert len(frames) == 12 and loaded == [0]                         # one guide image, loaded once
    assert all(c["init"] for c in e.calls)                              # frame 0 starts from the guide image
    assert all(abs(c["denoise"] - 0.4) < 1e-9 for c in e.calls)         # image strength replaces strength_schedule
    seeds = [c["seed"] for c in e.calls]                                # seed_behavior forced to "schedule"
    assert seeds[0] == seeds[-2] == seeds[-1] == 5 and all(x == -1 for x in seeds[1:-2]), seeds   # max_f-1, max_f
    # the looper's formulas (generate.py): keys with max_f, the interval length, the cosine blend
    keys = parse_guide_keys('{"0": "a.png", "max_f/2": "b.png", "max_f": "a.png"}', 21)
    assert [f for f, _ in keys] == [0, 10, 20]
    assert [guide_index(f, [0, 10, 20]) for f in (0, 9, 10, 19, 20)] == [0, 0, 1, 1, 2]
    assert abs(blend_factor(3, [0, 10, 20], 20, 0.35, 0.25) - (0.35 - 0.25 * np.cos(3 / 10))) < 1e-12
    assert blend_factor(30, [0, 10], 4, 0.35, 0.25) == 0.07             # 30 % 25 = 5 > 4 tweening frames


def test_guided_blend_and_colour_correction():
    from PIL import Image
    from deforum_core.guided import color_correct, pil_blend
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    for alpha in (0.0, 0.07, 0.1, 0.35, 0.6, 1.0):
        ref = np.asarray(Image.blend(Image.fromarray(a), Image.fromarray(b), alpha))
        assert np.array_equal(pil_blend(a, b, alpha), ref), alpha
    same = color_correct(a, a)                                          # towards itself: (almost) unchanged
    assert np.abs(same.astype(int) - a).mean() < 3
    blue = np.zeros_like(a)
    blue[..., 0] = 200                                                  # BGR: a blue reference turns it blue
    out = color_correct(a, blue)
    assert out[..., 0].mean() > out[..., 2].mean() + 30


def test_sizes_rounded_to_8():
    s = make(W=130, H=101)
    assert (s["W"], s["H"]) == (128, 96)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
