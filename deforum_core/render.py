"""The Deforum render loop (2D, 3D, Video Input), with cadence.

Port of render_animation in render.py of the Deforum extension for AUTOMATIC1111 (MIT, commit a7b829e), in the same
order: warp of the previous frame -> colour match -> contrast -> anti-blur -> noise -> img2img with 'strength'
(= how much of the image is kept) -> colour match of the result -> next seed. AUTOMATIC1111 is replaced by an
'engine' given by the caller (the ComfyUI node), so this loop runs without ComfyUI.

engine.generate(init, denoise, positive, negative, seed, cfg, steps, sampler, clipskip, noise_multiplier,
                subseed, subseed_strength) -> uint8 BGR image; init None = txt2img
engine.predict_depth(img, frame_idx) -> (depth tensor (H, W), kind) or None
engine.frame_done(frame_idx, image, total); engine.check_interrupt()
Not ported (they need hybrid video, masks or optical flow): hybrid composite/motion, masks, optical-flow cadence and
redo, diffusion redo, checkpoint scheduling, resume, Interpolation mode.
"""
import numpy as np

from .animation import anim_frame_warp
from .colors import maintain_colors
from .guided import GuidedImages, color_correct, pil_blend
from .keyframes import DeformAnimKeys
from .noise import add_noise
from .prompt import blend_prompt_series, build_prompt_series, prepare_prompt
from .seed import SeedState, fixed_seed
from .sharpen import unsharp_mask


def _grey(img):
    """cv2 BGR2GRAY then GRAY2BGR."""
    y = np.rint(img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299)
    return np.repeat(np.clip(y, 0, 255).astype(np.uint8)[..., None], 3, axis=-1)


def render_animation(settings, prompts, engine, positive="", negative="", init_image=None, video_frames=None,
                     color_image=None, device="cpu", keep_frames=True, guide_loader=None):
    """Returns ({frame_idx: uint8 BGR image}, {frame_idx: depth tensor}). Images are (H, W, 3) BGR like Deforum.
    keep_frames=False: the images are only handed to engine.frame_done (saved to disk there); the dict keeps the
    frame numbers with None, so a long animation does not fill the RAM.
    guide_loader(value) -> uint8 BGR image: the guide images of the Guided Images mode (use_looper)."""
    s = settings
    max_frames = s["max_frames"]
    looper = bool(s.get("use_looper")) and s["animation_mode"] in ("2D", "3D")
    if looper:                                        # as Deforum (render.py) in Guided Images mode
        if guide_loader is None:
            raise ValueError("use_looper needs the guide images (guided_images input or files in init_images)")
        s = dict(s, strength_0_no_init=False, seed_behavior="schedule")
    seed_state = SeedState(fixed_seed(s["seed"]), s["seed_behavior"], s["seed_iter_N"])
    keys = DeformAnimKeys(s, seed_state.seed)
    guided = GuidedImages(s, seed_state.seed, guide_loader) if looper else None
    prompt_series = blend_prompt_series(prompts, max_frames, positive, negative, seed_state.seed,
                                        s.get("prompt_blend_frames", 0))
    # Extension of this port: colour anchor per scene. Deforum anchors the colours to the first frame for the whole
    # animation, so a new prompt is tinted with the old palette. With "scene" the anchor is released around every
    # prompt keyframe (the new scene takes its own colours) and re-taken from the frame where the scene has settled.
    prompt_cut_series = build_prompt_series(prompts, max_frames, positive, negative, seed_state.seed)
    free_windows, reanchors = [], []
    if s.get("color_coherence_anchor", "first_frame") == "scene" and s["color_coherence"] not in ("None", "Image"):
        hard = prompt_cut_series
        before = max(int(s.get("prompt_blend_frames", 0)), 4)
        settle = max(before // 2, 4)
        for k in (i for i in range(1, max_frames) if hard[i] != hard[i - 1]):
            free_windows.append((k - before, k + settle))
            reanchors.append(k + settle)

    def colours_free(idx):
        return any(a <= idx <= b for a, b in free_windows)

    # Extension of this port: transition_strength. Around every prompt change the strength goes smoothly down to this
    # value and back (the model redraws more, so the new scene can come in) without writing it in strength_schedule.
    # 0 = off (Deforum).
    cuts = [i for i in range(1, max_frames) if prompt_cut_series[i] != prompt_cut_series[i - 1]]
    t_strength = float(s.get("transition_strength", 0) or 0)
    t_before = max(int(s.get("prompt_blend_frames", 0)), 8)
    t_after = max(t_before // 2, 6)

    def _smooth(x):
        x = min(max(x, 0.0), 1.0)
        return x * x * (3 - 2 * x)

    def with_transition(idx, value):
        if t_strength <= 0 or t_strength >= value:
            return value
        w = 0.0
        for k in cuts:
            if k - t_before <= idx <= k:
                w = max(w, _smooth((idx - (k - t_before)) / t_before))
            elif k < idx <= k + t_after:
                w = max(w, 1.0 - _smooth((idx - k) / t_after))
        return value + (t_strength - value) * w
    using_vid_init = s["animation_mode"] == "Video Input"
    if using_vid_init and video_frames is None:
        raise ValueError("animation_mode 'Video Input' needs the video frames (video_init input)")
    turbo_steps = 1 if using_vid_init else int(s["diffusion_cadence"])
    predict_depths = s["animation_mode"] == "3D" and s["use_depth_warping"]
    use_init = bool(s["use_init"]) and init_image is not None
    strength = float(s["strength"])

    frames, depths = {}, {}
    current = {"idx": 0}

    def predict(img):
        out = engine.predict_depth(img, current["idx"])
        if out is None:
            return None
        depth, kind = out
        current["kind"] = kind
        return depth

    current["kind"] = "near_bright"

    def warp(img, idx, depth=None):
        current["idx"] = idx
        if predict_depths and depth is None:
            depth = predict(img)              # first, so that current["kind"] matches this depth
        return anim_frame_warp(img, s, keys, idx, None, depth=depth, depth_kind=current["kind"], device=device)

    turbo_prev_image, turbo_prev_frame_idx = None, 0
    turbo_next_image, turbo_next_frame_idx = None, 0
    prev_img, color_match_sample, init_sample = None, None, None
    if s["color_coherence"] == "Image":
        if color_image is None:
            raise ValueError("color_coherence 'Image' needs the colour image (color_image input)")
        color_match_sample = color_image

    frame_idx = 0
    while frame_idx < max_frames:
        engine.check_interrupt()
        noise = keys.noise_schedule_series[frame_idx]
        sched_strength = keys.strength_schedule_series[frame_idx]
        scale = keys.cfg_scale_schedule_series[frame_idx]
        contrast = keys.contrast_schedule_series[frame_idx]
        kernel = int(keys.kernel_schedule_series[frame_idx])
        sigma = keys.sigma_schedule_series[frame_idx]
        amount = keys.amount_schedule_series[frame_idx]
        threshold = keys.threshold_schedule_series[frame_idx]
        steps = int(keys.steps_schedule_series[frame_idx]) if s["enable_steps_scheduling"] else s["steps"]
        sampler = keys.sampler_schedule_series[frame_idx] if s["enable_sampler_scheduling"] else s["sampler"]
        clipskip = int(keys.clipskip_schedule_series[frame_idx]) if s["enable_clipskip_scheduling"] else None
        noise_multiplier = (float(keys.noise_multiplier_schedule_series[frame_idx])
                            if s["enable_noise_multiplier_scheduling"] else None)
        depth = None

        # in-between frames (cadence)
        if turbo_steps > 1:
            tween_frame_start_idx = max(0, frame_idx - turbo_steps)
            for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(frame_idx - tween_frame_start_idx)
                advance_prev = turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx
                advance_next = tween_frame_idx > turbo_next_frame_idx
                if predict_depths:
                    current["idx"] = tween_frame_idx
                    depth = predict(turbo_next_image)
                if advance_prev:
                    turbo_prev_image, _ = warp(turbo_prev_image, tween_frame_idx, depth)
                if advance_next:
                    turbo_next_image, _ = warp(turbo_next_image, tween_frame_idx, depth)
                turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx
                if turbo_prev_image is not None and tween < 1.0:
                    img = turbo_prev_image * (1.0 - tween) + turbo_next_image * tween
                else:
                    img = turbo_next_image
                if s["color_force_grayscale"]:
                    img = _grey(np.asarray(img).astype(np.uint8))
                prev_img = img
                out_img = np.clip(np.rint(img), 0, 255).astype(np.uint8)              # cv2.imwrite rounding
                frames[tween_frame_idx] = out_img if keep_frames else None
                if depth is not None and keep_frames:
                    depths[tween_frame_idx] = depth
                engine.frame_done(tween_frame_idx, out_img, max_frames)

        # after the 1st frame, prev_img exists: it becomes the init of this frame
        if prev_img is not None:
            prev_img, depth = warp(prev_img, frame_idx)
            if depth is not None and keep_frames:
                depths[frame_idx] = depth
            if s["color_coherence"] != "None" and not colours_free(frame_idx):
                if color_match_sample is None:
                    color_match_sample = prev_img.copy()
                else:
                    prev_img = maintain_colors(prev_img, color_match_sample, s["color_coherence"], device)
            if s["color_force_grayscale"]:
                prev_img = _grey(prev_img)
            contrast_image = np.clip((prev_img * contrast).round(), 0, 255).astype(np.uint8)
            if amount > 0:
                contrast_image = unsharp_mask(contrast_image, (kernel, kernel), sigma, amount, threshold, device=device)
            noised_image = add_noise(contrast_image, noise, seed_state.seed, s["noise_type"],
                                     (s["perlin_w"], s["perlin_h"], s["perlin_octaves"], s["perlin_persistence"]),
                                     device=device)
            use_init = True
            init_sample = noised_image
            strength = max(0.0, min(1.0, with_transition(frame_idx, sched_strength)))
        if looper:                                    # the image strength replaces strength_schedule
            strength = max(0.0, min(1.0, with_transition(frame_idx, float(guided.image_strength[frame_idx]))))

        prompt = prompt_series[frame_idx]
        if s["seed_behavior"] == "schedule":
            seed_state.seed = int(keys.seed_schedule_series[frame_idx])
        subseed, subseed_strength = -1, 0.0
        if s["enable_subseed_scheduling"]:
            subseed = int(keys.subseed_schedule_series[frame_idx])
            subseed_strength = float(keys.subseed_strength_schedule_series[frame_idx])
        positive_prompt, negative_prompt = prepare_prompt(prompt, max_frames, frame_idx)

        init = None
        if using_vid_init:
            init = video_frames(frame_idx)
            use_init = True
            strength = max(0.0, min(1.0, with_transition(frame_idx, sched_strength)))

        # generate_inner (generate.py): which image goes in, and with which denoise
        if not use_init and strength > 0 and s["strength_0_no_init"]:
            strength = 0.0
        correction = None
        if init_sample is not None and not using_vid_init:
            init = init_sample
            if looper:                                # pulled towards the guide image, colours corrected towards it
                guide = guided.guide(frame_idx)
                init = pil_blend(init, guide, guided.factor(frame_idx))
                correction = pil_blend(init, guide, float(guided.correction[frame_idx]))
        elif looper:
            init = guided.first()
        elif init is None and use_init and init_image is not None:
            init = init_image
        image = engine.generate(init, 1.0 - strength, positive_prompt, negative_prompt, seed_state.seed, scale,
                                steps, sampler, clipskip, noise_multiplier if init is not None else None,
                                subseed, subseed_strength)
        if correction is not None:
            image = color_correct(image, correction, device)

        reanchor = bool(reanchors) and frame_idx >= reanchors[0]
        if reanchor:                                          # the new scene has settled: its colours are the anchor
            while reanchors and frame_idx >= reanchors[0]:
                reanchors.pop(0)
            color_match_sample = image.copy()
        elif colours_free(frame_idx):
            pass
        elif frame_idx == 0 and s["color_coherence"] == "Image":
            image = maintain_colors(image, color_match_sample, s["color_coherence"], device)
        elif color_match_sample is not None and s["color_coherence"] != "None" and not s["legacy_colormatch"]:
            image = maintain_colors(image, color_match_sample, s["color_coherence"], device)
        if s["color_force_grayscale"]:
            image = _grey(image)
        if (((not s["legacy_colormatch"] and not use_init) or (s["legacy_colormatch"] and sched_strength == 0))
                and s["color_coherence"] != "Image"):
            color_match_sample = image.copy()
        if not using_vid_init:
            prev_img = image

        if turbo_steps > 1:
            turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            turbo_next_image, turbo_next_frame_idx = image, frame_idx
            frame_idx += turbo_steps
        else:
            frames[frame_idx] = image if keep_frames else None
            engine.frame_done(frame_idx, image, max_frames)
            frame_idx += 1
        seed_state.next()

    # Deforum never saved the last generated frame with cadence (it waited for the next tween): we keep it, so the
    # animation ends on a real frame and a short clip with cadence is never empty.
    if turbo_steps > 1 and turbo_next_image is not None and turbo_next_frame_idx not in frames:
        frames[turbo_next_frame_idx] = turbo_next_image if keep_frames else None
        engine.frame_done(turbo_next_frame_idx, turbo_next_image, max_frames)
        # the frames after it (fewer than one cadence step): only the camera motion, like the cadence tweens
        img = turbo_next_image
        for idx in range(turbo_next_frame_idx + 1, max_frames):
            engine.check_interrupt()
            img, depth = warp(img, idx)
            frames[idx] = img if keep_frames else None
            if depth is not None and keep_frames:
                depths[idx] = depth
            engine.frame_done(idx, img, max_frames)
    return frames, depths
