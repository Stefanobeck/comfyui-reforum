"""Deforum settings: the defaults of the Deforum extension for AUTOMATIC1111 (MIT, commit a7b829e, args.py), the
reading of its settings files (the .txt/JSON files A1111 Deforum saves next to every animation) and the A1111 sampler
names. The names of the parameters are the original ones, so a settings file can be used as it is."""
import json

from .guided import GUIDED_DEFAULTS

DEFAULT_PROMPTS = {
    "0": "tiny cute bunny, vibrant diffraction, highly detailed, intricate, ultra hd, sharp photo, crepuscular rays, "
         "in focus, by tomasz alen kopera",
    "30": "anthropomorphic clean cat, surrounded by fractals, epic angle and pose, symmetrical, 3d, depth of field, "
          "ruan jia and fenghua zhong",
    "60": "a beautiful coconut --neg photo, realistic",
    "90": "a beautiful durian, trending on Artstation",
}

DEFAULTS = {
    # DeforumArgs
    "W": 512, "H": 512, "seed": -1, "sampler": "Euler a", "steps": 25, "seed_behavior": "iter", "seed_iter_N": 1,
    "use_init": False, "strength": 0.8, "strength_0_no_init": True,
    # DeforumAnimArgs: motion
    "animation_mode": "2D", "max_frames": 120, "border": "replicate",
    "angle": "0: (0)", "zoom": "0: (1.0025+0.002*sin(1.25*3.14*t/30))",
    "translation_x": "0: (0)", "translation_y": "0: (0)", "translation_z": "0: (1.75)",
    "transform_center_x": "0: (0.5)", "transform_center_y": "0: (0.5)",
    "rotation_3d_x": "0: (0)", "rotation_3d_y": "0: (0)", "rotation_3d_z": "0: (0)",
    "enable_perspective_flip": False, "perspective_flip_theta": "0: (0)", "perspective_flip_phi": "0: (0)",
    "perspective_flip_gamma": "0: (0)", "perspective_flip_fv": "0: (53)",
    # generation schedules
    "noise_schedule": "0: (0.065)", "strength_schedule": "0: (0.65)", "contrast_schedule": "0: (1.0)",
    "cfg_scale_schedule": "0: (7)", "enable_steps_scheduling": False, "steps_schedule": "0: (25)",
    "fov_schedule": "0: (70)", "aspect_ratio_schedule": "0: (1)", "aspect_ratio_use_old_formula": False,
    "near_schedule": "0: (200)", "far_schedule": "0: (10000)",
    "seed_schedule": '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)',
    "enable_subseed_scheduling": False, "subseed_schedule": "0: (1)", "subseed_strength_schedule": "0: (0)",
    "enable_sampler_scheduling": False, "sampler_schedule": '0: ("Euler a")',
    "enable_clipskip_scheduling": False, "clipskip_schedule": "0: (2)",
    "enable_noise_multiplier_scheduling": True, "noise_multiplier_schedule": "0: (1.05)",
    # anti-blur
    "amount_schedule": "0: (0.1)", "kernel_schedule": "0: (5)", "sigma_schedule": "0: (1)",
    "threshold_schedule": "0: (0)",
    # coherence
    "color_coherence": "LAB", "color_force_grayscale": False, "legacy_colormatch": False,
    "diffusion_cadence": 1,
    # noise
    "noise_type": "perlin", "perlin_w": 8, "perlin_h": 8, "perlin_octaves": 4, "perlin_persistence": 0.5,
    # 3D depth warping
    "use_depth_warping": True, "depth_algorithm": "Midas-3-Hybrid", "midas_weight": 0.2,
    "padding_mode": "border", "sampling_mode": "bicubic",
    # output
    "fps": 15,
    # extension of this port: crossfade between keyframed prompts (0 = Deforum's hard cut)
    "prompt_blend_frames": 0,
    # extension of this port: colour coherence anchored to the first frame (Deforum) or re-anchored at every scene
    "color_coherence_anchor": "first_frame",
    # extension of this port: strength reached smoothly around every prompt change (0 = off, as in Deforum)
    "transition_strength": 0.0,
    # extension: how two prompts are mixed (morph, A1111 AND): "predictions" = A1111 composable diffusion (two model
    # passes); "embeddings" = the encoded texts averaged (one pass, a morph costs as much as a fixed scene)
    "prompt_mix": "predictions",
    # Guided Images (Deforum's looper, LoopArgs): see guided.py
    **GUIDED_DEFAULTS,
}
PROMPT_MIX = ["embeddings", "predictions"]
COLOR_ANCHORS = ["first_frame", "scene"]

ANIMATION_MODES = ["2D", "3D", "Video Input"]
BORDERS = ["replicate", "wrap"]
COLOR_COHERENCE = ["None", "HSV", "LAB", "RGB", "Image"]
SEED_BEHAVIORS = ["iter", "fixed", "random", "ladder", "alternate", "schedule"]
NOISE_TYPES = ["perlin", "uniform"]
PADDING_MODES = ["border", "reflection", "zeros"]
SAMPLING_MODES = ["bicubic", "bilinear", "nearest"]

# A1111 sampler name -> (ComfyUI sampler, ComfyUI scheduler)
SAMPLERS = {
    "Euler a": ("euler_ancestral", "normal"), "Euler": ("euler", "normal"), "LMS": ("lms", "normal"),
    "Heun": ("heun", "normal"), "DPM2": ("dpm_2", "normal"), "DPM2 a": ("dpm_2_ancestral", "normal"),
    "DPM++ 2S a": ("dpmpp_2s_ancestral", "normal"), "DPM++ 2M": ("dpmpp_2m", "normal"),
    "DPM++ SDE": ("dpmpp_sde", "normal"), "DPM++ 2M SDE": ("dpmpp_2m_sde", "normal"),
    "DPM fast": ("dpm_fast", "normal"), "DPM adaptive": ("dpm_adaptive", "normal"),
    "LMS Karras": ("lms", "karras"), "DPM2 Karras": ("dpm_2", "karras"), "DPM2 a Karras": ("dpm_2_ancestral", "karras"),
    "DPM++ 2S a Karras": ("dpmpp_2s_ancestral", "karras"), "DPM++ 2M Karras": ("dpmpp_2m", "karras"),
    "DPM++ SDE Karras": ("dpmpp_sde", "karras"), "DPM++ 2M SDE Karras": ("dpmpp_2m_sde", "karras"),
    "DPM++ 2M SDE Exponential": ("dpmpp_2m_sde", "exponential"), "DPM++ 3M SDE": ("dpmpp_3m_sde", "normal"),
    "DPM++ 3M SDE Karras": ("dpmpp_3m_sde", "karras"), "DPM++ 3M SDE Exponential": ("dpmpp_3m_sde", "exponential"),
    "DDIM": ("ddim", "ddim_uniform"), "UniPC": ("uni_pc", "normal"),
    # few-step models (LCM-LoRA, Lightning, Hyper-SD, TCD, DMD2): the scheduler they are made for
    "LCM": ("lcm", "sgm_uniform"), "Euler SGM Uniform": ("euler", "sgm_uniform"),
    "Euler a SGM Uniform": ("euler_ancestral", "sgm_uniform"), "DPM++ SDE SGM Uniform": ("dpmpp_sde", "sgm_uniform"),
    # SD/SDXL Turbo with ComfyUI's SDTurboScheduler: 'steps' are the real steps, from the noise level of the denoise
    # (1 step works in img2img), instead of A1111's tail of a full schedule
    "Euler a SD Turbo": ("euler_ancestral", "sd_turbo"),
    # modern flow models: FLUX.2 Klein (its resolution-dependent schedule) and Z-Image / Lumina2 / AuraFlow-like
    "Euler Flux2": ("euler", "flux2"), "Euler Simple": ("euler", "simple"),
    "Res Multistep Simple": ("res_multistep", "simple"),
}

# Features of A1111 Deforum that this port does not do (yet): if a settings file turns them on we say so.
UNSUPPORTED = {
    "hybrid_motion": "None", "hybrid_composite": "None", "optical_flow_cadence": "None",
    "optical_flow_redo_generation": "None", "use_mask": False, "use_noise_mask": False, "use_mask_video": False,
    "enable_checkpoint_scheduling": False, "resume_from_timestring": False,
}


def sampler_for(name):
    key = str(name).strip()
    for a1111, comfy in SAMPLERS.items():
        if a1111.casefold() == key.casefold():
            return comfy
    raise ValueError(f"unknown sampler '{name}'. Known: {', '.join(SAMPLERS)}")


def default_settings():
    return dict(DEFAULTS)


def _as_bool(v):
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def normalize(settings):
    """Types as the engine expects them; W and H down to a multiple of 8 (A1111 did the same)."""
    s = dict(settings)
    for k in ("W", "H", "steps", "max_frames", "seed", "seed_iter_N", "perlin_w", "perlin_h", "perlin_octaves",
              "diffusion_cadence", "prompt_blend_frames"):
        s[k] = int(float(s[k]))
    for k in ("strength", "perlin_persistence", "midas_weight", "fps", "transition_strength"):
        s[k] = float(s[k])
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            s[k] = _as_bool(s[k])
    s["W"] -= s["W"] % 8
    s["H"] -= s["H"] % 8
    if s["animation_mode"] not in ANIMATION_MODES:
        raise ValueError(f"animation_mode '{s['animation_mode']}' is not supported (use {', '.join(ANIMATION_MODES)})")
    if s["color_coherence"] == "Video Input":
        s["color_coherence"] = "None"         # needs hybrid video in Deforum
    return s


def unsupported_in(settings):
    found = []
    for k, off in UNSUPPORTED.items():
        if k not in settings or settings[k] is None:
            continue
        on = _as_bool(settings[k]) if isinstance(off, bool) else str(settings[k]).strip() not in ("", "None", off)
        if on:
            found.append(k)
    if settings.get("diffusion_redo") not in (None, 0, "0"):
        found.append("diffusion_redo")
    return found


def load_settings_text(text):
    """A1111 Deforum settings file (JSON) -> (settings, prompts, positive, negative, warnings)."""
    data = json.loads(text)
    settings = default_settings()
    for k in DEFAULTS:
        if k in data:
            settings[k] = data[k]
    prompts = data.get("prompts", data.get("animation_prompts", DEFAULT_PROMPTS))
    if isinstance(prompts, str):
        prompts = json.loads(prompts)
    positive = data.get("animation_prompts_positive", data.get("positive_prompts", ""))
    negative = data.get("animation_prompts_negative", data.get("negative_prompts", ""))
    warnings = [f"'{k}' is on in the file but not supported by this port: ignored" for k in unsupported_in(data)]
    return settings, {str(k): str(v) for k, v in prompts.items()}, positive or "", negative or "", warnings

# Ready recipes (extension of this port). DEFAULTS above stay A1111's, so an A1111 settings file behaves as in A1111.
# "Turbo veloce": SDXL Turbo (sd_xl_turbo) + a TAESDXL VAE, 1 real step per frame, about 0.7-0.9 s a frame at 768x432
# on an RTX 3060; prompts morph into each other, colours follow every scene, the camera changes direction by itself.
_TURBO = {
    "W": 768, "H": 432, "max_frames": 384, "sampler": "Euler a SD Turbo", "steps": 1, "cfg_scale_schedule": "0: (1)",
    "strength_schedule": "0: (0.65)", "noise_schedule": "0: (0.065)",
    "color_coherence": "LAB", "color_coherence_anchor": "scene", "prompt_blend_frames": 66, "transition_strength": 0.5,
    "fps": 25, "prompt_mix": "embeddings",
}
# The camera of the fast presets. 25 fps with the speed the user liked watching the first tests at 12 fps: the motion
# per frame is 12/25 of the first recipe (zoom: its 12/25 power) and the periods of the curves are 25/12 longer, like
# the scenes (every 96 frames instead of 46) and the morph (66 frames instead of 32). 18/10/2026.
_CAM_3D = {"zoom": "0: (1.0)",
           "translation_z": "0: (1.44 + 0.96*sin(2*pi*t/360))",
           "translation_x": "0: (1.92*sin(2*pi*t/285))",
           "translation_y": "0: (1.2*sin(2*pi*t/398 + 1))",
           "rotation_3d_y": "0: (0.432*sin(2*pi*t/235))",
           "rotation_3d_x": "0: (0.24*sin(2*pi*t/327 + 2))",
           "rotation_3d_z": "0: (0.288*sin(2*pi*t/185))"}
_CAM_2D = {"zoom": "0: (1.0048 + 0.00576*sin(2*pi*t/202))",
           "angle": "0: (0.576*sin(2*pi*t/235))",
           "translation_x": "0: (1.92*sin(2*pi*t/285))",
           "translation_y": "0: (1.2*sin(2*pi*t/398 + 1))"}
# Z-Image fills every frame with new detail: it is redrawn less (strength 0.85) and with less pixel noise, otherwise
# the details pile up from frame to frame (test 18/10/2026: 0.65/0.065 confused, 0.85/0.03 cleaner; still to be tuned).
_ZIMAGE = {"sampler": "Res Multistep Simple", "steps": 8, "strength_schedule": "0: (0.85)", "noise_schedule": "0: (0.03)"}
_KLEIN = {"sampler": "Euler Flux2", "steps": 4}
PRESETS = {
    "Turbo veloce 3D": dict({**DEFAULTS, **_TURBO, **_CAM_3D}, animation_mode="3D"),
    "Turbo veloce 2D": dict({**DEFAULTS, **_TURBO, **_CAM_2D}, animation_mode="2D"),
    "Klein 4B veloce 3D": dict({**DEFAULTS, **_TURBO, **_KLEIN, **_CAM_3D}, animation_mode="3D"),
    "Klein 4B veloce 2D": dict({**DEFAULTS, **_TURBO, **_KLEIN, **_CAM_2D}, animation_mode="2D"),
    "Z-Image Turbo veloce 3D": dict({**DEFAULTS, **_TURBO, **_ZIMAGE, **_CAM_3D}, animation_mode="3D"),
    "Z-Image Turbo veloce 2D": dict({**DEFAULTS, **_TURBO, **_ZIMAGE, **_CAM_2D}, animation_mode="2D"),
    "Deforum classico (A1111)": dict(DEFAULTS),
}
PRESET_MODELS = {
    "Turbo veloce 3D": "SDXL Turbo checkpoint + Load VAE 'taesdxl' + a depth model (Load Depth Anything 3)",
    "Turbo veloce 2D": "SDXL Turbo checkpoint + Load VAE 'taesdxl'",
    "Klein 4B veloce 3D": "UNETLoader flux-2-klein-4b + CLIPLoader qwen_3_4b (type flux2) + VAELoader flux2-vae "
                         "+ a depth model (Load Depth Anything 3)",
    "Klein 4B veloce 2D": "UNETLoader flux-2-klein-4b(-fp8) + CLIPLoader qwen_3_4b (type flux2) + VAELoader flux2-vae",
    "Z-Image Turbo veloce 3D": "UNETLoader z_image_turbo_int8_convrot + Model Attention Backend (comfy kitchen) + ModelSamplingAuraFlow (shift 3) + CLIPLoader qwen_3_4b "
                               "(type lumina2) + VAELoader ae + a depth model (Load Depth Anything 3)",
    "Z-Image Turbo veloce 2D": "UNETLoader z_image_turbo_int8_convrot + Model Attention Backend (comfy kitchen) + ModelSamplingAuraFlow (shift 3) + CLIPLoader qwen_3_4b "
                               "(type lumina2) + VAELoader ae",
    "Deforum classico (A1111)": "an SD 1.5 checkpoint (as in AUTOMATIC1111)",
}
BIZARRE_PROMPTS = {
    "0": "a giant clockwork jellyfish floating over a flooded gothic city at dusk, tiny lanterns hanging from its "
         "tentacles, surreal, bizarre, highly detailed, cinematic",
    "96": "a victorian tea party on the surface of the moon, elephants in tuxedos pouring tea from floating teapots, "
          "pocket watches orbiting the earth, surreal, bizarre, highly detailed, cinematic",
    "192": "a neon cyberpunk fish market inside the ribcage of a gigantic whale, octopus merchants with paper umbrellas, "
          "a rain of glowing coins, surreal, bizarre, highly detailed, cinematic",
    "288": "a desert of melting grand pianos under three red suns, a giant marble cat playing chess against a rusty "
           "robot knight, surreal, bizarre, highly detailed, cinematic",
}
