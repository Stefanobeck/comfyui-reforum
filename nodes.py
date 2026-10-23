"""ComfyUI nodes of Deforum (revived). The engine is in deforum_core/ (ported from the Deforum extension for
AUTOMATIC1111, MIT, commit a7b829e); this file replaces AUTOMATIC1111 with ComfyUI: CLIP text encoding, VAE, the native
samplers (img2img done like A1111: denoise = 1 - strength, int(steps * denoise) + 1 steps on the tail of the
schedule), the native depth models (Depth Anything 3, MoGe) and the progress bar with a preview of every frame."""
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from .deforum_core import settings as ds
from .deforum_core.prompt import split_and
from .deforum_core.render import render_animation
from .deforum_core.seed import fixed_seed

try:
    import comfy.model_management as mm
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import folder_paths
    import node_helpers
except ImportError:            # outside ComfyUI (tests of the engine)
    mm = None

CATEGORY = "Deforum (revived)"

# Schedules shown as widgets on the Settings node (names of A1111 Deforum); the others go in extra_json.
_SCHEDULE_FIELDS = ["angle", "zoom", "translation_x", "translation_y", "translation_z", "transform_center_x",
                    "transform_center_y", "rotation_3d_x", "rotation_3d_y", "rotation_3d_z", "strength_schedule",
                    "noise_schedule", "contrast_schedule", "cfg_scale_schedule", "fov_schedule", "near_schedule",
                    "far_schedule", "amount_schedule", "kernel_schedule", "sigma_schedule", "threshold_schedule"]


def _rgb_to_bgr_u8(t):
    """ComfyUI IMAGE (H, W, C) float -> uint8 BGR numpy (Deforum's frames)."""
    a = (t[..., :3].float().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(a[..., ::-1])


def _bgr_u8_to_rgb(a):
    return torch.from_numpy(np.ascontiguousarray(a[..., ::-1]).astype(np.float32) / 255.0)


def _resize_bgr(a, w, h):
    if a.shape[1] == w and a.shape[0] == h:
        return a
    t = torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    return t[0].permute(1, 2, 0).clamp(0, 255).round().numpy().astype(np.uint8)


def _slerp(val, low, high):
    """A1111's variation seed (subseed) blend, on (B, N) tensors."""
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm * high_norm).sum(1)
    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    return ((torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low
            + (torch.sin(val * omega) / so).unsqueeze(1) * high)


class ComfyEngine:
    """What AUTOMATIC1111 did for Deforum, done with ComfyUI."""

    def __init__(self, model, clip, vae, settings, da3_model=None, moge_model=None, depth_maps=None, preview=True,
                 out_dir=None, timestring="", save_video=False, save_pngs=True):
        self.model, self.clip, self.vae, self.s = model, clip, vae, settings
        self.da3_model, self.moge_model, self.depth_maps = da3_model, moge_model, depth_maps
        self.cond_cache = {}
        self.latent_shape = None
        self.bar = comfy.utils.ProgressBar(settings["max_frames"])
        self.preview = preview
        self.out_dir, self.timestring = out_dir, timestring
        self.save_pngs = bool(out_dir and save_pngs)
        self.saver = ThreadPoolExecutor(max_workers=2) if self.save_pngs else None   # PNGs written while the next frame runs
        self.save_video = bool(out_dir and save_video)
        self.video, self.stream = None, None
        self.last = None

    def _encode(self, text, clipskip):
        # Every prompt text is encoded once (cache per text, not per whole prompt): during a morph the prompt is
        # 'old :w AND new :1-w' with weights changing every frame, and only the weights change, not the texts. With a
        # big text encoder (Qwen3-4B for Klein / Z-Image) re-encoding every frame swapped models in VRAM each frame.
        parts = split_and(text)
        if len(parts) > 1 and self.s.get("prompt_mix", "predictions") == "embeddings":
            # one conditioning: the encoded texts mixed by weight with ComfyUI's ConditioningAverage, so the model
            # runs once per step (a morph costs as much as a fixed scene)
            import nodes as comfy_nodes
            average = comfy_nodes.ConditioningAverage()
            mixed, acc = None, 0.0
            for part, weight in parts:
                c = self._encode_text(part, clipskip)
                weight = max(float(weight), 0.0)
                if mixed is None:
                    mixed, acc = c, weight
                elif weight > 0:
                    mixed = average.addWeighted(c, mixed, weight / (acc + weight))[0]
                    acc += weight
            return mixed
        cond = []
        for part, weight in parts:                         # A1111 'AND': one conditioning each, with its weight
            c = self._encode_text(part, clipskip)
            if weight != 1.0:
                c = node_helpers.conditioning_set_values(c, {"strength": weight})
            cond += c
        return cond

    def _encode_text(self, part, clipskip):
        key = (part, clipskip)
        if key not in self.cond_cache:
            clip = self.clip
            if clipskip:
                clip = clip.clone()
                clip.clip_layer(-int(clipskip))
            self.cond_cache[key] = clip.encode_from_tokens_scheduled(clip.tokenize(part))
        return self.cond_cache[key]

    def generate(self, init, denoise, positive, negative, seed, cfg, steps, sampler, clipskip, noise_multiplier,
                 subseed, subseed_strength):
        w, h = self.s["W"], self.s["H"]
        if seed == -1:                                        # A1111 (get_fixed_seed): -1 = a random seed
            seed = random.randrange(4294967294)
        if subseed == -1 and subseed_strength > 0:
            subseed = random.randrange(4294967294)
        sampler_name, scheduler = ds.sampler_for(sampler)
        pos, neg = self._encode(positive, clipskip), self._encode(negative, clipskip)
        if init is not None:
            latent = self.vae.encode(_bgr_u8_to_rgb(_resize_bgr(init, w, h))[None])
        else:
            if self.latent_shape is None:
                self.latent_shape = self.vae.encode(torch.zeros(1, h, w, 3)).shape
            latent = comfy.sample.fix_empty_latent_channels(self.model, torch.zeros(self.latent_shape))
        noise = comfy.sample.prepare_noise(latent, seed)
        if subseed_strength > 0:
            sub = comfy.sample.prepare_noise(latent, subseed)
            noise = _slerp(subseed_strength, noise.flatten(1), sub.flatten(1)).reshape(noise.shape)
        if init is not None and noise_multiplier and noise_multiplier != 1.0:
            noise = noise * noise_multiplier                  # A1111's initial_noise_multiplier (img2img only)
        if scheduler == "flux2":
            # FLUX.2 Klein: the schedule of ComfyUI's Flux2Scheduler (it depends on the image size), then the tail of
            # it as A1111 did for img2img (int(steps * denoise) + 1 steps)
            from comfy_extras.nodes_flux import get_schedule
            sigmas = torch.as_tensor(get_schedule(steps, round(w * h / (16 * 16))), dtype=torch.float32)
            if init is not None:
                t_enc = int(min(denoise, 0.999) * steps)
                sigmas = sigmas[steps - t_enc - 1:]
            samples = comfy.sample.sample(self.model, noise, steps, cfg, sampler_name, "normal", pos, neg, latent,
                                          sigmas=sigmas, seed=seed, disable_pbar=True)
        elif scheduler == "sd_turbo":
            # ComfyUI's SDTurboScheduler (comfy_extras/nodes_custom_sampler.py): timesteps 999, 899 ... 99, starting
            # from the one of the denoise; 'steps' real steps (at least one, also for a very small denoise)
            start = min(10 - int(10 * (denoise if init is not None else 1.0)), 10 - max(1, steps))
            timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[start:start + steps]
            sigmas = self.model.get_model_object("model_sampling").sigma(timesteps)
            sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
            samples = comfy.sample.sample(self.model, noise, steps, cfg, sampler_name, "normal", pos, neg, latent,
                                          sigmas=sigmas, seed=seed, disable_pbar=True)
        elif init is not None:
            t_enc = int(min(denoise, 0.999) * steps)          # A1111 img2img with img2img_fix_steps off
            start = steps - t_enc - 1
            samples = comfy.sample.sample(self.model, noise, steps, cfg, sampler_name, scheduler, pos, neg, latent,
                                          denoise=1.0, start_step=start, last_step=steps, force_full_denoise=True,
                                          seed=seed, disable_pbar=True)
        else:
            samples = comfy.sample.sample(self.model, noise, steps, cfg, sampler_name, scheduler, pos, neg, latent,
                                          denoise=1.0, seed=seed, disable_pbar=True)
        img = self.vae.decode(samples)
        if img.dim() == 5:                                    # video VAEs: (B, T, H, W, C)
            img = img.reshape(-1, *img.shape[-3:])
        return _resize_bgr(_rgb_to_bgr_u8(img[0]), w, h)

    def predict_depth(self, img, frame_idx):
        if self.depth_maps is not None:                       # given maps: bright = near
            d = self.depth_maps[min(frame_idx, self.depth_maps.shape[0] - 1)].float()
            d = d[..., :3].mean(-1)
            if d.shape != img.shape[:2]:
                d = F.interpolate(d[None, None], size=img.shape[:2], mode="bilinear", align_corners=False)[0, 0]
            return d, "near_bright"
        rgb = _bgr_u8_to_rgb(img)[None]
        if self.da3_model is not None:
            from comfy_extras.nodes_depth_anything_3 import _run_da3
            depth = _run_da3(self.da3_model, rgb, 504)[0][0].float()
        elif self.moge_model is not None:
            res = self.moge_model.infer(rgb.movedim(-1, -3), resolution_level=9, fov_x=None, force_projection=True,
                                        apply_mask=False, refine_steps=0)
            depth = res["depth"][0].float().cpu()
            if depth.shape != img.shape[:2]:
                depth = F.interpolate(depth[None, None], size=img.shape[:2], mode="bilinear",
                                      align_corners=False)[0, 0]
        else:
            return None                                       # no depth model: flat, as Deforum without depth
        finite = torch.isfinite(depth)
        if not finite.all():
            depth = torch.where(finite, depth, depth[finite].max() if finite.any() else torch.ones_like(depth))
        return depth, "far_bright"                            # distances, like Zoe/AdaBins in Deforum

    def frame_done(self, frame_idx, image, total):
        from PIL import Image
        pil = Image.fromarray(np.ascontiguousarray(image[..., ::-1]))
        if self.save_pngs:                                    # as Deforum: one PNG per frame, written as it is made
            self.saver.submit(pil.save, os.path.join(self.out_dir, f"{self.timestring}_{frame_idx:09}.png"))
        if self.save_video:                                   # the MP4 grows with the frames: playable if stopped
            self._video_frame(pil)
        self.last = image
        self.bar.update_absolute(min(frame_idx + 1, total), total, ("JPEG", pil, 512) if self.preview else None)

    def check_interrupt(self):
        mm.throw_exception_if_processing_interrupted()

    def _video_frame(self, pil):
        import av
        if self.video is None:
            path = os.path.join(self.out_dir, f"{self.timestring}.mp4")
            # fragmented MP4: the file can be watched while it grows, and stays valid if ComfyUI is closed
            self.video = av.open(path, "w", options={"movflags": "frag_keyframe+empty_moov+default_base_moof",
                                                     "flush_packets": "1"})
            self.stream = self.video.add_stream("libx264", rate=max(1, int(round(self.s["fps"]))))
            self.stream.width, self.stream.height = pil.size[0] - pil.size[0] % 2, pil.size[1] - pil.size[1] % 2
            self.stream.pix_fmt = "yuv420p"
            # a fragment a second, no frame held back by the encoder: the file is always up to the last frame
            self.stream.options = {"crf": "18", "g": str(max(1, int(round(self.s["fps"])))), "tune": "zerolatency"}
        if pil.size != (self.stream.width, self.stream.height):
            pil = pil.crop((0, 0, self.stream.width, self.stream.height))
        for packet in self.stream.encode(av.VideoFrame.from_image(pil)):
            self.video.mux(packet)

    def close(self):
        if self.saver is not None:
            self.saver.shutdown(wait=True)          # every PNG on disk before the node returns
        if self.video is not None:
            try:
                for packet in self.stream.encode():
                    self.video.mux(packet)
            finally:
                self.video.close()
            _to_regular_mp4(os.path.join(self.out_dir, f"{self.timestring}.mp4"))


def _to_regular_mp4(path):
    """The fragmented MP4 (watchable while it grows) declares no duration, so Windows Explorer and some players show
    it as empty. At the end it is rewritten as a normal MP4 with its index at the start: the packets are copied, not
    re-encoded. If anything goes wrong the fragmented file stays as it is."""
    import av
    tmp = path + ".tmp.mp4"
    try:
        with av.open(path) as src, av.open(tmp, "w", options={"movflags": "faststart"}) as dst:
            s_in = src.streams.video[0]
            s_out = dst.add_stream_from_template(s_in)
            for packet in src.demux(s_in):
                if packet.dts is None:
                    continue
                packet.stream = s_out
                dst.mux(packet)
        os.replace(tmp, path)
    except Exception as e:                                    # e.g. the MP4 is open in a player
        if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            normal = path[:-4] + "_normal.mp4"
            os.replace(tmp, normal)
            print(f"[Deforum revived] {os.path.basename(path)} is in use: the normal MP4 is {os.path.basename(normal)} ({e})")
        else:
            print(f"[Deforum revived] the MP4 stays fragmented ({e})")


def _bundle(settings, prompts, positive, negative, warnings):
    return {"settings": settings, "prompts": prompts, "positive": positive, "negative": negative,
            "warnings": warnings}


class DeforumRevivedSettings:
    """The parameters of A1111 Deforum, with the same names and defaults. Anything else goes in 'extra_json'."""

    @classmethod
    def INPUT_TYPES(cls):
        d = ds.PRESETS["Turbo veloce 3D"]               # the default recipe (A1111's values: the Preset node)
        req = {
            "animation_mode": (ds.ANIMATION_MODES, {"default": d["animation_mode"]}),
            "max_frames": ("INT", {"default": d["max_frames"], "min": 1, "max": 100000}),
            "W": ("INT", {"default": d["W"], "min": 64, "max": 8192, "step": 8}),
            "H": ("INT", {"default": d["H"], "min": 64, "max": 8192, "step": 8}),
            "seed": ("INT", {"default": d["seed"], "min": -1, "max": 0xFFFFFFFF,
                             "tooltip": "-1 = a random seed at every run, as in Deforum."}),
            "seed_behavior": (ds.SEED_BEHAVIORS, {"default": d["seed_behavior"]}),
            "sampler": (list(ds.SAMPLERS), {"default": d["sampler"]}),
            "steps": ("INT", {"default": d["steps"], "min": 1, "max": 200}),
            "diffusion_cadence": ("INT", {"default": d["diffusion_cadence"], "min": 1, "max": 50,
                                          "tooltip": "Generate one frame every N; the ones in between are warped "
                                                     "and blended (faster, smoother)."}),
            "border": (ds.BORDERS, {"default": d["border"]}),
            "color_coherence": (ds.COLOR_COHERENCE, {"default": d["color_coherence"]}),
            "noise_type": (ds.NOISE_TYPES, {"default": d["noise_type"]}),
            "use_depth_warping": ("BOOLEAN", {"default": d["use_depth_warping"]}),
            "padding_mode": (ds.PADDING_MODES, {"default": d["padding_mode"]}),
            "sampling_mode": (ds.SAMPLING_MODES, {"default": d["sampling_mode"]}),
            "strength": ("FLOAT", {"default": d["strength"], "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "Strength of the init image on the FIRST frame (Deforum's Init tab)."}),
        }
        for name in _SCHEDULE_FIELDS:
            req[name] = ("STRING", {"default": d[name]})
        req["prompts"] = ("STRING", {"multiline": True, "default": json.dumps(ds.BIZARRE_PROMPTS, indent=4),
                                     "tooltip": 'JSON as in Deforum: {"0": "a prompt", "30": "another --neg ..."}'})
        req["positive_prompts"] = ("STRING", {"default": ""})
        req["negative_prompts"] = ("STRING", {"default": "nsfw, nude"})
        req["extra_json"] = ("STRING", {"multiline": True, "default": "{}",
                                        "tooltip": "Any other A1111 Deforum parameter, e.g. "
                                                   '{"enable_perspective_flip": true, '
                                                   '"perspective_flip_theta": "0:(5)"}'})
        opt = {
            "fps": ("FLOAT", {"default": d["fps"], "min": 1.0, "max": 120.0, "step": 1.0,
                              "tooltip": "Frames per second of the video (Deforum's Output tab, 15). The motion values "
                                         "are per frame: more fps = the same motion takes less time."}),
            "prompt_blend_frames": ("INT", {"default": d["prompt_blend_frames"], "min": 0, "max": 1000,
                                            "tooltip": "Morph between keyframed prompts over this many frames "
                                                       "before each keyframe (0 = Deforum's hard cut)."}),
            "color_coherence_anchor": (ds.COLOR_ANCHORS, {
                "default": d["color_coherence_anchor"],
                "tooltip": "first_frame = Deforum: every frame keeps the colours of the first one. scene = the "
                           "colours are free while the prompt changes, then the new scene's colours are kept."}),
            "prompt_mix": (ds.PROMPT_MIX, {"default": d["prompt_mix"],
                                         "tooltip": "embeddings = the two prompts of a morph mixed into one (fast). "
                                                    "predictions = A1111 AND, two model passes (slower)."}),
            "transition_strength": ("FLOAT", {"default": d["transition_strength"], "min": 0.0, "max": 1.0, "step": 0.01,
                                              "tooltip": "Around every prompt change the strength goes smoothly down "
                                                         "to this value and back, so the new scene comes in (0 = off)."}),
        }
        return {"required": req, "optional": opt}

    RETURN_TYPES = ("DEFORUM_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, prompts, positive_prompts, negative_prompts, extra_json, **values):
        s = ds.default_settings()
        s.update(values)
        extra = json.loads(extra_json or "{}")
        unknown = [k for k in extra if k not in ds.DEFAULTS]
        if unknown:
            raise ValueError(f"extra_json: unknown or unsupported parameters: {', '.join(unknown)}")
        s.update(extra)
        return (_bundle(s, {str(k): str(v) for k, v in json.loads(prompts).items()}, positive_prompts,
                        negative_prompts, []),)



class DeforumRevivedPreset:
    """A ready recipe: write the scenes, choose the length, render. 'Turbo veloce' needs SDXL Turbo + TAESDXL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preset": (list(ds.PRESETS), {"default": "Turbo veloce 3D",
                                          "tooltip": "\n".join(f"{k}: {v}" for k, v in ds.PRESET_MODELS.items())}),
            "prompts": ("STRING", {"multiline": True, "default": json.dumps(ds.BIZARRE_PROMPTS, indent=4),
                                   "tooltip": 'Scenes as in Deforum: {"0": "first scene", "96": "second scene", ...}'}),
            "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "0 = the preset's (384)."}),
            "W": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8, "tooltip": "0 = the preset's."}),
            "H": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8, "tooltip": "0 = the preset's."}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF}),
        }, "optional": {
            "positive_prompts": ("STRING", {"default": ""}),
            "negative_prompts": ("STRING", {"default": "nsfw, nude"}),
            "extra_json": ("STRING", {"multiline": True, "default": "{}",
                                      "tooltip": "Any parameter to change in the preset (A1111 Deforum names)."}),
        }}

    RETURN_TYPES = ("DEFORUM_SETTINGS", "STRING")
    RETURN_NAMES = ("settings", "models")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, preset, prompts, max_frames, W, H, seed, positive_prompts="", negative_prompts="nsfw, nude",
            extra_json="{}"):
        s = dict(ds.PRESETS[preset])
        for key, value in (("max_frames", max_frames), ("W", W), ("H", H)):
            if value:
                s[key] = value
        s["seed"] = seed
        extra = json.loads(extra_json or "{}")
        unknown = [k for k in extra if k not in ds.DEFAULTS]
        if unknown:
            raise ValueError(f"extra_json: unknown or unsupported parameters: {', '.join(unknown)}")
        s.update(extra)
        return (_bundle(s, {str(k): str(v) for k, v in json.loads(prompts).items()}, positive_prompts,
                        negative_prompts, []), ds.PRESET_MODELS[preset])

class DeforumRevivedLoadSettings:
    """Reads a settings file saved by A1111 Deforum (put it in ComfyUI's input folder) or the same JSON pasted."""

    @classmethod
    def INPUT_TYPES(cls):
        files = []
        try:
            folder = folder_paths.get_input_directory()
            files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".txt", ".json")))
        except Exception:
            pass
        return {"required": {"settings_file": (["(pasted text)"] + files,),
                             "settings_text": ("STRING", {"multiline": True, "default": ""}),
                             "overrides_json": ("STRING", {"multiline": True, "default": "{}",
                                                           "tooltip": "Parameters to change after loading, e.g. "
                                                                      '{"max_frames": 48, "W": 768}'})}}

    RETURN_TYPES = ("DEFORUM_SETTINGS", "STRING")
    RETURN_NAMES = ("settings", "warnings")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, settings_file, settings_text, overrides_json):
        if settings_file == "(pasted text)":
            text = settings_text
        else:
            path = os.path.join(folder_paths.get_input_directory(), os.path.basename(settings_file))
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        s, prompts, positive, negative, warnings = ds.load_settings_text(text)
        overrides = json.loads(overrides_json or "{}")
        unknown = [k for k in overrides if k not in ds.DEFAULTS]
        if unknown:
            raise ValueError(f"overrides_json: unknown or unsupported parameters: {', '.join(unknown)}")
        s.update(overrides)
        return (_bundle(s, prompts, positive, negative, warnings), "\n".join(warnings) or "ok")


class DeforumRevivedRender:
    """Renders the Deforum animation frame by frame (the loop is inside the node: one Queue = the whole video)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",), "settings": ("DEFORUM_SETTINGS",)},
            "optional": {
                "init_image": ("IMAGE", {"tooltip": "First frame to start from (Deforum's init image)."}),
                "guided_images": ("IMAGE", {"tooltip": "Guided Images mode (use_looper): the guide images; in "
                                                       "init_images write the frame and the index in this batch, "
                                                       "e.g. {\"0\": 0, \"max_f\": 0} = a loop on the first one."}),
                "video_init": ("IMAGE", {"tooltip": "Frames of a video, for animation_mode 'Video Input'."}),
                "color_image": ("IMAGE", {"tooltip": "Colour reference, for color_coherence 'Image'."}),
                "da3_model": ("DA3_MODEL", {"tooltip": "3D: Depth Anything 3 (Load Depth Anything 3 Model)."}),
                "moge_model": ("MOGE_MODEL", {"tooltip": "3D: MoGe (Load MoGe Model)."}),
                "depth_maps": ("IMAGE", {"tooltip": "3D: ready depth maps, bright = near, one per frame or one."}),
                "save_frames": ("BOOLEAN", {"default": True,
                                            "tooltip": "As Deforum: a folder in ComfyUI's output with every frame "
                                                       "(PNG, written as it is made) and the settings file."}),
                "batch_name": ("STRING", {"default": "Deforum_{timestring}",
                                          "tooltip": "Name of the folder; {timestring} = date and time."}),
                "save_video": ("BOOLEAN", {"default": True,
                                            "tooltip": "Also write <timestring>.mp4 in the folder, frame by frame as "
                                                       "they are made (there is a video also if you stop it)."}),
                "save_pngs": ("BOOLEAN", {"default": True,
                                          "tooltip": "With save_frames: also one PNG per frame in the folder. Off = "
                                                     "only the MP4 (and the settings file), much less disk."}),
                "keep_frames_in_memory": ("BOOLEAN", {"default": True,
                                                       "tooltip": "Off for long animations: the frames go only to "
                                                                  "the folder (PNG + MP4) and the node outputs only "
                                                                  "the last one, so the RAM stays small."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("frames", "depth", "fps", "folder")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, settings=None, **kwargs):
        """seed -1 = a new random seed at every Queue, so the cached result must not be reused."""
        try:
            return float("nan") if int(settings["settings"]["seed"]) == -1 else ""
        except Exception:
            return ""

    def run(self, model, clip, vae, settings, init_image=None, video_init=None, color_image=None, da3_model=None,
            moge_model=None, depth_maps=None, save_frames=True, batch_name="Deforum_{timestring}",
            save_video=True, keep_frames_in_memory=True, guided_images=None, save_pngs=True):
        s = ds.normalize(settings["settings"])
        for warning in settings.get("warnings", []):
            print(f"[Deforum revived] {warning}")
        s["use_init"] = s["use_init"] or init_image is not None
        s["seed"] = fixed_seed(s["seed"])                     # the seed really used goes in the settings file
        if s["animation_mode"] == "Video Input":
            if video_init is None:
                raise ValueError("animation_mode 'Video Input' needs the video frames on 'video_init'")
            s["max_frames"] = int(video_init.shape[0])        # as Deforum: as many frames as the video
        out_dir, timestring = None, time.strftime("%Y%m%d%H%M%S")
        if not keep_frames_in_memory:
            save_frames = True                                # the frames must go somewhere
            save_pngs = save_pngs or not save_video
        if save_frames:
            name = (batch_name or "Deforum_{timestring}").replace("{timestring}", timestring)
            name = "".join(c for c in name if c not in '<>:"/\\|?*').strip(" .") or f"Deforum_{timestring}"
            out_dir = os.path.join(folder_paths.get_output_directory(), name)
            os.makedirs(out_dir, exist_ok=True)
            saved = dict(s, prompts=settings["prompts"], animation_prompts_positive=settings.get("positive", ""),
                         animation_prompts_negative=settings.get("negative", ""))
            with open(os.path.join(out_dir, f"{timestring}_settings.json"), "w", encoding="utf-8") as f:
                json.dump(saved, f, ensure_ascii=False, indent=4)
            print(f"[Deforum revived] saving the frames to {out_dir}")
        engine = ComfyEngine(model, clip, vae, s, da3_model, moge_model, depth_maps, out_dir=out_dir,
                             timestring=timestring, save_video=save_video, save_pngs=save_pngs)
        w, h = s["W"], s["H"]
        init = _resize_bgr(_rgb_to_bgr_u8(init_image[0]), w, h) if init_image is not None else None
        color = _resize_bgr(_rgb_to_bgr_u8(color_image[0]), w, h) if color_image is not None else None
        def guide(value):
            """A guide image of init_images: an index in guided_images, or a file (absolute or in ComfyUI's input)."""
            if isinstance(value, (int, float)) or str(value).strip().lstrip("-").isdigit():
                if guided_images is None:
                    raise ValueError("init_images uses indexes: connect the guide images to 'guided_images'")
                i = int(value)
                if not -guided_images.shape[0] <= i < guided_images.shape[0]:
                    raise ValueError(f"init_images: index {i}, but guided_images has {guided_images.shape[0]} images")
                return _resize_bgr(_rgb_to_bgr_u8(guided_images[i]), w, h)
            path = str(value).strip()
            if not os.path.isabs(path):
                path = os.path.join(folder_paths.get_input_directory(), path)
            from PIL import Image
            img = Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)    # as Deforum's load_img
            return np.ascontiguousarray(np.asarray(img)[..., ::-1])

        video = None
        if video_init is not None:
            def video(idx):
                return _resize_bgr(_rgb_to_bgr_u8(video_init[min(idx, video_init.shape[0] - 1)]), w, h)
        try:
            frames, depths = render_animation(s, settings["prompts"], engine, settings.get("positive", ""),
                                              settings.get("negative", ""), init_image=init, video_frames=video,
                                              color_image=color, device=mm.get_torch_device(),
                                              keep_frames=keep_frames_in_memory, guide_loader=guide)
        finally:
            engine.close()
        order = sorted(frames)
        if not order:
            raise ValueError("Deforum produced no frames: check max_frames and diffusion_cadence")
        if not keep_frames_in_memory:                         # only the last frame comes out of the node
            frames, depths, order = {order[-1]: engine.last}, {}, [order[-1]]
        out = torch.stack([_bgr_u8_to_rgb(frames[i]) for i in order])
        dep = torch.zeros(len(order), h, w, 3)
        for k, i in enumerate(order):
            if depths.get(i) is not None:
                d = depths[i].float().cpu()
                d = (d - d.min()) / (d.max() - d.min()).clamp_min(1e-6)
                dep[k] = d[..., None].expand(h, w, 3)
        return (out, dep, float(s["fps"]), out_dir or "")


NODE_CLASS_MAPPINGS = {"DeforumRevivedPreset": DeforumRevivedPreset,
                       "DeforumRevivedSettings": DeforumRevivedSettings,
                       "DeforumRevivedLoadSettings": DeforumRevivedLoadSettings,
                       "DeforumRevivedRender": DeforumRevivedRender}
NODE_DISPLAY_NAME_MAPPINGS = {"DeforumRevivedPreset": "Deforum Preset (revived)",
                              "DeforumRevivedSettings": "Deforum Settings (revived)",
                              "DeforumRevivedLoadSettings": "Deforum Load Settings File (revived)",
                              "DeforumRevivedRender": "Deforum Render (revived)"}
