# Reforum: Deforum animations for ComfyUI

**Reforum** (formerly "Deforum revived") brings Deforum animations back to ComfyUI: an **unofficial**, revived port
of the [Deforum](https://deforum.github.io/) animation engine — the one of the
AUTOMATIC1111 extension — to ComfyUI 0.40+ (Python 3.13, torch 2.10, numpy 2). The classic Deforum look:
frame-by-frame img2img feedback with keyframed 2D/3D camera motion, colour coherence, Perlin noise and anti-blur.

It is not affiliated with, nor endorsed by, the Deforum team. All credit for the engine goes to
*deforum and contributors*; see [Credits](#credits).

## Why

The existing Deforum nodes for ComfyUI no longer load on recent ComfyUI (they rely on the old auto-queue, on a
hook into ComfyUI's executor and on Deforum Studio, which pins numpy 1.x and an old diffusers). This package
ports the engine itself, so it runs inside one node with ComfyUI's own samplers.

## Nodes

| Node | What it does |
|---|---|
| **Deforum Preset (revived)** | The quick way: choose a recipe, write the scenes. **Turbo veloce 3D** and **Turbo veloce 2D** (SDXL Turbo + TAESDXL, 1 real step a frame, prompts that morph into each other, colours that follow every scene, a camera that changes direction by itself; about 0.7-0.9 s a frame at 768×432 on an RTX 3060) or **Deforum classico (A1111)** (the historical defaults). `max_frames`, `W`, `H` at 0 = the preset's. |
| **Deforum Settings (revived)** | Every parameter, with the names of A1111 Deforum; the widgets start from the Turbo veloce 3D recipe (the historical A1111 values are the Deforum classico preset, and the defaults of a loaded A1111 settings file). Prompts as Deforum JSON. Any other parameter in `extra_json`. |
| **Deforum Load Settings File (revived)** | Reads a settings file saved by A1111 Deforum (put it in ComfyUI's `input` folder) or the same JSON pasted, with optional overrides. Unsupported options found in the file are reported. |
| **Deforum Render (revived)** | The engine. `MODEL`, `CLIP`, `VAE`, settings; optional `init_image`, `video_init` (Video Input mode), `color_image` (colour coherence "Image"), `da3_model` / `moge_model` / `depth_maps` (3D). The loop runs inside the node: one Queue renders the whole animation, with a preview of every frame. As Deforum did, every frame is also written as it is made to a folder in ComfyUI's `output` (`Deforum_<timestring>/<timestring>_000000000.png` …) together with `<timestring>_settings.json` (the same JSON A1111 Deforum wrote in its `_settings.txt`), which **Load Settings File** reads back (as well as A1111 `.txt` files) (the real seed is in it, also when you asked for -1). Outputs `frames`, `depth`, `fps`, `folder`. |

Then use ComfyUI's native nodes: `CreateVideo` + `SaveVideo`, and `FrameInterpolate` (RIFE/FILM) if you want.
Examples (API format) are in [`examples/`](examples): `turbo_veloce_3d/2d`, `klein4b_veloce_3d/2d`,
`zimage_turbo_veloce_3d/2d` (`*_api.json`) and the classic SD 1.5 ones. The fast examples have a **Load Image**
on `init_image` (the first frame starts from that picture, kept as much as `strength_schedule` says): pick your
image there, or bypass the node (Ctrl+B) to start from the prompt alone.

## What is ported

- **2D motion**: angle, zoom, translation x/y, transform centre, border replicate/wrap, perspective flip.
- **3D motion**: translation x/y/z, rotation 3D x/y/z, fov, near/far, aspect ratio; the depth warp of Deforum
  (adapted from Disco Diffusion, PyTorch3D cameras). Depth from ComfyUI's native **Depth Anything 3** or **MoGe**,
  re-estimated on every frame like Deforum did, or from your own depth maps.
- **Schedules** with Deforum's syntax and maths (`t`, `max_f`, `s`, `sin`, `cos`, `where` …), evaluated safely
  (no `eval`, no numexpr).
- **Generation**: strength, noise, contrast, cfg, steps scheduling, sampler scheduling, clip skip scheduling, noise
  multiplier, subseed; txt2img for the first frame; init image; **cadence**; **Video Input** mode.
- **Colour coherence** LAB / HSV / RGB / Image, grayscale; **seed behaviour** iter / fixed / random / ladder /
  alternate / schedule; Perlin and uniform **noise**; **anti-blur** (unsharp mask).
- **Prompts**: keyframed prompts, `--neg`, maths in backticks, global positive/negative prompts, A1111 `AND`.
- **Extension**: `prompt_blend_frames` morphs from one keyframed prompt to the next over N frames (the two prompts
  are mixed with moving weights); 0, the default, is Deforum's hard cut. `color_coherence_anchor: scene` releases the colour anchor around every
  prompt change and re-takes it from the new scene (Deforum keeps the first frame's colours for the whole
  animation, `first_frame`, the default). `transition_strength` lowers the strength smoothly around every prompt change (0, the default for A1111 files, = off). `fps` is on the Settings node.

**Same behaviour**: the order of the operations, the schedules, `denoise = 1 - strength`, and the A1111 step count
for img2img (`int(steps × denoise) + 1` steps on the tail of the schedule, i.e. KSamplerAdvanced, not KSampler's
`denoise`). The functions were checked one by one against the original code (commit `a7b829e`): schedules, 3D warp
and seeds are identical; 2D warp, colour coherence, noise and sharpening differ by about 0.2 levels (out of 255)
or less on average (OpenCV → torch/numpy). The pictures are not identical to an A1111 render because ComfyUI's sampler noise is
not A1111's.

Kept on purpose, because they are part of Deforum's look: the colour matching happens on BGR data passed to the
RGB→LAB/HSV conversions; with perspective flip on, the 2D translation is ignored (a `np.matmul` quirk of the
original). Changed on purpose: with cadence the last generated frame is in the output (Deforum dropped it); in
Video Input mode the length is the length of the video, as in Deforum.

## Speed

Warp, colour coherence, noise and anti-blur run on the GPU (torch), the PNGs are written in a background
thread. For SDXL Turbo use the sampler **Euler a SD Turbo** (ComfyUI's SDTurboScheduler: `steps` are real steps from
the noise level of the denoise, so 1 step works in img2img) with cfg `0:(1)`; a TAESDXL VAE (Load VAE → taesdxl)
makes the VAE round trip of every frame almost free. `diffusion_cadence` is Deforum's own speed-up.

Long animations: the Render node also writes `<timestring>.mp4` in the folder as the frames are made
(`save_video`, on by default: there is a video also if you stop the render), and with `keep_frames_in_memory` off
the frames go only to the folder and the node outputs only the last one, so thousands of frames do not fill the RAM.

## Not ported (yet)

Hybrid video (composite and motion), masks and noise masks, optical-flow cadence and redo, diffusion redo, guided
images (looper), checkpoint scheduling, resume, Interpolation mode, Parseq, ControlNet. The old MiDaS / AdaBins /
LeReS / Zoe models are replaced by ComfyUI's native depth models.

## Differences you may notice

- A1111's `[word]` de-emphasis is not understood by ComfyUI's text encoder; `(word:1.2)` works.
- A1111 `AND` is done with ComfyUI's conditioning strength, i.e. a weighted *average* of the predictions (A1111
  added them): `a :1 AND b :1` gives half of each, and a weight on a single prompt has no effect.
- Width and height are rounded down to a multiple of 8, as A1111 did.
- Contrast above 1 is clipped to 255 (the original overflowed the uint8 image).

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI. No dependencies beyond ComfyUI's own.

## Tests

`python tests/test_render.py` runs the render loop with a fake sampler (no model, CPU only).

## Credits

- **Deforum** — the extension for AUTOMATIC1111 ([deforum-art/sd-webui-deforum](https://github.com/deforum-art/sd-webui-deforum))
  and the Deforum Stable Diffusion notebook, by *deforum and contributors*. Only code released under the MIT
  License was ported: the extension as of commit `a7b829e` (2023-08-10) and the notebook.
- **Disco Diffusion** — the original `transform_image_3d` (Maxwell Ingham, Adam Letts, Alex Spirin; MIT).
- **PyTorch3D** — `py3d_tools.py` is the extract of PyTorch3D used by Deforum (Meta Platforms; BSD).
- The 2D Perlin noise (`perlin_clean.py`) is our own clean-room implementation of Ken Perlin's public algorithm,
  matching Deforum's noise bit for bit.

The licence texts are in [LICENSE](LICENSE).
