"""Deforum frame noise (Perlin-shaped or uniform), added in pixels before every img2img.

Port of noise.py of the Deforum extension for AUTOMATIC1111 (MIT, commit a7b829e). The random numbers come from the
same CPU generator, drawn in the same order with the same seed (so they are Deforum's); the maths on the whole image
runs on 'device' (the GPU inside ComfyUI). cv2.addWeighted is rewritten with torch.
The Perlin noise itself is perlin_clean.py, written in a clean room from a specification of the public algorithm
(dev_tools/deforum/stanza_pulita in the development repository): same numbers as before, bit for bit.
"""
import numpy as np
import torch
from torch.nn.functional import interpolate

from . import perlin_clean

deforum_noise_gen = torch.Generator(device="cpu")


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5, device="cpu"):
    """Perlin noise drawn from Deforum's generator (perlin_clean.py: clean-room rewrite from the public algorithm)."""
    return perlin_clean.rand_perlin_2d_octaves(shape, res, octaves, persistence, deforum_noise_gen, device)


def add_noise(sample, noise_amt, seed, noise_type, noise_args, device="cpu"):
    """sample: uint8 (H, W, 3) numpy. noise_args: (perlin_w, perlin_h, octaves, persistence). Returns uint8 numpy."""
    deforum_noise_gen.manual_seed(int(seed))
    perlin_w, perlin_h = sample.shape[0], sample.shape[1]
    perlin_w, perlin_h = map(lambda x: x - x % 64, (perlin_w, perlin_h))
    shape2d = (perlin_w, perlin_h)
    noise = torch.randn((sample.shape[2], perlin_w, perlin_h), generator=deforum_noise_gen).to(device)
    if noise_type == "perlin":
        noise = noise * ((rand_perlin_2d_octaves(shape2d, (int(noise_args[0]), int(noise_args[1])),
                                                 octaves=noise_args[2], persistence=noise_args[3], device=device)
                          + torch.ones(shape2d, device=device)) / 2)
    if noise.shape[1:] != sample.shape[:2]:
        # Deforum resized only the Perlin noise; the uniform one crashed when W or H was not a multiple of 64
        noise = interpolate(noise.unsqueeze(1), size=(sample.shape[0], sample.shape[1])).squeeze(1)
    noise_to_add = (((noise.permute(1, 2, 0).float() * 0.5) + 0.5).clamp(0, 1) * 255).to(torch.uint8)  # truncates
    img = torch.from_numpy(np.ascontiguousarray(sample)).to(device)
    out = img.double() * (1 - noise_amt) + noise_to_add.double() * noise_amt                          # addWeighted
    return torch.round(out).clamp(0, 255).to(torch.uint8).cpu().numpy()
