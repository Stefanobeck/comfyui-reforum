"""2D Perlin noise in PyTorch.

Written from the specification dev_tools/deforum/stanza_pulita/SPEC_PERLIN.md only (clean-room),
MIT License, see LICENSE.

Classic gradient noise (Ken Perlin, 1985) with the quintic fade curve (Perlin, 2002).
"""

import math

import torch


def _fade(x):
    # 6x^5 - 15x^4 + 10x^3
    return x * x * x * (x * (x * 6 - 15) + 10)


def _lerp(a, b, w):
    return a + w * (b - a)


def rand_perlin_2d(shape, res, generator, device="cpu"):
    """One layer of Perlin noise of size `shape`, with `res` grid cells per axis.

    The random angles are drawn on the CPU from `generator` in a single call, then moved to `device`.
    """
    size_a, size_b = shape
    cells_a, cells_b = res
    step_a = cells_a / size_a
    step_b = cells_b / size_b
    cell_a = size_a // cells_a
    cell_b = size_b // cells_b

    # Local coordinates inside each cell, in [0, 1).
    coord_a = torch.arange(0, cells_a, step_a, device=device)
    coord_b = torch.arange(0, cells_b, step_b, device=device)
    local = torch.stack(torch.meshgrid(coord_a, coord_b, indexing="ij"), dim=-1) % 1
    u = local[..., 0]
    v = local[..., 1]

    # One random angle per grid vertex.
    angles = 2 * math.pi * torch.rand(cells_a + 1, cells_b + 1, generator=generator)
    grads = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).to(device)

    def corner(da, db):
        g = grads[da:da + cells_a, db:db + cells_b]
        return g.repeat_interleave(cell_a, 0).repeat_interleave(cell_b, 1)

    def dot(g, du, dv):
        return g[..., 0] * (u - du) + g[..., 1] * (v - dv)

    n00 = dot(corner(0, 0), 0, 0)
    n10 = dot(corner(1, 0), 1, 0)
    n01 = dot(corner(0, 1), 0, 1)
    n11 = dot(corner(1, 1), 1, 1)

    fu = _fade(u)
    fv = _fade(v)
    n0 = _lerp(n00, n10, fu)
    n1 = _lerp(n01, n11, fu)
    return math.sqrt(2) * _lerp(n0, n1, fv)


def rand_perlin_2d_octaves(shape, res, octaves, persistence, generator, device="cpu"):
    """Sum of `octaves` Perlin layers, doubling the frequency and scaling the amplitude by `persistence`."""
    total = torch.zeros(shape, dtype=torch.float32, device=device)
    frequency = 1
    amplitude = 1
    for _ in range(int(octaves)):
        layer_res = (frequency * res[0], frequency * res[1])
        total = total + amplitude * rand_perlin_2d(shape, layer_res, generator, device)
        frequency *= 2
        amplitude *= persistence
    return total
