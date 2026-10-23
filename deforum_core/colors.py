"""Deforum colour coherence: histogram matching of each frame to a colour sample, in RGB, HSV or LAB.

Port of colors.py of the Deforum extension for AUTOMATIC1111 (MIT, commit a7b829e). Deforum used OpenCV for the
colour conversions and skimage.exposure.match_histograms for the matching; both are rewritten here with torch (so they
run on the GPU inside ComfyUI), following OpenCV's 8-bit conventions (fixed-point RGB->LAB/HSV, L*255/100, a+128,
b+128, H/2) and skimage's uint8 algorithm (with its truncation back to uint8).
Note, kept on purpose: Deforum keeps frames in BGR order but calls the RGB->LAB / RGB->HSV conversions, so the
matching happens in a "channel-swapped" LAB/HSV. The look of Deforum's colour coherence depends on it.
Images in and out are uint8 numpy arrays (H, W, 3), as in Deforum; 'device' only says where the maths runs.
"""
import numpy as np
import torch

_M_RGB2XYZ = np.array([[0.412453, 0.357580, 0.180423],
                       [0.212671, 0.715160, 0.072169],
                       [0.019334, 0.119193, 0.950227]])
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE = np.array([0.950456, 1.0, 1.088754])

# OpenCV's 8-bit RGB->Lab is fixed-point (color_lab.cpp, RGB2Lab_b): same tables and shifts here.
_LAB_SHIFT, _GAMMA_SHIFT = 12, 3
_LAB_SHIFT2 = _LAB_SHIFT + _GAMMA_SHIFT
_x = np.arange(256) / 255.0
_GAMMA_TAB = np.clip(np.round(255.0 * (1 << _GAMMA_SHIFT) * np.where(_x <= 0.04045, _x / 12.92,
                                                                    ((_x + 0.055) / 1.055) ** 2.4)), 0, 65535).astype(np.int64)
_CBRT_N = 255 * 3 // 2 * (1 << _GAMMA_SHIFT)
_xs = np.arange(_CBRT_N) * (1.0 / (255.0 * (1 << _GAMMA_SHIFT)))
_CBRT_TAB = np.round((1 << _LAB_SHIFT2) * np.where(_xs < 0.008856, _xs * 7.787 + 0.13793103448275862,
                                                   np.cbrt(_xs))).astype(np.int64)
_C = np.round((1 << _LAB_SHIFT) * _M_RGB2XYZ / _WHITE[:, None]).astype(np.int64)

# OpenCV's 8-bit RGB->HSV is fixed-point too (color_hsv.cpp, RGB2HSV_b, hsv_shift 12, hue range 180).
_HSV_SHIFT = 12
_SDIV = np.concatenate([[0], np.round((255 << _HSV_SHIFT) / np.arange(1, 256))]).astype(np.int64)
_HDIV = np.concatenate([[0], np.round((180 << _HSV_SHIFT) / (6.0 * np.arange(1, 256)))]).astype(np.int64)
_SECTOR = [(1, 3, 0), (1, 0, 2), (3, 0, 1), (0, 2, 1), (0, 1, 3), (2, 1, 0)]       # OpenCV sector_data (b, g, r)

_TABLES = {}


def _tab(name, device):
    key = (name, str(device))
    if key not in _TABLES:
        _TABLES[key] = torch.from_numpy(globals()[name]).to(device)
    return _TABLES[key]


def _t(img, device):
    return torch.from_numpy(np.ascontiguousarray(img)).to(device)


def _descale(v, n):
    return (v + (1 << (n - 1))) >> n


def _rgb2lab(t):
    """int64 tensor (H, W, 3) 0..255 -> int64 LAB (OpenCV 8-bit)."""
    d = t.device
    gamma, cbrt, c = _tab("_GAMMA_TAB", d), _tab("_CBRT_TAB", d), _tab("_C", d)
    r, g, b = gamma[t[..., 0]], gamma[t[..., 1]], gamma[t[..., 2]]
    fx, fy, fz = (cbrt[_descale(r * c[i, 0] + g * c[i, 1] + b * c[i, 2], _LAB_SHIFT).clamp(0, _CBRT_N - 1)]
                  for i in range(3))
    l_scale = (116 * 255 + 50) // 100
    l_shift = -((16 * 255 * (1 << _LAB_SHIFT2) + 50) // 100)
    lum = _descale(l_scale * fy + l_shift, _LAB_SHIFT2)
    a = _descale(500 * (fx - fy) + 128 * (1 << _LAB_SHIFT2), _LAB_SHIFT2)
    b = _descale(200 * (fy - fz) + 128 * (1 << _LAB_SHIFT2), _LAB_SHIFT2)
    return torch.stack([lum, a, b], -1).clamp(0, 255)


def _lab2rgb(lab):
    lab = lab.double()
    lum = lab[..., 0] * 100.0 / 255.0
    a, b = lab[..., 1] - 128.0, lab[..., 2] - 128.0
    fy = (lum + 16.0) / 116.0
    f = torch.stack([fy + a / 500.0, fy, fy - b / 200.0], -1)
    xyz = torch.where(f > 0.206893, f ** 3, (f - 16.0 / 116.0) / 7.787)
    xyz[..., 1] = torch.where(lum > 7.9996, ((lum + 16.0) / 116.0) ** 3, lum / 903.3)
    white = torch.tensor(_WHITE, device=lab.device)
    rgb = (xyz * white) @ torch.tensor(_M_XYZ2RGB.T, device=lab.device)
    rgb = rgb.clamp(0, 1)
    rgb = torch.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb.clamp_min(1e-12) ** (1 / 2.4) - 0.055)
    return torch.round(rgb * 255.0).clamp(0, 255).long()


def _rgb2hsv(t):
    d = t.device
    sdiv, hdiv = _tab("_SDIV", d), _tab("_HDIV", d)
    r, g, b = t[..., 0], t[..., 1], t[..., 2]
    v = t.max(-1).values
    diff = v - t.min(-1).values
    vr = v == r
    vg = (v == g) & ~vr
    s = (diff * sdiv[v] + (1 << (_HSV_SHIFT - 1))) >> _HSV_SHIFT
    h = torch.where(vr, g - b, torch.where(vg, b - r + 2 * diff, r - g + 4 * diff))
    h = (h * hdiv[diff] + (1 << (_HSV_SHIFT - 1))) >> _HSV_SHIFT
    h = torch.where(h < 0, h + 180, h)
    return torch.stack([h, s, v], -1).clamp(0, 255)


def _hsv2rgb(hsv):
    """OpenCV 4.x 8-bit HSV->RGB (it truncates the result)."""
    x = hsv.double()
    h = x[..., 0] / 30.0
    s = x[..., 1] / 255.0
    v = x[..., 2]
    h = torch.where(h >= 6, h - 6, h)
    sector = torch.floor(h).long()
    h = h - sector
    tab = [v, v * (1 - s), v * (1 - s * h), v * (1 - s * (1 - h))]
    out = torch.zeros_like(x)
    for k, (ib, ig, ir) in enumerate(_SECTOR):
        m = sector == k
        out[..., 2] = torch.where(m, tab[ib], out[..., 2])
        out[..., 1] = torch.where(m, tab[ig], out[..., 1])
        out[..., 0] = torch.where(m, tab[ir], out[..., 0])
    return torch.floor(out + 1e-7).clamp(0, 255).long()


def _interp(x, xp, fp):
    """numpy.interp in torch (xp increasing)."""
    i = torch.searchsorted(xp, x, right=True).clamp(1, xp.numel() - 1)
    x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    w = torch.where(x1 > x0, (x - x0) / (x1 - x0), torch.zeros_like(x))
    out = f0 + w * (f1 - f0)
    out = torch.where(x <= xp[0], fp[0], out)
    return torch.where(x >= xp[-1], fp[-1], out)


def _match_channel(source, template):
    """skimage.exposure._match_cumulative_cdf for unsigned integers; int64 tensors 0..255."""
    src = source.reshape(-1)
    src_counts = torch.bincount(src, minlength=256)
    tmpl_counts = torch.bincount(template.reshape(-1), minlength=256)
    tmpl_values = torch.nonzero(tmpl_counts).flatten()
    tmpl_counts = tmpl_counts[tmpl_values]
    src_quantiles = torch.cumsum(src_counts, 0).double() / src.numel()
    tmpl_quantiles = torch.cumsum(tmpl_counts, 0).double() / template.numel()
    values = _interp(src_quantiles, tmpl_quantiles, tmpl_values.double())
    return torch.floor(values[src]).long().reshape(source.shape)      # skimage writes into uint8: truncation


def _match(image, reference):
    return torch.stack([_match_channel(image[..., c], reference[..., c]) for c in range(image.shape[-1])], -1)


def maintain_colors(prev_img, color_match_sample, mode, device="cpu"):
    img, ref = _t(prev_img, device).long(), _t(color_match_sample, device).long()
    if mode == "RGB":
        out = _match(img, ref)
    elif mode == "HSV":
        out = _hsv2rgb(_match(_rgb2hsv(img), _rgb2hsv(ref)))
    else:                                                                # LAB
        out = _lab2rgb(_match(_rgb2lab(img), _rgb2lab(ref)))
    return out.to(torch.uint8).cpu().numpy()


# numpy-in / numpy-out versions of the single conversions (used by the equivalence test)
def rgb2lab_u8(img, device="cpu"):
    return _rgb2lab(_t(img, device).long()).to(torch.uint8).cpu().numpy()


def lab2rgb_u8(lab, device="cpu"):
    return _lab2rgb(_t(lab, device).long()).to(torch.uint8).cpu().numpy()


def rgb2hsv_u8(img, device="cpu"):
    return _rgb2hsv(_t(img, device).long()).to(torch.uint8).cpu().numpy()


def hsv2rgb_u8(hsv, device="cpu"):
    return _hsv2rgb(_t(hsv, device).long()).to(torch.uint8).cpu().numpy()
