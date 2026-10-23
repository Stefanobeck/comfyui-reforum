"""Deforum anti-blur: unsharp mask. Port of image_sharpening.py of the Deforum extension for AUTOMATIC1111 (MIT,
commit a7b829e); cv2.GaussianBlur is rewritten with torch (same kernel as cv2.getGaussianKernel, border
reflect-101, result rounded to uint8 like OpenCV), so it runs on 'device' (the GPU inside ComfyUI).
Images in and out are uint8 numpy arrays (H, W, 3)."""
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_kernel(ksize, sigma):
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    x = np.arange(ksize) - (ksize - 1) / 2.0
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(img, ksize, sigma):
    """img: uint8 tensor (H, W, C) -> uint8 tensor, cv2.GaussianBlur(img, ksize, sigma) (BORDER_REFLECT_101)."""
    kx, ky = ksize
    if kx % 2 == 0 or ky % 2 == 0 or kx < 1 or ky < 1:
        raise ValueError(f"the sharpen kernel must be an odd number (got {kx}): Deforum/OpenCV accept only 1, 3, 5, 7 ...")
    x = img.permute(2, 0, 1)[None].double()                                   # (1, C, H, W)
    c = x.shape[1]
    gx = torch.tensor(gaussian_kernel(kx, sigma), dtype=torch.float64, device=img.device)
    gy = torch.tensor(gaussian_kernel(ky, sigma), dtype=torch.float64, device=img.device)
    x = F.conv2d(F.pad(x, (kx // 2, kx // 2, 0, 0), mode="reflect"), gx.view(1, 1, 1, kx).expand(c, 1, 1, kx), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, ky // 2, ky // 2), mode="reflect"), gy.view(1, 1, ky, 1).expand(c, 1, ky, 1), groups=c)
    return torch.round(x[0].permute(1, 2, 0)).clamp(0, 255).to(torch.uint8)


def gaussian_blur_u8(img, ksize, sigma, device="cpu"):
    return gaussian_blur(torch.from_numpy(np.ascontiguousarray(img)).to(device), ksize, sigma).cpu().numpy()


def unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0, device="cpu"):
    if amount == 0:
        return img
    t = torch.from_numpy(np.ascontiguousarray(img)).to(device)
    blurred = gaussian_blur(t, kernel_size, sigma)
    sharpened = float(amount + 1) * t.double() - float(amount) * blurred.double()
    sharpened = torch.round(sharpened.clamp(0, 255)).to(torch.uint8)
    if threshold > 0:
        low_contrast = ((t.long() - blurred.long()) % 256) < threshold     # uint8 arithmetic, as in Deforum
        sharpened = torch.where(low_contrast, t, sharpened)
    return sharpened.cpu().numpy()
