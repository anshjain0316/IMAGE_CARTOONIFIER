from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
from typing import Literal, Tuple

# ------------- Utils

def _auto_white_balance_bgr(img: np.ndarray, clip: float = 0.0) -> np.ndarray:
    """Simple gray-world AWB with optional clipping (percent)."""
    b, g, r = cv2.split(img.astype(np.float32))
    mean_b, mean_g, mean_r = [np.mean(c) for c in (b, g, r)]
    mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
    b = np.clip(b * (mean_gray / mean_b), 0, 255)
    g = np.clip(g * (mean_gray / mean_g), 0, 255)
    r = np.clip(r * (mean_gray / mean_r), 0, 255)
    out = cv2.merge([b, g, r]).astype(np.uint8)
    if clip > 0:
        # clip extremes per-channel
        def _clip_channel(c):
            lo = np.percentile(c, clip)
            hi = np.percentile(c, 100 - clip)
            return np.clip((c - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255)
        b, g, r = [ _clip_channel(c.astype(np.float32)) for c in cv2.split(out) ]
        out = cv2.merge([b, g, r]).astype(np.uint8)
    return out

def _estimate_gamma(gray_0_1: np.ndarray) -> float:
    """Estimate gamma from midtone; returns gamma to brighten/darken."""
    mid = np.median(gray_0_1)
    mid = np.clip(mid, 1e-4, 1 - 1e-4)
    # If mid < 0.5, image is dark -> gamma < 1 to brighten (since we apply pow)
    # Solve mid_out = mid**gamma ~ 0.5 -> gamma = log(0.5)/log(mid)
    return float(np.log(0.5) / np.log(mid))

def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    lut = (np.linspace(0, 1, 256) ** gamma * 255.0).astype(np.uint8)
    return cv2.LUT(img, lut)

def _clahe_on_L(img: np.ndarray, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE on L channel (LAB) to gently add local contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def _resize_max_dim(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    scale = max_dim / float(m)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

# ------------- Edge maps

def edge_adapt_thresh(gray: np.ndarray, block_size: int = 9, C: int = 9, blur_ksize: int = 7) -> np.ndarray:
    g = cv2.medianBlur(gray, blur_ksize)
    edges = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    return edges

def edge_canny(gray: np.ndarray, low: int = 80, high: int = 160, sigma: float = 1.0) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (0, 0), sigma)
    e = cv2.Canny(g, low, high)
    # binarize to 255
    return (e > 0).astype(np.uint8) * 255

def edge_DoG(gray: np.ndarray, sigma1: float = 1.0, sigma2: float = 2.0, tau: float = 0.98) -> np.ndarray:
    """Difference-of-Gaussians -> xDoG-like binary edges."""
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
    dog = g1.astype(np.float32) - tau * g2.astype(np.float32)
    dog = (dog - dog.min()) / (dog.ptp() + 1e-6)
    e = (dog < 0.5).astype(np.uint8) * 255
    return e

def thicken_edges(edges: np.ndarray, k: int = 1) -> np.ndarray:
    """Optional dilation to make lines bolder."""
    if k <= 0: return edges
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)

# ------------- Color smoothing & quantization

def color_bilateral(img: np.ndarray, d: int = 9, sc: int = 200, ss: int = 200, iterations: int = 1) -> np.ndarray:
    out = img.copy()
    for _ in range(max(iterations, 1)):
        out = cv2.bilateralFilter(out, d, sc, ss)
    return out

def color_edge_preserving(img: np.ndarray, flags: int = 1, sigma_s: float = 60, sigma_r: float = 0.4) -> np.ndarray:
    return cv2.edgePreservingFilter(img, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)

def kmeans_quantize(img: np.ndarray, k: int = 8, attempts: int = 3) -> np.ndarray:
    Z = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)
    _compactness, labels, centers = cv2.kmeans(Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    q = centers[labels.flatten()].reshape(img.shape)
    return q

# ------------- Configs

EdgeMode = Literal["adaptive", "canny", "dog"]
StyleMode = Literal["classic", "quantized", "stylization", "pencil"]

@dataclass
class Preprocess:
    do_awb: bool = True               # auto white balance
    awb_clip: float = 0.0             # 0..5 typical
    do_gamma: bool = True
    gamma_strength: float = 0.85      # mix: 1.0 = use estimated gamma fully
    do_clahe: bool = False
    clahe_clip: float = 3.0
    clahe_tile: Tuple[int, int] = (8, 8)
    max_dim: int = 1600               # resize for speed/memory

@dataclass
class EdgeParams:
    mode: EdgeMode = "adaptive"
    # Adaptive
    block_size: int = 9
    C: int = 9
    blur_ksize: int = 7
    # Canny
    canny_low: int = 80
    canny_high: int = 160
    canny_sigma: float = 1.0
    # DoG
    dog_sigma1: float = 1.0
    dog_sigma2: float = 2.0
    dog_tau: float = 0.98
    # Post
    thicken: int = 1                  # 0 to skip

@dataclass
class ColorParams:
    # Smoothing
    use_edge_preserving: bool = False # else bilateral
    bilateral_d: int = 9
    bilateral_sc: int = 200
    bilateral_ss: int = 200
    bilateral_iter: int = 1
    ep_flags: int = 1
    ep_sigma_s: float = 60
    ep_sigma_r: float = 0.4
    # Quantization
    kmeans_k: int = 8

@dataclass
class CombineParams:
    invert_edges: bool = False        # set True if edges are white-on-black
    edge_opacity: float = 1.0         # 0..1 alpha of edges over color
    edge_color: Tuple[int, int, int] = (0, 0, 0)  # black ink

@dataclass
class CartoonConfig:
    preprocess: Preprocess = field(default_factory=Preprocess)
    edges: EdgeParams = field(default_factory=EdgeParams)
    color: ColorParams = field(default_factory=ColorParams)
    combine: CombineParams = field(default_factory=CombineParams)
    style: StyleMode = "classic"

# ------------- Pipeline steps

def _preprocess(img_bgr: np.ndarray, p: Preprocess) -> np.ndarray:
    out = _resize_max_dim(img_bgr, p.max_dim)
    if p.do_awb:
        out = _auto_white_balance_bgr(out, clip=p.awb_clip)
    if p.do_gamma:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        gamma_est = _estimate_gamma(gray.astype(np.float32) / 255.0)
        gamma = 1.0 * (1 - p.gamma_strength) + gamma_est * p.gamma_strength
        out = _apply_gamma(out, gamma)
    if p.do_clahe:
        out = _clahe_on_L(out, p.clahe_clip, p.clahe_tile)
    return out

def _edge_map(gray: np.ndarray, e: EdgeParams) -> np.ndarray:
    if e.mode == "adaptive":
        edges = edge_adapt_thresh(gray, e.block_size, e.C, e.blur_ksize)
    elif e.mode == "canny":
        edges = edge_canny(gray, e.canny_low, e.canny_high, e.canny_sigma)
    else:
        edges = edge_DoG(gray, e.dog_sigma1, e.dog_sigma2, e.dog_tau)
    if e.thicken > 0:
        edges = thicken_edges(edges, e.thicken)
    return edges

def _smooth_color(img: np.ndarray, c: ColorParams) -> np.ndarray:
    if c.use_edge_preserving:
        sm = color_edge_preserving(img, c.ep_flags, c.ep_sigma_s, c.ep_sigma_r)
    else:
        sm = color_bilateral(img, c.bilateral_d, c.bilateral_sc, c.bilateral_ss, c.bilateral_iter)
    return sm

def _combine(color_img: np.ndarray, edges_bin: np.ndarray, comb: CombineParams) -> np.ndarray:
    """Overlay edges (as ink) on color image with opacity."""
    if comb.invert_edges:
        edges_bin = 255 - edges_bin
    # Ensure edges are black lines on white bg
    edges03 = edges_bin / 255.0
    ink = np.full_like(color_img, 255, dtype=np.uint8)
    ink[:] = (255, 255, 255)
    # Where edges03 == 0 (black), paint with edge_color; where 1 (white), leave white
    edge_layer = ink.copy()
    mask = (edges03 < 0.5).astype(np.uint8) * 255
    edge_layer[mask > 0] = comb.edge_color
    out = cv2.addWeighted(color_img, 1.0, edge_layer, comb.edge_opacity, 0)
    return out

# ------------- Public API

def cartoonify(
    img_bgr: np.ndarray,
    config: CartoonConfig | None = None
) -> np.ndarray:
    """
    Main entry point. Provide a BGR image (cv2.imread).
    Returns a BGR cartoonified image.
    """
    if config is None:
        config = CartoonConfig()

    # Preprocess
    base = _preprocess(img_bgr, config.preprocess)

    # Mode-specific shortcuts
    if config.style == "stylization":
        # OpenCV's built-in stylization gives a painterly look
        # You can tweak sigma_s (10-200) and sigma_r (0-1)
        return cv2.stylization(base, sigma_s=75, sigma_r=0.25)

    if config.style == "pencil":
        # Returns (gray, color) sketches; we keep the color one
        gray_sketch, color_sketch = cv2.pencilSketch(base, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        return color_sketch

    # For classic/quantized:
    # Edges
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    edges = _edge_map(gray, config.edges)

    # Color base
    color = _smooth_color(base, config.color)

    if config.style == "quantized":
        color = kmeans_quantize(color, k=config.color.kmeans_k)

    # Combine
    out = _combine(color, edges, config.combine)
    return out


# ------------- Presets (handy defaults)

def preset_classic() -> CartoonConfig:
    return CartoonConfig(
        style="classic",
        preprocess=Preprocess(do_awb=True, do_gamma=True, gamma_strength=0.9, do_clahe=False, max_dim=1400),
        edges=EdgeParams(mode="adaptive", block_size=9, C=9, blur_ksize=7, thicken=1),
        color=ColorParams(use_edge_preserving=False, bilateral_d=9, bilateral_sc=175, bilateral_ss=175, bilateral_iter=2),
        combine=CombineParams(invert_edges=False, edge_opacity=1.0, edge_color=(0, 0, 0))
    )

def preset_clean_quantized() -> CartoonConfig:
    return CartoonConfig(
        style="quantized",
        preprocess=Preprocess(do_awb=True, do_gamma=True, gamma_strength=0.9, do_clahe=False, max_dim=1400),
        edges=EdgeParams(mode="canny", canny_low=60, canny_high=140, canny_sigma=1.2, thicken=1),
        color=ColorParams(use_edge_preserving=True, ep_sigma_s=80, ep_sigma_r=0.3, kmeans_k=8),
        combine=CombineParams(invert_edges=False, edge_opacity=0.9, edge_color=(0, 0, 0))
    )

def preset_ink_dog() -> CartoonConfig:
    return CartoonConfig(
        style="classic",
        preprocess=Preprocess(do_awb=True, do_gamma=True, gamma_strength=0.8, do_clahe=True, clahe_clip=2.0),
        edges=EdgeParams(mode="dog", dog_sigma1=0.8, dog_sigma2=1.6, dog_tau=0.98, thicken=2),
        color=ColorParams(use_edge_preserving=False, bilateral_d=7, bilateral_sc=150, bilateral_ss=150, bilateral_iter=1),
        combine=CombineParams(invert_edges=False, edge_opacity=1.0, edge_color=(10, 10, 10))
    )

def preset_stylization() -> CartoonConfig:
    return CartoonConfig(style="stylization")

def preset_pencil() -> CartoonConfig:
    return CartoonConfig(style="pencil")
