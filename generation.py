import numpy as np
import noise
import cv2
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.ndimage import (
    gaussian_filter,
    label,
    map_coordinates,
    distance_transform_edt,
    zoom,
    convolve,
)
from scipy.signal import fftconvolve
from scipy.fft import fft2, ifft2, fftshift
from scipy.interpolate import PchipInterpolator
import numba
from asteval import Interpreter as AstevalInterpreter # Use safe interpreter

from constants import (
    STEFAN_BOLTZMANN,
    KELVIN_TO_CELSIUS_OFFSET,
    cc_rainfall_multiplier,
    SEA_LEVEL,
)

try:
    from opensimplex import OpenSimplex
except ModuleNotFoundError:
    OpenSimplex = None
# Hypsometric target curves in metres
EARTHLIKE_OCEAN_CDF_METERS = np.array(
    [
        (0.000, -11000),
        (0.02, -8000),  # only the lowest 2 % reach abyssal depth
        (0.05, -6000),
        (0.15, -4500),
        (0.40, -3800),
        (0.60, -3000),
        (0.68, -2500),
        (0.70, -600),
        (0.73, -130),
    ],
    dtype=np.float32,
)
EARTHLIKE_LAND_CDF_METERS = np.array(
    [
        (0.00, 0),
        (0.10, 40),
        (0.33, 300),
        (0.74, 840),
        (0.97, 2000),
        (0.997, 4500),
        (1.000, 8848),
    ],
    dtype=np.float32,
)

# Fraction of the remapped value to use; remainder retains raw ranking
ADHERENCE = 0.85

# --- Helper functions specific to generation ---
# ---------- OpenSimplex (or Perlin) helpers: paste this block ----------
import numpy as np

# 1.  Robust 2-D noise factory ------------------------------------------------
def _make_noise2(seed: int):
    """
    Return a vectorised function f(x,y)->[-1,1] using OpenSimplex if the
    package is installed, otherwise Perlin’s snoise2 from the ‘noise’ lib.
    Works with both OpenSimplex 0.3 (noise2) and 0.4+ (noise2d) APIs.
    """
    try:
        from opensimplex import OpenSimplex
        gen = OpenSimplex(seed)
        # pick whichever name this library version provides
        base_fn = getattr(gen, "noise2", None) or (lambda x, y, _g=gen: _g.noise2d(x, y))
    except (ModuleNotFoundError, AttributeError):
        # Fallback to Perlin simplex (still quite good)
        from noise import snoise2
        base_fn = lambda x, y, _b=seed: snoise2(x, y, octaves=1, base=_b)

    # Vectorise so we can feed NumPy arrays directly
    return np.vectorize(base_fn, otypes=[np.float32])

# 2.  Low-frequency “continent” field ----------------------------------------
def generate_simplex_continents(w: int,
                                h: int,
                                scale_km: float = 3000.0,
                                seed: int = 0) -> np.ndarray:
    """
    Produce a broad OpenSimplex/Perlin mask with wavelengths on the order
    of 'scale_km'.  Output is float32 in the range [0,1].
    """
    freq = 1.0 / scale_km                   # km⁻¹  → px⁻¹  (≈1 px = 1 km)
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    noise2 = _make_noise2(seed)
    raw   = noise2(X * freq, Y * freq)      # [-1,1]
    return normalize_map(raw)               # [0,1]

# 3.  Small-scale Perlin helper ---------------------------------------------
def _perlin(shape, freq=0.02, warp=0.0, seed=0):
    """Return a Perlin/OpenSimplex field in [-1,1] with optional warping."""
    h, w = shape
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    noise2 = _make_noise2(seed)
    if warp != 0.0:
        warp_noise = _make_noise2(seed + 1)
        X = X + warp * w * warp_noise(X * freq, Y * freq)
        Y = Y + warp * h * warp_noise((X + 11.3) * freq, (Y - 7.9) * freq)

    return noise2(X * freq, Y * freq)
# ---------- end of helper block --------------------------------------------

def hex_to_rgb(h):
    """Converts a hex color string to an (r, g, b) tuple."""
    return tuple(int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

def normalize_map(d):
    """Normalizes a NumPy array to the 0.0 - 1.0 range."""
    m, M = np.min(d), np.max(d)
    return (d - m) / (M - m) if M > m else np.zeros_like(d)

# --- Vectorized Wrappers for the 'noise' library ---
vec_pnoise2 = np.vectorize(noise.pnoise2)

def radial_taper(shape, knee_px, slope=6):
    """Return a smooth radial mask that tapers near the Nyquist frequency."""
    H, W = shape
    kx = np.fft.fftfreq(W)[:, None]
    ky = np.fft.fftfreq(H)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    return 1.0 / (1.0 + np.exp(slope * (k - 1.0 / knee_px)))

def logistic(x, mid, spread):
    """Basic logistic function used for smooth blending."""
    return 1.0 / (1.0 + np.exp(-(x - mid) / spread))

def exp_kernel(radius_px, L_px):
    """Return a normalized exponential-decay kernel."""
    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    r = np.hypot(x, y)
    k = np.exp(-r / float(L_px))
    return k / k.sum()


def convolve_wrap_fft(img, kernel):
    """Convolve using FFT assuming periodic (wrap) boundary."""
    img = img.astype(np.float32, copy=False)
    kernel = kernel.astype(np.float32, copy=False)
    return fftconvolve(img, kernel, mode="same").astype(np.float32)

def coastal_taper(h, dist, sea=0.4, h_interior=0.5, width_km=150.0):
    """Feather elevations near the coast toward sea level."""
    w = logistic(dist, width_km * 0.5, 0.15 * width_km)
    return sea + (1.0 - w) * np.clip(h, sea, h_interior) + w * h

def meters_to_norm(h_m, sea_level, elevation_range_m, min_depth_m=-11000.0):
    """Convert a physical elevation to normalized units using fixed land and ocean ranges."""
    land_max = elevation_range_m - abs(min_depth_m)
    ocean_depth = abs(min_depth_m)
    if h_m >= 0:
        return sea_level + (h_m / land_max) * (1.0 - sea_level)
    else:
        return sea_level + (h_m / ocean_depth) * sea_level

def build_earthlike_curves(sea_level, elevation_range_m):
    """Return ocean/land CDF curves normalised to 0..1 using the elevation range."""
    vfunc = np.vectorize(meters_to_norm)
    ocean = EARTHLIKE_OCEAN_CDF_METERS.copy()
    land = EARTHLIKE_LAND_CDF_METERS.copy()
    ocean[:, 1] = vfunc(ocean[:, 1], sea_level, elevation_range_m)
    land[:, 1] = vfunc(land[:, 1], sea_level, elevation_range_m)
    return ocean, land

def dynamic_sea_level(raw, ocean_fraction=SEA_LEVEL):
    """Return the normalized elevation that yields the requested ocean fraction."""
    return np.percentile(raw, 100.0 * ocean_fraction)


def advanced_cdf_split(hmap, sea_level, elevation_range_m):
    """Split heightmap into land and ocean and remap using Earth-like curves.

    The ocean portion is scaled so that 1 corresponds to the water surface and
    0 to the deepest abyss.  The lowest 2% of ocean values are further
    compressed into the range ``[0, 0.1]`` while the remaining ocean values are
    mapped to ``[0.1, 1]``.  Land values are scaled relative to sea level and
    split into four ranges:

    - ``land_low``: bottom 7% → ``[0.005, 0.0995]``
    - ``land_median``: next portion up to the 50% percentile (excluding low)
      → ``[0.1, 0.74]``
    - ``land_upper``: between the 50% and 97% percentiles → ``[0.74, 0.97]``
    - ``land_peaks``: top 3% → ``[0.97, 1]``
    """

    dyn_level = dynamic_sea_level(hmap, ocean_fraction=sea_level)
    land_mask = hmap > dyn_level
    ocean_mask = ~land_mask

    land_map = np.zeros_like(hmap, dtype=np.float32)
    ocean_map = np.ones_like(hmap, dtype=np.float32)
    diag_presplit_land = np.zeros_like(hmap, dtype=np.float32)
    diag_presplit_ocean = np.zeros_like(hmap, dtype=np.float32)
    land_low_scaled = np.zeros_like(hmap, dtype=np.float32)
    land_median_scaled = np.zeros_like(hmap, dtype=np.float32)
    land_upper_scaled = np.zeros_like(hmap, dtype=np.float32)
    land_peaks_scaled = np.zeros_like(hmap, dtype=np.float32)
    ocean_abyss_scaled = np.zeros_like(hmap, dtype=np.float32)
    ocean_floor_scaled = np.zeros_like(hmap, dtype=np.float32)
    coastline = np.zeros_like(hmap, dtype=np.float32)

    ocean_vals = hmap[ocean_mask]
    ocean_curve, land_curve = build_earthlike_curves(sea_level, elevation_range_m)

    result = np.zeros_like(hmap, dtype=np.float32)

    if ocean_vals.size:
        mapped_ocean_vals = partial_remap(ocean_map[ocean_mask].ravel(), ocean_curve)
        mapped_ocean = np.zeros_like(hmap, dtype=np.float32)
        mapped_ocean[ocean_mask] = mapped_ocean_vals.reshape(ocean_map[ocean_mask].shape)
        coast_mask = mapped_ocean == 1.0
        coastline[coast_mask] = 0.01
        result[ocean_mask & ~coast_mask] = mapped_ocean[ocean_mask & ~coast_mask]
    else:
        coast_mask = np.zeros_like(hmap, dtype=bool)

    land_mask = land_mask | coast_mask
    ocean_mask = ~land_mask
    ocean_vals = hmap[ocean_mask]

    land_vals = hmap[land_mask]

    if land_vals.size:
        land_rel = land_vals - dyn_level
        max_land = float(land_rel.max()) if land_rel.max() > 0 else 1.0
        base_norm = land_rel / max_land
        diag_presplit_land[land_mask] = base_norm
        diag_presplit_land[coast_mask] = 0.01
        max_v = diag_presplit_land[land_mask | coast_mask].max()
        if max_v > 0:
            diag_presplit_land[land_mask | coast_mask] /= max_v
        land_norm = diag_presplit_land[land_mask]

        low_cut = np.percentile(land_norm, 7.0)
        median_cut = np.percentile(land_norm, 50.0)
        peak_cut = np.percentile(land_norm, 97.0)
        low_mask_local = land_norm <= low_cut
        peak_mask_local = land_norm >= peak_cut
        median_mask_local = (land_norm <= median_cut) & ~low_mask_local
        upper_mask_local = ~(low_mask_local | median_mask_local | peak_mask_local)

        land_scaled = np.zeros_like(land_norm, dtype=np.float32)
        local_low = np.zeros_like(land_norm, dtype=np.float32)
        local_median = np.zeros_like(land_norm, dtype=np.float32)
        local_upper = np.zeros_like(land_norm, dtype=np.float32)
        local_peaks = np.zeros_like(land_norm, dtype=np.float32)

        if low_mask_local.any():
            vals = land_norm[low_mask_local]
            rng = float(vals.max() - vals.min())
            if rng > 0:
                scaled = ((vals - vals.min()) / rng) * 0.0995 + 0.005
            else:
                scaled = np.full_like(vals, 0.005)
            land_scaled[low_mask_local] = scaled
            tmp = np.zeros_like(land_norm)
            tmp[low_mask_local] = scaled
            local_low = tmp

        if median_mask_local.any():
            vals = land_norm[median_mask_local]
            rng = float(vals.max() - vals.min())
            if rng > 0:
                scaled = ((vals - vals.min()) / rng) * 0.64 + 0.1
            else:
                scaled = np.full_like(vals, 0.1)
            land_scaled[median_mask_local] = scaled
            tmp = np.zeros_like(land_norm)
            tmp[median_mask_local] = scaled
            local_median = tmp

        if upper_mask_local.any():
            vals = land_norm[upper_mask_local]
            rng = float(vals.max() - vals.min())
            if rng > 0:
                scaled = ((vals - vals.min()) / rng) * 0.23 + 0.74
            else:
                scaled = np.full_like(vals, 0.74)
            land_scaled[upper_mask_local] = scaled
            tmp = np.zeros_like(land_norm)
            tmp[upper_mask_local] = scaled
            local_upper = tmp

        if peak_mask_local.any():
            vals = land_norm[peak_mask_local]
            rng = float(vals.max() - vals.min())
            if rng > 0:
                scaled = ((vals - vals.min()) / rng) * 0.03 + 0.97
            else:
                scaled = np.full_like(vals, 1.0)
            land_scaled[peak_mask_local] = scaled
            tmp = np.zeros_like(land_norm)
            tmp[peak_mask_local] = scaled
            local_peaks = tmp

        land_map[land_mask] = land_scaled.astype(np.float32)
        land_low_scaled[land_mask] = local_low
        land_median_scaled[land_mask] = local_median
        land_upper_scaled[land_mask] = local_upper
        land_peaks_scaled[land_mask] = local_peaks

    ocean_curve, land_curve = build_earthlike_curves(sea_level, elevation_range_m)

    if land_vals.size:
        mapped_land = partial_remap(land_map[land_mask].ravel(), land_curve)
        result[land_mask] = mapped_land.reshape(land_map[land_mask].shape)

    # Smooth land and ocean separately to remove jitter without blending
    if land_vals.size:
        lm = land_mask.astype(np.float32)
        sm = gaussian_filter(result * lm, sigma=0.3)
        norm = gaussian_filter(lm, sigma=0.3)
        result[land_mask] = np.divide(sm, norm, out=np.zeros_like(sm), where=norm>0)[land_mask]
    if ocean_vals.size:
        om = ocean_mask.astype(np.float32)
        sm = gaussian_filter(result * om, sigma=0.3)
        norm = gaussian_filter(om, sigma=0.3)
        result[ocean_mask] = np.divide(sm, norm, out=np.zeros_like(sm), where=norm>0)[ocean_mask]

    result = np.clip(result, 0.0, 1.0)


    diagnostics = {
        "earthlike_land_cdf": land_map,
        "earthlike_ocean_cdf": ocean_map,
        "land_presplit": diag_presplit_land,
        "ocean_presplit": diag_presplit_ocean,
        "land_low_scaled": land_low_scaled,
        "land_median_scaled": land_median_scaled,
        "land_upper_scaled": land_upper_scaled,
        "land_peaks_scaled": land_peaks_scaled,
        "ocean_abyss_scaled": ocean_abyss_scaled if ocean_vals.size else np.zeros_like(hmap, dtype=np.float32),
        "ocean_floor_scaled": ocean_floor_scaled if ocean_vals.size else np.zeros_like(hmap, dtype=np.float32),
    }
    return result, land_mask, diagnostics


def _build_earthlike_cdf(sea_level=SEA_LEVEL, elevation_range_m=19848.0):
    """Create a single Earth-like CDF for hypsometric remapping."""
    ocean, land = build_earthlike_curves(sea_level, elevation_range_m)
    x_ocean = sea_level * ocean[:, 0]
    x_land = sea_level + (1.0 - sea_level) * land[:, 0]
    cdf = np.vstack([
        np.column_stack((x_ocean, ocean[:, 1])),
        np.column_stack((x_land, land[:, 1])),
    ])
    return cdf


EARTHLIKE_CDF = _build_earthlike_cdf()

def partial_remap(values, target, adherence=ADHERENCE):
    """Map values to a target CDF while retaining some native variance."""
    sorter = np.argsort(values)
    ranks = np.empty_like(sorter, dtype=np.float32)
    ranks[sorter] = np.linspace(0.0, 1.0, len(sorter))
    interp = PchipInterpolator(target[:, 0], target[:, 1], extrapolate=True)
    earthlike = interp(ranks)
    native_sorted = np.sort(values)
    blended = adherence * earthlike + (1.0 - adherence) * native_sorted
    out = np.empty_like(values, dtype=np.float32)
    out[sorter] = blended
    return out

# --- Heightmap shaping based on tectonic distance and Earth-like statistics ---
def earthlike(
    height,
    plate_mask,
    world_diameter_km,
    cdf_factor=1.0,
    spectral_slope=-2.0,
    sigma_edge=1.0,
    sigma_core=12.0,
    blend_km=250.0,
    oversample=1.5,
    target_cdf=None,
    apply_cdf_remap=True,
    apply_blend=True,
    apply_spectral_shape=False,
    taper_knee_px=4,
    micro_relief_amp=0.00,
    sea_level=SEA_LEVEL,
    seed=0,
    elevation_range_m=19848.0,
):
    """Return a smoothed, Earth-like heightmap in the range [0,1].

    Set ``apply_cdf_remap`` to ``False`` to skip the Earth-like cumulative
    distribution remapping step. ``apply_blend`` and ``apply_spectral_shape``
    control whether the distance-weighted Gaussian blend and subsequent
    spectral shaping are applied.
    """

    # ------------------------------------------------------------------ 1 — Optional up-sampling
    h, w = height.shape
    if oversample > 1.01:
        hi_w, hi_h = int(w * oversample), int(h * oversample)
        hi_height = cv2.resize(height, (hi_w, hi_h), interpolation=cv2.INTER_CUBIC)
        hi_mask = cv2.resize(plate_mask.astype(np.float32), (hi_w, hi_h), interpolation=cv2.INTER_NEAREST)
    else:
        hi_height = height
        hi_mask = plate_mask.astype(np.float32)

    if apply_cdf_remap and target_cdf is None:
        target_cdf = _build_earthlike_cdf(sea_level=sea_level, elevation_range_m=elevation_range_m)

    # ------------------------------------------------------------------ 2 — Distance-weighted Gaussian blend from plate centres
    km_per_px = world_diameter_km / (w * oversample) if w else 1.0
    if apply_blend:
        dist_px = distance_transform_edt(1.0 - hi_mask)
        mid, spread = blend_km * 0.5, max(1.0, blend_km / 10.0)
        w_blend = 1.0 / (1.0 + np.exp(-(dist_px * km_per_px - mid) / spread))
        h_edge = gaussian_filter(hi_height, sigma_edge / km_per_px)
        h_core = gaussian_filter(hi_height, sigma_core / km_per_px)
        blurred = (1.0 - w_blend) * h_edge + w_blend * h_core
    else:
        blurred = hi_height

    # ------------------------------------------------------------------ 3 — Spectral shaping with soft radial taper
    if apply_spectral_shape:
        H2, W2 = blurred.shape
        kx = np.fft.fftfreq(W2)[None, :]
        ky = np.fft.fftfreq(H2)[:, None]
        k = np.sqrt(kx ** 2 + ky ** 2) + 1e-9
        spec = np.fft.fftshift(fft2(blurred))
        spec *= k ** (spectral_slope / 2.0) * radial_taper((H2, W2), knee_px=taper_knee_px)
        shaped = np.fft.ifft2(np.fft.ifftshift(spec)).real
    else:
        shaped = blurred

    # ------------------------------------------------------------------ 4 — Down-sample to the requested resolution
    if oversample > 1.01:
        shaped = cv2.resize(shaped, (w, h), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------ 5 — Placeholder for CDF remap later
    diagnostics = {}
    land_mask = shaped > sea_level

    # ------------------------------------------------------------------ 6 — Decide what is land after the remap
    # land_mask already determined above

    # -----
    lp_sigma_km = 90.0
    sigma_lp_px = max(1.0, (lp_sigma_km / world_diameter_km) * w)
    lowpass = gaussian_filter(shaped, sigma_lp_px)
    hires = shaped - lowpass

    amp = micro_relief_amp * (np.percentile(shaped, 95) - np.percentile(shaped, 5))
    micro = amp * _perlin(shaped.shape, freq=0.02, warp=0.005, seed=seed)

    shaped += (hires + micro) * land_mask

    # ------------------------------------------------------------------ 8 — Re-align the sea-level percentile and feather the coastline
    delta = sea_level - np.percentile(shaped, 100.0 * sea_level)
    shaped += delta

    coast_px = distance_transform_edt(shaped > sea_level)
    coast_km = coast_px * (world_diameter_km / w)
    shaped = np.where(
        shaped > sea_level,
        coastal_taper(shaped, coast_km, sea=sea_level, h_interior=min(sea_level + 0.12, 1.0)),
        shaped,
    )


    # ------------------------------------------------------------------ 8b — keep extremes but squeeze them back into 0‒1
    lo, hi = np.percentile(shaped, (0.05, 99.95))
    spread = max(1e-6, hi - lo)
    shaped = (shaped - lo) / spread
    # ------------------------------------------------------------------ 9 — Final low-key Nyquist roll-off
    spec = fft2(shaped)
    spec *= radial_taper(shaped.shape, knee_px=taper_knee_px)
    shaped = np.real(ifft2(spec))

    # ----------------------------------------------------------------- 9b — Apply CDF remapping after Nyquist roll-off
    if apply_cdf_remap and target_cdf is not None:
        shaped, land_mask, diag_maps = advanced_cdf_split(shaped, sea_level, elevation_range_m)
        diagnostics.update(diag_maps)
    else:
        land_mask = shaped > sea_level

    # ----------------------------------------------------------------- 10 — Return normalised, clipped result
    shaped = 0.5 + cdf_factor * (shaped - 0.5)
    lo, hi = np.percentile(shaped, (0.05, 99.95))
    shaped = (shaped - lo) / (hi - lo + 1e-9)
    delta = sea_level - np.percentile(shaped, 100.0 * sea_level)
    shaped += delta
    return np.clip(shaped, 0.0, 1.0), diagnostics

# --- Generation Algorithms ---
def generate_perlin_noise(w, h, scale, octaves, seed):
    x, y = np.arange(w), np.arange(h)
    xv, yv = np.meshgrid(x, y)
    arr = vec_pnoise2(xv / scale, yv / scale, octaves=octaves, persistence=0.5, lacunarity=2.0, base=seed)
    return normalize_map(arr)

def apply_rainfall_erosion(hmap, params, scaling_manager):
    """Erode terrain proportionally to blurred sea moisture before remapping."""
    world_diameter_km = params.get("world_diameter_km", 12000)
    sea_fraction = params.get("sea_level", SEA_LEVEL)
    sea_level_norm = np.percentile(hmap, 100.0 * sea_fraction)

    sea_mask = (hmap <= sea_level_norm).astype(np.float32)
    sigma_px = (400.0 / world_diameter_km) * hmap.shape[1]
    sigma_px = max(0.1, sigma_px)
    blurred = gaussian_filter(sea_mask, sigma=sigma_px, mode="wrap")
    max_int = blurred.max() if blurred.max() > 0 else 1.0
    depth_norm = scaling_manager.to_normalized(params.get("max_erosion_depth_m", 250.0), above_sea=True)
    erosion = depth_norm * (blurred / max_int)
    return np.clip(hmap - erosion, 0.0, 1.0)

def generate_flow_field_landmass(w, h, scale, ws, octaves, seed):
    fs = scale * 2.0
    x, y = np.arange(w), np.arange(h)
    xv, yv = np.meshgrid(x, y)
    flow_x = vec_pnoise2(xv / fs, yv / fs, octaves=4, base=seed + 555) * ws
    flow_y = vec_pnoise2(xv / fs, yv / fs, octaves=4, base=seed + 777) * ws
    arr = vec_pnoise2((xv + flow_x) / scale, (yv + flow_y) / scale, octaves=octaves, base=seed)
    return normalize_map(arr)

def generate_ridged_multifractal_noise(
    w,
    h,
    scale,
    octaves,
    seed,
    freq_map=None,
    warp_scale=None,
    warp_amp=0.0,
    min_freq_mult=0.01,
):
    """Ridged multifractal noise with optional domain warping and per-pixel frequency."""

    x, y = np.arange(w), np.arange(h)
    xv, yv = np.meshgrid(x, y)

    if warp_scale is not None and warp_amp != 0.0:
        wf = 1.0 / warp_scale
        warp_noise = _make_noise2(seed + 99)
        xv = xv + warp_amp * w * warp_noise(xv * wf, yv * wf)
        yv = yv + warp_amp * h * warp_noise((xv + 11.3) * wf, (yv - 7.9) * wf)

    xv_scaled = xv / scale
    yv_scaled = yv / scale

    if freq_map is not None:
        freq_mult = np.clip(freq_map, min_freq_mult, None)
        xv_scaled = xv_scaled * freq_mult
        yv_scaled = yv_scaled * freq_mult

    raw_noise = vec_pnoise2(xv_scaled, yv_scaled, octaves=octaves, base=seed)
    arr = (1.0 - np.abs(raw_noise)) ** 2
    return normalize_map(arr)

def generate_tectonic_map(w, h, num_plates, seed, lloyd_steps=2):
    """Voronoi plates grown by successive fission.

    Parameters
    ----------
    w, h : int
        Output dimensions.
    num_plates : int
        Initial number of major plates. The algorithm will continue fission
        until a total of 17 plates is produced.
    seed : int
        Random seed.
    lloyd_steps : int, optional
        Number of Lloyd relaxation steps for plate centres.
    """

    rng = np.random.default_rng(seed)

    # 1. start from the requested number of majors, Poisson-disc to avoid crowding
    points = rng.random((num_plates, 2)) * [w, h]
    min_d2 = (0.18 * min(w, h)) ** 2
    keep = []
    for p in points:
        if all(((p - q) ** 2).sum() > min_d2 for q in keep):
            keep.append(p)
    points = np.array(keep)

    target_plate_count = 17

    # 2. grow to target count by fission (multiple splits per iteration)
    while len(points) < target_plate_count:
        tree = cKDTree(points)
        yi, xi = np.indices((h, w))
        px = np.stack((xi.ravel(), yi.ravel()), axis=-1)
        owner = tree.query(px, k=1)[1].reshape(h, w)
        areas = np.bincount(owner.ravel(), minlength=len(points))
        probs = areas / areas.sum()
        parent = rng.choice(len(points), p=probs)

        mask = owner == parent
        ys, xs = np.where(mask)
        splits = min(target_plate_count - len(points), rng.integers(1, 4))
        for _ in range(splits):
            k = rng.integers(0, len(xs))
            new_pt = np.array([xs[k], ys[k]])
            points = np.vstack([points, new_pt])
            if len(points) >= target_plate_count:
                break

    # 3. a couple of Lloyd relaxations
    for _ in range(lloyd_steps):
        tree = cKDTree(points)
        yi, xi = np.indices((h, w))
        px = np.stack((xi.ravel(), yi.ravel()), axis=-1)
        owner = tree.query(px, k=1)[1]
        sums = np.zeros_like(points)
        counts = np.zeros(len(points))
        np.add.at(sums, owner, px)
        np.add.at(counts, owner, 1)
        points = sums / counts[:, None]

    # 4. assign random velocity to each plate
    velocities = rng.random((len(points), 2)) * 2 - 1

    # 5. build tectonic potential & boundary mask
    tree = cKDTree(points)
    neighbor_indices = tree.query(points, k=min(len(points), 5))[1]
    plate_potentials = np.zeros(len(points))
    for i in range(len(points)):
        for j in neighbor_indices[i, 1:]:
            v_rel = velocities[j] - velocities[i]
            norm = np.linalg.norm(points[j] - points[i])
            if norm > 1e-9:
                boundary_normal = (points[j] - points[i]) / norm
                plate_potentials[i] -= np.dot(v_rel, boundary_normal)

    yi, xi = np.indices((h, w))
    pixel_coords = np.stack((xi.ravel(), yi.ravel()), axis=-1)
    dist, indices = tree.query(pixel_coords, k=2)
    d1, d2 = dist[:, 0], dist[:, 1]
    p1, p2 = plate_potentials[indices[:, 0]], plate_potentials[indices[:, 1]]
    eps = 1e-6
    w1, w2 = 1 / (d1**2 + eps), 1 / (d2**2 + eps)
    tectonic_map = (p1 * w1 + p2 * w2) / (w1 + w2)

    nearest = indices[:, 0].reshape(h, w)
    plate_mask = np.zeros((h, w), dtype=np.uint8)
    plate_mask[1:, :] |= nearest[1:, :] != nearest[:-1, :]
    plate_mask[:, 1:] |= nearest[:, 1:] != nearest[:, :-1]
    plate_mask[:-1, :] |= nearest[:-1, :] != nearest[1:, :]
    plate_mask[:, :-1] |= nearest[:, :-1] != nearest[:, 1:]

    return normalize_map(tectonic_map.reshape(h, w)), points, velocities, plate_mask


def calculate_flow_field(hmap, params):
    h, w = hmap.shape
    dy, dx = np.gradient(hmap)
    f_scale, f_octaves, seed = (params["flow_scale"], params["flow_octaves"], params["seed"])
    x, y = np.arange(w), np.arange(h)
    xv, yv = np.meshgrid(x, y)
    base_wind_angle = vec_pnoise2(xv * f_scale, yv * f_scale, octaves=f_octaves, base=seed + 100) * 2 * np.pi
    steepness = np.clip(np.sqrt(dx**2 + dy**2) * 5.0, 0, 1)
    contour_angle = np.arctan2(-dx, dy) # Angle perpendicular to the gradient
    vx_wind, vy_wind = np.cos(base_wind_angle), np.sin(base_wind_angle)
    vx_contour, vy_contour = np.cos(contour_angle), np.sin(contour_angle)
    # Blend wind and contour flow - more steepness means more contour-aligned flow
    final_vx = vx_wind * (1 - steepness) + vx_contour * steepness
    final_vy = vy_wind * (1 - steepness) + vy_contour * steepness
    # Use both vector components when computing the final angle
    flow_field_angles = np.arctan2(final_vy, final_vx)
    return flow_field_angles # This function should still exist

# --- Update the Numba function to handle spawning and clipping ---
@numba.njit(fastmath=True)
def _run_river_simulation_numba(hmap, sea_level_norm, flow_field_angles, temp_c, spawn_cdf,
                                particle_count, max_steps, fade, simulation_duration, step_length):
    """Run the river particle simulation with temperature-aware spawning."""
    h, w = hmap.shape
    image_data = np.zeros((h, w), dtype=np.float32)

    def sample_spawn():
        idx = np.searchsorted(spawn_cdf, np.random.random())
        iy = idx // w
        ix = idx - iy * w
        return float(ix), float(iy)

    particles_xy = np.zeros((particle_count, 2), dtype=np.float32)
    particle_steps = np.zeros(particle_count, dtype=np.int32)

    for i in range(particle_count):
        while True:
            sx, sy = sample_spawn()
            ix, iy = int(sx), int(sy)
            if 0 <= ix < w and 0 <= iy < h and hmap[iy, ix] > sea_level_norm and temp_c[iy, ix] > 0:
                particles_xy[i, 0], particles_xy[i, 1] = sx, sy
                break

    # Main simulation loop
    for _ in range(simulation_duration):
        image_data *= fade # Fade the image from the previous iteration

        for i in range(particle_count):
            # Check if particle has exceeded max steps OR terminated (e.g. hit sea)
            if particle_steps[i] >= max_steps:
                # Re-spawn using the probability map
                while True:
                    sx, sy = sample_spawn()
                    nix, niy = int(sx), int(sy)
                    if 0 <= nix < w and 0 <= niy < h and hmap[niy, nix] > sea_level_norm and temp_c[niy, nix] > 0:
                        particles_xy[i, 0], particles_xy[i, 1] = sx, sy
                        particle_steps[i] = 0
                        break
                # If re-spawned, skip the movement for this iteration
                continue

            # Get current position
            px, py = particles_xy[i, 0], particles_xy[i, 1]
            ix, iy = int(px), int(py)

            # Check bounds for current position (shouldn't happen if clipping works, but defensive)
            if not (0 <= ix < w and 0 <= iy < h):
                particle_steps[i] = max_steps # Terminate if somehow out of bounds
                continue # Move to next particle

            # Get flow angle at current integer position
            angle = flow_field_angles[iy, ix]

            # Calculate next position
            nx, ny = px + np.cos(angle) * step_length, py + np.sin(angle) * step_length
            nix, niy = int(nx), int(ny)

            # --- Clipping Logic: Terminate if next step is into the sea ---
            # Check bounds for next position first
            if 0 <= nix < w and 0 <= niy < h:
                 if hmap[niy, nix] <= sea_level_norm:
                      particle_steps[i] = max_steps # Mark particle as finished
                      # Do *not* update position or deposit if terminated by sea
                      continue # Move to next particle
            else:
                 # If the next step is out of bounds, also terminate
                 particle_steps[i] = max_steps
                 continue # Move to next particle


            # If not terminated, perform deposition and update position
            # Use Bresenham-like line drawing for deposition between current and next point
            x0, y0 = ix, iy # Use integer indices for start of line segment
            x1, y1 = nix, niy # Use integer indices for end of line segment

            dx_line = abs(x1 - x0)
            dy_line = -abs(y1 - y0) # Negative because y increases downwards in image coords

            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1

            err = dx_line + dy_line # Error variable

            while True:
                # Deposit at the current cell in the line drawing algorithm
                # Check bounds before depositing
                if 0 <= y0 < h and 0 <= x0 < w and temp_c[y0, x0] > 0:
                     image_data[y0, x0] += 0.05

                # Check if we reached the end point of the line segment
                if x0 == x1 and y0 == y1:
                    break

                e2 = 2 * err # Look ahead error

                # Move x
                if e2 >= dy_line:
                    err += dy_line
                    x0 += sx

                # Move y
                if e2 <= dx_line:
                    err += dx_line
                    y0 += sy

            # Update the particle's continuous position
            particles_xy[i, 0], particles_xy[i, 1] = nx, ny

            # Increment step count
            particle_steps[i] += 1


    return image_data # Return the accumulated deposition map
# --- End of updated Numba function ---

def generate_flow_field_rivers(hmap, params):
    h, w = hmap.shape
    sea_level = params.get('sea_level', SEA_LEVEL)
    flow_field_angles = calculate_flow_field(hmap, params)
    # Correct parameter order for the numba simulation helper
    temp_dummy = np.full_like(hmap, 20.0, dtype=np.float32)
    cdf = np.linspace(0, 1, h * w)
    deposition_map = _run_river_simulation_numba(
        hmap,
        sea_level,
        flow_field_angles,
        temp_dummy,
        cdf,
        params["particle_count"],
        params["particle_steps"],
        params["particle_fade"],
        params["particle_steps"] // 2,
        1.0,
    )

    processed_deposition = np.power(normalize_map(deposition_map), 0.7)
    
    # Create a land mask to clip rivers from appearing in water
    land_mask = hmap > sea_level
    
    # Apply the mask to the final alpha channel
    alpha = (processed_deposition * 255).astype(np.uint8)
    alpha[~land_mask] = 0
    
    rgba_data = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_data[..., 0:3] = hex_to_rgb("#4169E1")
    rgba_data[..., 3] = alpha
    return {"image": Image.fromarray(rgba_data), "diagnostics": {"river_flow_angles": flow_field_angles, "river_deposition": deposition_map}}

def generate_world_data(params, scaling_manager):
        w, h = params["size"], params["size"]
        ms = params["seed"]
        diagnostic_maps = {}

        # --- Generate Base Maps ---
        # Start from the user-specified major plates but always fissure into 17 total plates
        tsm, pp, pv, plate_mask = generate_tectonic_map(w, h, params["plate_points"], ms)
        diagnostic_maps["0_plate_mask"] = plate_mask.copy()
        diagnostic_maps["1_tectonic_potential"] = tsm.copy()
        if params["tectonic_smoothing"] > 0:
            tsm = gaussian_filter(tsm, sigma=params["tectonic_smoothing"])
        diagnostic_maps["2_tectonic_smoothed"] = tsm.copy()
        cm = generate_flow_field_landmass(w, h, params["continent_scale"], params["flow_distortion"], params["continent_octaves"], ms + 1)
        diagnostic_maps["3_continent_base"] = cm.copy()
        # Use ridged multifractal noise with 40× the frequency of the continent base
        ridge_scale = max(params["continent_scale"] / 40.0, 1e-3)
        freq_mult_map = np.clip(cm ** 2, 0.01, 1.0)
        rm = generate_ridged_multifractal_noise(
            w,
            h,
            ridge_scale,
            params["ridge_octaves"],
            ms + 4,
            freq_map=freq_mult_map,
        )
        diagnostic_maps["4_mountain_ridges"] = rm.copy()
        bn = generate_perlin_noise(w, h, params["boundary_detail_scale"], 8, ms + 2)
        diagnostic_maps["5_boundary_noise"] = bn.copy()

        # --- Formula Evaluation for Heightmap ---
        aeval = AstevalInterpreter()
        aeval.symtable.update(params)  # Add all UI parameters
        aeval.symtable.update({'cm': cm, 'tsm': tsm, 'rm': rm, 'bn': bn})
        aeval.symtable['np'] = np  # allow formulas to reference np.*

        try:
            hm = aeval.eval(params['heightmap_formula'])
        except Exception as e:
            raise ValueError(f"Invalid Heightmap Formula: {e}") from e

        diagnostic_maps["6_raw_combined_heightmap"] = hm.copy()
        final_map = normalize_map(hm)
        # Apply preliminary erosion using blurred sea moisture before remap
        final_map = apply_rainfall_erosion(final_map, params, scaling_manager)
        final_map, cdf_diags = earthlike(
            final_map,
            plate_mask,
            params.get("world_diameter_km", 12000),
            params.get("hypsometric_strength", 1.0),
            sea_level=params.get("sea_level", SEA_LEVEL),
            elevation_range_m=params.get("elevation_range_m", 19848),
            apply_blend=True,
            apply_spectral_shape=False,
        )
        diagnostic_maps["6b_earthlike_heightmap"] = final_map.copy()
        diagnostic_maps.update(cdf_diags)


        # --- Calculate Base Flow Field (used for rivers and rain shadow) ---
        # This is now calculated as part of the initial world data
        flow_angles = calculate_flow_field(final_map, params)
        diagnostic_maps["7_base_flow_angles"] = flow_angles.copy()


        # --- Calculate River Deposition Map ---
        # This is also calculated as part of the initial world data
        # Ensure hmap, flow_angles, and params are passed
        river_deposition_data = calculate_river_deposition(final_map, flow_angles, params, scaling_manager)
        river_deposition_map = river_deposition_data['deposition_map']
        diagnostic_maps.update(river_deposition_data.get("diagnostics", {})) # Add river diagnostics


        override = np.full((w, h), -1, dtype=np.int8)

        # Return heightmap, tectonic data, flow angles, AND river deposition map
        return {
            "heightmap": final_map,
            "plate_points": pp,
            "plate_velocities": pv,
            "plate_mask": plate_mask,
            "flow_angles": flow_angles, # Return base flow angles
            "river_deposition_map": river_deposition_map, # Return deposition map
            "biome_override_map": override,
            "diagnostic_maps": diagnostic_maps
        }

# --- New function to calculate river deposition ---
# Extracted the calculation part from the old generate_flow_field_rivers
# It takes hmap and flow_field_angles as inputs now
def calculate_river_deposition(hmap, flow_field_angles, params, scaling_manager, temp_map_kelvin=None):
    """Run the particle simulation to determine river deposition paths."""
    h, w = hmap.shape
    sea_level_norm = params.get('sea_level', SEA_LEVEL)

    temp_c = np.full_like(hmap, 20.0, dtype=np.float32)
    if temp_map_kelvin is not None:
        temp_c = temp_map_kelvin - KELVIN_TO_CELSIUS_OFFSET

    height_m = scaling_manager.to_real(np.maximum(hmap - sea_level_norm, 0), above_sea=True)
    dist_px = distance_transform_edt(hmap <= sea_level_norm)
    km_per_pixel = params.get('world_diameter_km', 12000) / w if w > 0 else 1.0
    dist_km = dist_px * km_per_pixel

    temp_factor = cc_rainfall_multiplier(temp_c)
    spawn_prob = temp_factor * 0.63
    spawn_prob *= (1.0 + 0.0013 * np.minimum(height_m, 2000.0))
    spawn_prob *= 1.0 / (1.0 + np.exp((height_m - 2000.0) / 400.0))
    spawn_prob *= np.exp(-dist_km / 150.0)
    spawn_prob[(temp_c <= 0) | (hmap <= sea_level_norm)] = 0

    flat = spawn_prob.ravel()
    s = flat.sum()
    if s > 0:
        cdf = np.cumsum(flat / s)
    else:
        cdf = np.linspace(0, 1, flat.size)

    deposition_map = _run_river_simulation_numba(
        hmap,
        sea_level_norm,
        flow_field_angles,
        temp_c,
        cdf,
        params["particle_count"],
        params["particle_steps"],
        params["particle_fade"],
        params["particle_steps"] // 2,
        1.0,
    )
    deposition_map[temp_c <= 0] = 0
    return {
        "deposition_map": deposition_map,
        "diagnostics": {
            "river_deposition_raw": deposition_map.copy(),
            "river_spawn_probability": spawn_prob,
        },
    }


def create_river_image(deposition_map, land_mask):
        """
        Creates the visual river layer image from the river deposition map.
        Rivers are clipped to land areas using the land_mask.
        """
        h, w = deposition_map.shape
        # Normalize and power the deposition map for visual strength
        processed_deposition = np.power(normalize_map(deposition_map), 0.7)

        # Apply the land mask to the final alpha channel
        alpha = (processed_deposition * 255).astype(np.uint8)
        alpha[~land_mask] = 0 # Clip alpha to land using the passed mask

        rgba_data = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_data[..., 0:3] = hex_to_rgb("#4169E1") # River color
        rgba_data[..., 3] = alpha

        return Image.fromarray(rgba_data, 'RGBA') # Ensure mode is RGBA

def calculate_albedo_map(biome_indices_map, biomes, hmap, sea_level):
    """Return a constant albedo map based on ocean vs land fraction."""
    land_fraction = np.mean(hmap > sea_level)
    ocean_fraction = 1.0 - land_fraction
    albedo_value = ocean_fraction * 0.06 + land_fraction * 0.4
    return np.full_like(hmap, albedo_value, dtype=np.float32)

def generate_temperature_map(hmap, albedo_map, params, scaling_manager):
    # --- Formula Evaluation for Temperature ---
    aeval = AstevalInterpreter(use_numpy=True)
    aeval.symtable.update(np.__dict__)  # give access to numpy functions
    aeval.symtable['np'] = np  # allow formulas to use np.* syntax
    aeval.symtable.update(params)  # Add all UI parameters
    aeval.symtable['stefan_boltzmann'] = STEFAN_BOLTZMANN
    aeval.symtable['albedo_mean'] = np.mean(albedo_map)
    aeval.symtable['to_real'] = scaling_manager.to_real

    try:
        # 1. Calculate a GLOBAL AVERAGE base temperature (black-body)
        base_k = aeval.eval(params['temperature_base_k_formula'])
        aeval.symtable['base_k'] = base_k

        # 2. Add GLOBAL AVERAGE greenhouse effect
        avg_temp_k = aeval.eval(params['temperature_greenhouse_k_formula'])

        # 3. Create a latitude-based temperature gradient.
        lat_rad = np.deg2rad(np.linspace(params['latitude_south_pole'], params['latitude_north_pole'], hmap.shape[0])).reshape(-1, 1)
        diff = params['equator_pole_temp_diff_c']
        cos45 = np.cos(np.deg2rad(45))
        pole_temp_k = avg_temp_k - (diff * cos45)
        equator_temp_k = avg_temp_k + (diff * (1.0 - cos45))
        
        aeval.symtable.update({'pole_k': pole_temp_k, 'equator_k': equator_temp_k, 'lat_rad': lat_rad})
        sea_level_equivalent_temp = aeval.eval(params['temperature_latitude_gradient_formula'])

        # 4. Simulate atmospheric and oceanic heat distribution (Smoothing)
        smoothing_sigma = params.get('climate_smoothing_sigma', 0)
        smoothed_temp = gaussian_filter(sea_level_equivalent_temp, sigma=smoothing_sigma) if smoothing_sigma > 0 else sea_level_equivalent_temp
        
        # 5. Apply altitude-based temperature drop (Lapse Rate) AFTER smoothing
        aeval.symtable.update({'hmap': hmap})
        lapse_term = aeval.eval(params['temperature_lapse_rate_term_formula'])
        # Ensure lapse term is not negative (can't get warmer on mountains)
        lapse_term = np.maximum(0, lapse_term)

    except Exception as e:
        raise ValueError(f"Invalid Temperature Formula: {e}") from e
    
    # 6. Final temperature.
    final_temp_kelvin = smoothed_temp - lapse_term
    
    return {"temperature_map": final_temp_kelvin, "diagnostics": {"climate_albedo": albedo_map, "climate_base_temp_k": sea_level_equivalent_temp}}

def apply_geostrophic_wind_advection(temp_map_kelvin, params):
    """
    Modifies a temperature map by advecting it with a divergence-free flow
    field derived from the temperature gradient (curl noise).
    """
    if not params.get('enable_geostrophic_winds', False):
        return {"temperature_map": temp_map_kelvin, "diagnostics": {}}

    h, w = temp_map_kelvin.shape
    
    try:
        sigmas_str = params['geostrophic_blur_sigmas']
        sigmas = [float(s.strip()) for s in sigmas_str.split(',') if s.strip()]
        wind_speed_kmh = params['geostrophic_wind_kmh']
        world_diameter_km = params['world_diameter_km']
        advection_steps = params['geostrophic_advection_steps']
    except (ValueError, KeyError) as e:
        raise ValueError(f"Invalid Geostrophic Wind parameter: {e}") from e

    km_per_pixel = world_diameter_km / w
    pixels_per_step = wind_speed_kmh / km_per_pixel if km_per_pixel > 0 else 0

    total_flow_x = np.zeros_like(temp_map_kelvin, dtype=np.float32)
    total_flow_y = np.zeros_like(temp_map_kelvin, dtype=np.float32)
    initial_temp = temp_map_kelvin.copy()

    for sigma in sigmas:
        blurred_temp = gaussian_filter(initial_temp, sigma=sigma, mode='wrap')
        grad_y, grad_x = np.gradient(blurred_temp)
        total_flow_x += grad_y
        total_flow_y += -grad_x

    magnitude = np.sqrt(total_flow_x**2 + total_flow_y**2)
    magnitude[magnitude == 0] = 1.0 
    
    disp_x = (total_flow_x / magnitude) * pixels_per_step
    disp_y = (total_flow_y / magnitude) * pixels_per_step

    advected_temp = temp_map_kelvin.copy()
    y_coords, x_coords = np.indices((h, w))

    for _ in range(advection_steps):
        source_coords_y = y_coords - disp_y
        source_coords_x = x_coords - disp_x
        advected_temp = map_coordinates(
            advected_temp, 
            [source_coords_y, source_coords_x], 
            order=1,
            mode='wrap'
        )
    
    diagnostics = {
        "climate_geostrophic_flow_x": disp_x,
        "climate_geostrophic_flow_y": disp_y,
        "climate_temp_before_advection": initial_temp,
    }

    return {"temperature_map": advected_temp, "diagnostics": diagnostics}

def apply_temperature_correction(temp_map_kelvin, params, target_c=15.0):
    """Adjust global temperatures so the mean equals ``target_c``."""
    current_avg_c = np.mean(temp_map_kelvin - KELVIN_TO_CELSIUS_OFFSET)
    delta = target_c - current_avg_c
    corrected = temp_map_kelvin + delta
    # Provide a diagnostic map so the UI can visualise the correction amount
    delta_map = np.full_like(temp_map_kelvin, delta, dtype=np.float32)
    return {"temperature_map": corrected, "diagnostics": {"temp_correction": delta_map}}

# --- Replace generate_rainfall_map with the corrected version ---
def generate_rainfall_map(hmap, temp_map_kelvin, flow_angles, river_deposition_map, params, scaling_manager):
    """
    Generates a rainfall map based on blurred moisture sources (sea, rivers)
    and a height/wind-based rain shadow effect.
    """
    h, w = hmap.shape
    temp_c = temp_map_kelvin - KELVIN_TO_CELSIUS_OFFSET
    sigma_factor = cc_rainfall_multiplier(np.mean(temp_c))
    cc_factor_map = cc_rainfall_multiplier(temp_c)
    sea_level_norm = params.get('sea_level', SEA_LEVEL)
    world_diameter_km = params.get('world_diameter_km', 12000) # Default 12000 km if not set
    global_rainfall_target = params.get('global_rainfall', 985.5) # Default 985.5 mm/yr if not set
    orographic_strength = params.get('orographic_strength', 1.5) # Used to scale rain shadow
    rain_shadow_strength_factor = params.get('rain_shadow_strength_factor', 1.0) # New parameter to scale rain shadow

    # Ensure essential inputs are not None
    if temp_map_kelvin is None or flow_angles is None or river_deposition_map is None or scaling_manager is None:
         print("Error: Missing inputs for generate_rainfall_map. Returning zero rainfall.")
         return {"rainfall_map": np.zeros_like(hmap), "diagnostics": {"rainfall_error": np.ones_like(hmap)}}


    # 1. Calculate Sea Blur using an exponential kernel
    sea_blur_radius_km = 1000.0
    sea_radius_px = (sea_blur_radius_km / world_diameter_km) * w if world_diameter_km > 1e-9 and w > 0 else 1.0
    sea_radius_px = max(1.0, sea_radius_px) * sigma_factor
    sea_kernel = exp_kernel(int(sea_radius_px), sea_radius_px)

    # Create initial sea mask (float 0-1)
    sea_mask_float = (hmap <= sea_level_norm).astype(np.float32)

    # Apply exponential blur to the sea mask using wrap mode
    blurred_sea_moisture = convolve_wrap_fft(sea_mask_float, sea_kernel)

    # Add blurred sea moisture as diagnostic
    diagnostic_maps = {"rainfall_blurred_sea": blurred_sea_moisture.copy()}


    # 2. Calculate Rain Shadow Map
    # Rain shadow is based on downhill slope component * height above sea level * total strength
    # Need height gradient (dy, dx) for slope
    dy, dx = np.gradient(hmap)

    # Height above sea level in KM (convert normalized height to meters then to km)
    hmap_above_sea_norm = np.maximum(0, hmap - sea_level_norm)
    # Use the scaling manager to convert normalized height difference to real meters
    real_height_above_sea_m = scaling_manager.to_real(hmap_above_sea_norm, above_sea=True)
    # Use elevation_range_m with fixed land fraction if scaling manager fails
    if real_height_above_sea_m.max() > 1e-9:
        height_in_km = real_height_above_sea_m / 1000.0
    else:
        land_max = params.get('elevation_range_m', 19848) * (8848.0 / 19848.0)
        height_in_km = (hmap_above_sea_norm / max(1e-9, 1 - sea_level_norm)) * land_max / 1000.0

    # Downhill slope component (positive where wind blows downhill relative to gradient)
    # This calculates the component of the height gradient vector (dx, dy) in the direction of the wind (flow_angles)
    # If flow_angles is the direction the wind *comes from*, we want the component in the *opposite* direction (flow_angles + pi)
    # Gradient points uphill, so (-dx, -dy) points downhill. Dot product with wind vector (cos(angle), sin(angle)) gives downhill component in wind direction.
    downhill_slope_component = (np.cos(flow_angles) * (-dx) + np.sin(flow_angles) * (-dy))
    # We only care about the wind blowing *downhill* for rain shadow (positive component)
    downhill_slope_component = np.maximum(0, downhill_slope_component)

    # Calculate total rain shadow strength incorporating UI parameters
    rain_shadow_strength_total = orographic_strength * rain_shadow_strength_factor

    # Calculate temperature influence factor for rain shadow
    # Related to local temp deviation from average. Need average temp in C.
    # Assume temp_map_kelvin is the map *after* advection and final blur
    avg_temp_k = np.mean(temp_map_kelvin)
    local_temp_c = temp_map_kelvin - KELVIN_TO_CELSIUS_OFFSET
    avg_temp_c = avg_temp_k - KELVIN_TO_CELSIUS_OFFSET
    temp_range_c = params.get('equator_pole_temp_diff_c', 60.0) # Default 60 C

    # Calculate temp influence factor. Use a small epsilon to avoid division by zero if range is 0.
    temp_influence = 1.0
    if temp_range_c > 1e-9:
         # Factor based on how much colder/warmer the area is compared to average, relative to the global temp range
         # Colder areas might have less moisture capacity, increasing rain shadow effect?
         # Or warmer areas increase evaporation, reducing moisture?
         # Let's use the suggested formula: 1.0 + 1.07 * TempDelta / TempRange
         temp_influence = 1.0 + 1.07 * ((local_temp_c - avg_temp_c) / temp_range_c)
         # Clamp the influence to avoid extreme values
         temp_influence = np.clip(temp_influence, 0.1, 5.0) # Arbitrary clamp values


    # Combine factors for final rain shadow multiplier
    final_rain_shadow_multiplier = rain_shadow_strength_total * temp_influence

    # Calculate rain shadow map: Effect scales with height above sea *AND* downhill slope component *AND* multiplier
    # The downhill_slope_component is already a rate (change per pixel).
    # Multiply height_in_km (total height) by the downhill slope rate AND the multiplier.
    # rain_shadow_map = height_in_km * downhill_slope_component * final_rain_shadow_multiplier
    # Let's rethink the formula: Shadow is stronger for taller mountains AND for wind blowing directly downhill.
    # A simpler model: Rain shadow = HeightAboveSea(km) * DownhillInfluence * Multiplier
    # DownhillInfluence could be related to the magnitude of downhill_slope_component, perhaps non-linearly
    # Let's use a simple multiplier based on the clamped positive downhill slope component.
    # The scale of downhill_slope_component depends on pixel size. Normalize it? Max gradient can be sqrt(2).
    # Let's try: Rain Shadow = height_in_km * (1 + clamped_downhill_slope * scale) * final_rain_shadow_multiplier
    clamped_downhill_slope = np.clip(downhill_slope_component, 0, 1) # Assuming max gradient is 1 in pixel coords
    slope_influence_scale = 100.0 # Arbitrary scale
    rain_shadow_map = height_in_km * (1.0 + clamped_downhill_slope * slope_influence_scale) * final_rain_shadow_multiplier


    # Add rain shadow map as diagnostic
    diagnostic_maps["rainfall_rain_shadow_amount"] = rain_shadow_map.copy() # Store the amount to subtract


    # Subtract rain shadow from blurred sea moisture
    # Moisture is lost as wind goes downhill behind a mountain
    sea_moisture_after_shadow = blurred_sea_moisture - rain_shadow_map
    sea_moisture_after_shadow = np.maximum(0, sea_moisture_after_shadow) # Ensure non-negative


    # Add sea moisture after shadow as diagnostic
    diagnostic_maps["rainfall_sea_after_shadow"] = sea_moisture_after_shadow.copy()


    # 3. Blur River Deposition Map
    if river_deposition_map is None:
         print("Warning: River deposition map is None in generate_rainfall_map.")
         blurred_river_moisture = np.zeros_like(hmap)
         diagnostic_maps["rainfall_blurred_rivers"] = blurred_river_moisture.copy()
    else:
        # Normalize raw deposition map first for consistent scaling
        normalized_raw_deposition = normalize_map(river_deposition_map)
        normalized_raw_deposition[normalized_raw_deposition > 0.01] = (1 / 0.35)
        # River blur radius in km
        river_blur_radius_km = 300.0
        river_radius_px = (
            river_blur_radius_km / world_diameter_km
        ) * w if world_diameter_km > 1e-9 and w > 0 else 1.0
        river_radius_px = max(1.0, river_radius_px) * sigma_factor
        river_kernel = exp_kernel(int(river_radius_px), river_radius_px)
        # Apply exponential blur to normalized river deposition map
        blurred_river_moisture = convolve_wrap_fft(
            normalized_raw_deposition,
            river_kernel,
        )
        # Scale river contribution to rainfall
        river_influence_strength = 0.5  # Arbitrary scale factor for river influence
        blurred_river_moisture *= river_influence_strength

        # Add blurred river moisture as diagnostic
        diagnostic_maps["rainfall_blurred_rivers"] = blurred_river_moisture.copy()


       # 4. Combine Blurred Maps
    # Total moisture influence is sum of sea moisture (after shadow) and river moisture
    total_moisture_influence = sea_moisture_after_shadow + blurred_river_moisture
    diagnostic_maps["rainfall_total_influence"] = total_moisture_influence.copy()

     # Apply Clausius-Clapeyron scaling to moisture influence
    intensity_map = total_moisture_influence * cc_factor_map
    diagnostic_maps["rainfall_intensity_raw"] = intensity_map.copy()

    # Clip intensities so values >1 do not receive disproportionate rain
    diagnostic_maps["rainfall_intensity"] = intensity_map.copy()

    total_weight = intensity_map.sum()

    final_rainfall_map = np.zeros_like(hmap, dtype=np.float32)
    if total_weight > 1e-9:
        scale = global_rainfall_target * intensity_map.size / total_weight
        final_rainfall_map = intensity_map * scale
    # else remain zeros

    # Ensure final rainfall is non-negative
    final_rainfall_map = np.maximum(0, final_rainfall_map)


    # Add final rainfall map as diagnostic
    diagnostic_maps["rainfall_final_map"] = final_rainfall_map.copy()


    return {"rainfall_map": final_rainfall_map, "diagnostics": diagnostic_maps}

# ... (rest of the file) ...

def generate_classification_maps(hmap, temp_map, rain_map, biomes, params):
    h, w = hmap.shape
    sea_level = params['sea_level']
    temp_c = temp_map - KELVIN_TO_CELSIUS_OFFSET
    potential_water = hmap <= sea_level
    
    # Use scipy.ndimage.label to correctly distinguish oceans from inland water
    water_labels, num_features = label(potential_water)
    edge_labels = np.unique(np.concatenate([
        water_labels[0, :], water_labels[-1, :],
        water_labels[:, 0], water_labels[:, -1]
    ]))
    ocean_mask = np.isin(water_labels, edge_labels[edge_labels != 0])
    
    inland_water_mask = potential_water & ~ocean_mask
    land_mask = ~potential_water

    land_biomes = [b for b in biomes if not b.get('is_water', False)]
    ocean_biome_idx = next((i for i, b in enumerate(biomes) if b['name'] == 'OCEAN'), -1)
    inland_water_biome_idx = next((i for i, b in enumerate(biomes) if b['name'] == 'INLAND_WATER'), -1)
    fallback_biome_def = land_biomes[-1] if land_biomes else None
    fallback_idx = next((i for i, b in enumerate(biomes) if b['name'] == fallback_biome_def['name']), -1) if fallback_biome_def else -1

    final_indices = np.full((h, w), fallback_idx, dtype=np.int16)
    # Iterate through biomes in the order they are defined.
    for i, biome in enumerate(biomes):
        if biome.get('is_water', False): continue
        temp_check = (temp_c >= biome['temp_min']) & (temp_c < biome['temp_max'])
        rain_check = (rain_map >= biome['rain_min']) & (rain_map < biome['rain_max'])
        
        # Apply biome only to land that has not yet been classified.
        biome_mask = temp_check & rain_check & land_mask & (final_indices == fallback_idx)
        final_indices[biome_mask] = i
    
    final_indices[ocean_mask] = ocean_biome_idx
    final_indices[inland_water_mask] = inland_water_biome_idx
    
    colors = np.array([hex_to_rgb(b['color']) for b in biomes], dtype=np.uint8)
    rgb_map = np.zeros((h, w, 3), dtype=np.uint8)
    valid_mask = final_indices != -1
    rgb_map[valid_mask] = colors[final_indices[valid_mask]]
    
    return {"biome_image": Image.fromarray(rgb_map, 'RGB'), "biome_indices_map": final_indices, "land_mask": land_mask}

def create_heatmap_image(hmap):
    return Image.fromarray((normalize_map(hmap) * 255).astype(np.uint8)).convert("RGBA")

# --- ADD THIS NEW FUNCTION (UPDATED FORMULA) ---
def apply_altitude_tint(biome_image, hmap, sea_level_norm, scaling_manager, temp_map_kelvin, rain_map_mm):
        """
        Applies a complex tint to land based on temperature, rainfall, and altitude.
        Tint = (R, G, B) where:
        R = + (4 * (Temp - AverageTemp) - 0.01 * (Rain - AverageRain) + 18 * (Elevation/1 km))
        G = + (-3.5 * (Temp - AverageTemp) + 0.02 * (Rain - AverageRain) + 12 * (Elevation/1 km))
        B = + (-(Temp - AverageTemp) - 0.005 * (Rain - AverageRain) + 18 * (Elevation/1 km))
        Temp is in Celsius, Rain is in mm/yr, Elevation is height above sea level in km.
        """
        # ... (existing code - unchanged) ...
        rgba_array = np.array(biome_image.convert('RGBA'), dtype=np.float32)
        rgb_array = rgba_array[..., :3]
        alpha_channel = rgba_array[..., 3]
        h, w = hmap.shape
        land_above_sea_mask = (hmap > sea_level_norm) & (alpha_channel > 0)
        masked_indices = np.where(land_above_sea_mask)
        if masked_indices[0].size == 0: return biome_image

        # Calculate global averages for temperature and rainfall over the *entire map*
        # This gives a baseline for the 'delta' calculations
        avg_temp_k = np.mean(temp_map_kelvin)
        avg_rain_mm = np.mean(rain_map_mm) # Use mean of the *final* rainfall map


        # Get local temperature and rainfall values for masked pixels
        local_temp_k = temp_map_kelvin[masked_indices]
        local_rain_mm = rain_map_mm[masked_indices] # Use values from the *final* rainfall map


        # Convert local and average temp to Celsius for the formula
        local_temp_c = local_temp_k - KELVIN_TO_CELSIUS_OFFSET
        avg_temp_c = avg_temp_k - KELVIN_TO_CELSIUS_OFFSET


        # Get normalized height above sea level for masked pixels
        norm_height_above_sea = hmap[masked_indices] - sea_level_norm


        # Calculate real height above sea level in meters and then kilometers
        # Use the scaling manager passed from main.py
        real_height_above_sea_m = scaling_manager.to_real(norm_height_above_sea, above_sea=True)
        height_in_km = real_height_above_sea_m / 1000.0


        # Calculate the tint components for the masked pixels based on the formula
        delta_temp = local_temp_c - avg_temp_c
        delta_rain = local_rain_mm - avg_rain_mm


        # Calculate R, G, B tint values for each pixel in the mask
        # Use height_in_km[:, None] to enable broadcasting across the [R, G, B] components
        # Formula: R = + (4 × (Temp - Average) - 0.01 × (Precipitation - Average) + 18 × (Elevation/1 km))
        # Formula: G = + (-3.5 × (Temp - Average) + 0.02 × (Precipitation - Average) + 12 × (Elevation/1 km))
        # Formula: B = + (-(Temp - Average) - 0.005 × (Precipitation - Average) + 18 × (Elevation/1 km))

        tint_r = (1.6 * delta_temp - 0.05 * delta_rain + 12.0 * height_in_km)
        tint_g = (-0.7 * delta_temp + 0.1 * delta_rain + 8.0 * height_in_km)
        tint_b = (-0.7 * delta_temp - 0.025 * delta_rain + 12.0 * height_in_km)

        # Stack the tint components into a (N, 3) array where N is the number of masked pixels
        tint_to_add_for_masked = np.stack([tint_r, tint_g, tint_b], axis=-1)


        # Add the calculated tint to the original pixel colors for the masked areas
        rgb_array[masked_indices] += tint_to_add_for_masked


        # Clip RGB values to the valid range [0, 255]
        rgb_array = np.clip(rgb_array, 0, 255) # Keep as float for now

        # Combine the potentially modified RGB array with the original alpha channel
        tinted_rgba_array = np.concatenate([rgb_array, alpha_channel[:, :, None]], axis=-1)


        # Convert back to PIL Image (ensure it's RGBA)
        tinted_image = Image.fromarray(tinted_rgba_array.astype(np.uint8), 'RGBA')


        return tinted_image
# --- END OF NEW FUNCTION (UPDATED FORMULA) ---

def create_color_image(data_map, colormap_points):
    h, w = data_map.shape
    rgb_map = np.zeros((h, w, 3), dtype=np.uint8)
    values, colors = zip(*colormap_points)
    values, colors = np.array(values), np.array(colors)
    r, g, b = [np.clip(np.interp(data_map, values, colors[:, i]), 0, 255) for i in range(3)]
    rgb_map[..., 0], rgb_map[..., 1], rgb_map[..., 2] = r, g, b
    return Image.fromarray(rgb_map, 'RGB').convert('RGBA')

def create_temperature_image(temp_map_kelvin):
    temp_c = temp_map_kelvin - KELVIN_TO_CELSIUS_OFFSET
    cmap = [(-40,[0,0,255]), (-20,[0,255,255]), (0,[255,255,255]), (20,[255,255,0]), (40,[255,0,0])]
    return create_color_image(temp_c, cmap)

def create_rainfall_image(rain_map_mm):
    cmap = [(0,[180,140,80]), (250,[220,200,100]), (500,[120,200,120]), (1000,[50,150,180]), (2000,[20,80,220])]
    return create_color_image(rain_map_mm, cmap)

def create_angle_map_image(angle_map_rad):
    h, w = angle_map_rad.shape
    hue = (angle_map_rad + np.pi) / (2 * np.pi)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0], hsv[..., 1], hsv[..., 2] = (hue * 255), 255, 255
    return Image.fromarray(hsv, mode='HSV').convert('RGBA')

def create_flow_map_image(pts, vels, w, h):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, p in enumerate(pts):
        v = vels[i]
        sp, ep = (p[0], p[1]), (p[0] + v[0] * 25, p[1] + v[1] * 25)
        draw.line([sp, ep], fill=(255, 0, 0, 200), width=2)
        draw.ellipse([sp[0] - 3, sp[1] - 3, sp[0] + 3, sp[1] + 3], fill=(255, 0, 0, 220), outline=(0, 0, 0, 255))
    return img