#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fire Detection with MTG-FCI Satellite Imagery
==============================================
AGH University of Krakow — Aerospace Engineering
Author: Valerio Pampanoni (valerio.pampanoni@uniroma1.it)

Case study: Biebrza National Park wildfire, Poland (April 2025)

This script implements a simplified fire detection algorithm based on the
MODIS Collection 5 approach (Giglio et al., 2003) applied to MTG-FCI
(Meteosat Third Generation — Flexible Combined Imager) data.

The algorithm detects active fires using:
  1. High MWIR (3.8 µm) brightness temperature
  2. Large MWIR – TIR (10.5 µm) brightness temperature difference
  3. Contextual verification against local background statistics
  4. FRP estimation using the MODIS equation (Wooster et al., 2005)

Multiple timestamps are processed automatically, and a cumulative FRP
timeseries is generated for the Biebrza National Park area of interest.
"""

# %% Imports
import datetime
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import geopandas as gpd
import pandas as pd
from osgeo import gdal, osr
from scipy.ndimage import uniform_filter
from pyorbital.astronomy import sun_zenith_angle
import leafmap

gdal.UseExceptions()

# %% =========================================================================
#    CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ---- MTG-FCI band mapping ----
# FDHSI bands (lower resolution, ~2 km over Europe)
FDHSI_BANDS = {
    "vis_06": "RED_REF",       # 0.64 µm — red reflectance
    "vis_08": "NIR_08_REF",    # 0.86 µm — NIR reflectance
    "nir_13": "NIR_13_REF",    # 1.3 µm  — NIR reflectance (used for cloud masking)
    "nir_22": "SWIR_22_REF",   # 2.2 µm  — shortwave IR reflectance
    "ir_123": "TIR_12_BT",     # 12.3 µm — brightness temperature
    "natural_color": "NCC",    # Natural colour composite (RGB)
}
# HRFI bands (higher resolution, ~1 km over Europe)
HRFI_BANDS = {
    "ir_38": "MWIR_BT",        # 3.8 µm  — brightness temperature (KEY for fire detection)
    "ir_38_rad": "MWIR_RAD",   # 3.8 µm  — radiance (for Wooster FRP)
    "ir_105": "TIR_11_BT",     # 10.5 µm — brightness temperature
}

# ---- Detection thresholds (MODIS Collection 5 adapted for MTG-FCI) ----
MWIR_THRESHOLD = 310.0         # K  — min MWIR BT for potential fire (daytime)
MWIR_TIR_DIFF_THRESHOLD = 10.0 # K  — min MWIR–TIR diff for potential fire

# Contextual confirmation coefficients
C1_ABSOLUTE_MWIR = 360.0       # K  — very high MWIR BT → automatic confirm
C2_DIFF_MAD_MULT = 3.0         #    — MWIR–TIR diff > bg_mean + C2*MAD
C3_DIFF_ABSOLUTE = 5.5         # K  — MWIR–TIR diff above background by this offset
C4_MWIR_MAD_MULT = 2.5         #    — MWIR BT > bg_mean + C4*MAD
CONTEXT_WINDOW = 21            # px — background statistics window size

# ---- Biebrza National Park approximate bounding box (WGS84) ----
BIEBRZA_BBOX = {
    "lon_min": 22.4, "lon_max": 23.6,
    "lat_min": 53.3, "lat_max": 53.75,
}


# %% =========================================================================
#    UTILITY FUNCTIONS
# =============================================================================

def discover_timestamps(data_dir):
    """
    Scan the data directory and return a sorted list of datetime objects
    for which FDHSI, HRFI, and CLM zip files are all present.

    Parameters
    ----------
    data_dir : Path
        Directory containing the MTG zip archives.

    Returns
    -------
    list of datetime.datetime
        Sorted list of available observation times.
    """
    pattern = re.compile(r"MTG_FDHSI_POL_(\d{12})\.zip")
    timestamps = []
    for f in sorted(data_dir.glob("MTG_FDHSI_POL_*.zip")):
        m = pattern.match(f.name)
        if not m:
            continue
        dtstr = m.group(1)
        dtime = datetime.datetime.strptime(dtstr, "%Y%m%d%H%M")
        # Only include if the HRFI file is also present
        hrfi = data_dir / f"MTG_HRFI_POL_{dtstr}.zip"
        if hrfi.exists():
            timestamps.append(dtime)
    return timestamps


def extract_bands_from_zip(zip_path, band_names, out_dir):
    """
    Extract specific GeoTIFF bands from a zip archive into a directory.

    Parameters
    ----------
    zip_path : Path
        Path to the zip file.
    band_names : list of str
        Band filenames (without .tif extension) to extract.
    out_dir : Path
        Destination directory for extracted files.

    Returns
    -------
    dict
        {band_name: extracted_file_path_str}
    """
    extracted = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            for band in band_names:
                if member.endswith(band + ".tif"):
                    out_path = out_dir / os.path.basename(member)
                    if not out_path.exists():
                        data = zf.read(member)
                        with open(out_path, "wb") as f:
                            f.write(data)
                    extracted[band] = str(out_path)
    return extracted


def read_band_as_masked_array(file_path, band_num=1):
    """
    Read a single band from a GeoTIFF into a NumPy masked array.

    The GDAL NoData value is used as the mask; NaN values are also masked.
    Accepts both regular file paths and GDAL virtual filesystem paths
    such as /vsizip/.

    Parameters
    ----------
    file_path : str
        Path to the GeoTIFF (or /vsizip/ path).
    band_num : int
        Band number to read (1-indexed).

    Returns
    -------
    numpy.ma.MaskedArray
    """
    ds = gdal.Open(file_path)
    if ds is None:
        raise ValueError(f"Could not open: {file_path}")
    band = ds.GetRasterBand(band_num)
    nodata = band.GetNoDataValue()
    arr = band.ReadAsArray().astype(np.float32)
    ds = None

    if nodata is not None and np.isnan(nodata):
        return ma.masked_invalid(arr)
    elif nodata is not None:
        return ma.masked_values(arr, nodata)
    else:
        return ma.masked_invalid(arr)


def get_geotransform(file_path):
    """Return the GDAL 6-element geotransform for a raster file."""
    ds = gdal.Open(file_path)
    gt = ds.GetGeoTransform()
    ds = None
    return gt


def get_projection_wkt(file_path):
    """Return the WKT projection string for a raster file."""
    ds = gdal.Open(file_path)
    wkt = ds.GetProjection()
    ds = None
    return wkt


def get_raster_extent(file_path):
    """Return [xmin, ymin, xmax, ymax] for a raster file."""
    ds = gdal.Open(file_path)
    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    ds = None
    xmin, ymax = gt[0], gt[3]
    xmax = xmin + w * gt[1] + h * gt[2]
    ymin = ymax + w * gt[4] + h * gt[5]
    return [xmin, ymin, xmax, ymax]


def get_lat_lon_grids(file_path, pixel_center=True):
    """
    Compute per-pixel longitude and latitude grids in WGS84 (EPSG:4326).

    Parameters
    ----------
    file_path : str
        Path to a georeferenced raster.
    pixel_center : bool
        If True, return coordinates for pixel centres.

    Returns
    -------
    lons, lats : numpy.ndarray
        2D arrays of the same shape as the raster.
    """
    ds = gdal.Open(file_path)
    gt = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    src_wkt = ds.GetProjection()
    ds = None

    ox, pw, xr, oy, yr, ph = gt
    if pixel_center:
        ox += pw / 2.0
        oy += ph / 2.0

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    x_nat = ox + cols * pw + rows * xr
    y_nat = oy + cols * yr + rows * ph

    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src_wkt)
    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(4326)
    tgt_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    pts = np.vstack((x_nat.ravel(), y_nat.ravel(), np.zeros(x_nat.size))).T
    transformed = np.array(transform.TransformPoints(pts))

    lons = transformed[:, 0].reshape(height, width)
    lats = transformed[:, 1].reshape(height, width)
    return lons, lats


def resample_to_target(source_path, target_shape, out_bounds, method="nearest"):
    """
    Resample a raster to a target grid using GDAL Warp (in-memory).

    Parameters
    ----------
    source_path : str
        Path to the source raster.
    target_shape : tuple
        (rows, cols) of the desired output.
    out_bounds : list
        [xmin, ymin, xmax, ymax] output bounds.
    method : str
        Resampling algorithm ('nearest', 'bilinear', etc.).

    Returns
    -------
    numpy.ndarray
    """
    methods = {"nearest": gdal.GRA_NearestNeighbour, "bilinear": gdal.GRA_Bilinear}
    ds = gdal.Open(source_path)
    mem_ds = gdal.Warp(
        "", ds, format="MEM",
        width=target_shape[1], height=target_shape[0],
        resampleAlg=methods[method], outputBounds=out_bounds,
    )
    arr = mem_ds.ReadAsArray()
    ds = None
    mem_ds = None
    return arr


def get_pixel_area_grid(geotransform, shape, src_crs_wkt, cache_file=None):
    """
    Compute per-pixel area (km²) using equal-area reprojection (EPSG:6933).

    Results are cached to a .npy file for repeated runs.

    Parameters
    ----------
    geotransform : tuple
        GDAL 6-element geotransform.
    shape : tuple
        (rows, cols).
    src_crs_wkt : str
        Source CRS as WKT.
    cache_file : str or Path, optional
        Path to .npy cache file.

    Returns
    -------
    numpy.ma.MaskedArray
        2D array of pixel areas in km².
    """
    if cache_file and os.path.exists(cache_file):
        return ma.masked_invalid(np.load(cache_file, allow_pickle=True))

    rows, cols = shape
    gt = geotransform
    ci, ri = np.meshgrid(np.arange(cols + 1), np.arange(rows + 1))
    X = gt[0] + ci * gt[1] + ri * gt[2]
    Y = gt[3] + ci * gt[4] + ri * gt[5]

    x_tl, y_tl = X[:-1, :-1].ravel(), Y[:-1, :-1].ravel()
    x_tr, y_tr = X[:-1, 1:].ravel(), Y[:-1, 1:].ravel()
    x_br, y_br = X[1:, 1:].ravel(), Y[1:, 1:].ravel()
    x_bl, y_bl = X[1:, :-1].ravel(), Y[1:, :-1].ravel()

    coords = np.array([[x_tl, y_tl], [x_tr, y_tr],
                       [x_br, y_br], [x_bl, y_bl]]).transpose(2, 0, 1)
    try:
        from shapely import polygons
        geoms = polygons(coords)
    except ImportError:
        from shapely.geometry import Polygon
        geoms = [Polygon(c) for c in coords]

    gs = gpd.GeoSeries(geoms, crs=src_crs_wkt)
    areas = gs.to_crs("EPSG:6933").area.values.reshape(rows, cols)
    area_km2 = ma.masked_array(areas, mask=~np.isfinite(areas)) / 1e6

    if cache_file:
        np.save(cache_file, area_km2.filled(np.nan))
    return area_km2


# %% =========================================================================
#    FIRE DETECTION FUNCTIONS
# =============================================================================

def build_data_cube(fdhsi_files, hrfi_files, target_shape, out_bounds):
    """
    Load all bands into a dictionary, resampling FDHSI bands to the
    HRFI resolution grid.

    Parameters
    ----------
    fdhsi_files : dict
        {band_name: file_path} for FDHSI bands.
    hrfi_files : dict
        {band_name: file_path} for HRFI bands.
    target_shape : tuple
        (rows, cols) of the HRFI grid.
    out_bounds : list
        [xmin, ymin, xmax, ymax] of the HRFI grid.

    Returns
    -------
    dict
        {tag: masked_array} for all bands.
    """
    cube = {}

    # HRFI bands are already at target resolution
    for band_name, tag in HRFI_BANDS.items():
        if band_name in hrfi_files:
            arr = read_band_as_masked_array(hrfi_files[band_name])
            np.place(arr, np.isnan(arr), 0.0)
            if hasattr(arr, 'fill_value') and arr.fill_value > 0.0:
                arr.fill_value = 0.0
            cube[tag] = arr

    # FDHSI bands need resampling to the HRFI grid
    for band_name, tag in FDHSI_BANDS.items():
        if band_name not in fdhsi_files:
            continue
        if tag == "NCC":
            arr_r = read_band_as_masked_array(fdhsi_files[band_name], band_num=1)
            arr_g = read_band_as_masked_array(fdhsi_files[band_name], band_num=2)
            arr_b = read_band_as_masked_array(fdhsi_files[band_name], band_num=3)
            cube[tag] = np.dstack([arr_r, arr_g, arr_b])
        else:
            resampled = resample_to_target(
                fdhsi_files[band_name], target_shape, out_bounds, method="nearest"
            )
            np.place(resampled, np.isnan(resampled), 0.0)
            cube[tag] = ma.masked_values(resampled, 0.0)

    return cube


def apply_exclusion_mask(cube, mask):
    """
    Mask out pixels in all cube bands (e.g. clouds, nighttime).

    Parameters
    ----------
    cube : dict
    mask : numpy.ndarray (bool)
        True = exclude this pixel.

    Returns
    -------
    dict
    """
    masked_cube = {}
    for key, arr in cube.items():
        if key == "NCC":
            masked_cube[key] = arr
            continue
        new_arr = arr.data.copy()
        fill = arr.fill_value if hasattr(arr, 'fill_value') else 0.0
        np.place(new_arr, mask, fill)
        masked_cube[key] = ma.masked_values(new_arr, fill)
    return masked_cube


def compute_cloud_mask(cube,
                       red_thresh=0.13,
                       nir13_thresh=0.01,
                       tir12_thresh=265.0):
    """
    Simple threshold-based cloud mask derived directly from the data cube.

    A pixel is flagged as cloudy if ANY of the following conditions is true:
      - Red reflectance (0.64 µm) > red_thresh       (bright cloud tops)
      - NIR 1.3 µm reflectance    > nir13_thresh      (cirrus detection)
      - TIR 12.3 µm BT           < tir12_thresh  [K]  (cold cloud tops)

    Parameters
    ----------
    cube : dict
        Data cube containing 'RED_REF', 'NIR_13_REF', and 'TIR_12_BT'.
    red_thresh : float
        Red reflectance threshold (default 0.13).
    nir13_thresh : float
        NIR 1.3 µm reflectance threshold (default 0.01).
    tir12_thresh : float
        TIR 12.3 µm brightness temperature threshold in K (default 265.0).

    Returns
    -------
    numpy.ndarray (bool)
        True = cloudy pixel.
    """
    mask_red  = cube["RED_REF"].filled(0)    > red_thresh
    mask_nir  = cube["NIR_13_REF"].filled(0) > nir13_thresh
    mask_tir  = cube["TIR_12_BT"].filled(999) < tir12_thresh

    return mask_red | mask_nir | mask_tir


def detect_potential_hotspots(cube, mwir_thresh, diff_thresh):
    """
    MODIS Collection 5 potential hotspot detection (daytime).

    A pixel is flagged if:
      (1) MWIR brightness temperature > mwir_thresh, AND
      (2) MWIR – TIR brightness temperature difference > diff_thresh.

    Parameters
    ----------
    cube : dict
    mwir_thresh : float
        MWIR BT threshold (K).
    diff_thresh : float
        MWIR–TIR difference threshold (K).

    Returns
    -------
    numpy.ndarray (bool)
    """
    mwir = cube["MWIR_BT"]
    tir = cube["TIR_11_BT"]
    return np.logical_and(mwir > mwir_thresh, (mwir - tir) > diff_thresh)


def compute_background_stats(image, fire_mask, window_size):
    """
    Compute windowed background mean and MAD, excluding fire pixels.

    Uses FFT-based convolution for efficiency. Fire pixels are excluded
    so that active fires do not bias the background statistics.

    Parameters
    ----------
    image : numpy.ma.MaskedArray
        Input band (e.g. MWIR BT).
    fire_mask : numpy.ndarray (bool)
        True = fire pixel (excluded from background computation).
    window_size : int
        Side length of the square window.

    Returns
    -------
    bg_mean, bg_mad : numpy.ndarray
        Same shape as input.
    """
    data = image.filled(0.0).astype(float)
    base_mask = ma.getmaskarray(image)
    valid = (~(base_mask | fire_mask)).astype(float)
    kernel = np.ones((window_size, window_size), dtype=float)

    def fft_conv(a, b):
        s = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)
        out = np.fft.irfftn(np.fft.rfftn(a, s=s) * np.fft.rfftn(b, s=s), s=s)
        pm, pn = (b.shape[0] - 1) // 2, (b.shape[1] - 1) // 2
        return out[pm:pm + a.shape[0], pn:pn + a.shape[1]]

    count = fft_conv(valid, kernel)
    with np.errstate(divide="ignore", invalid="ignore"):
        sum_x = fft_conv(data * valid, kernel)
        sum_x2 = fft_conv((data ** 2) * valid, kernel)
        bg_mean = np.where(count > 0, sum_x / count, np.nan)
        bg_var = np.where(count > 0, sum_x2 / count - bg_mean ** 2, np.nan)
        bg_std = np.sqrt(np.maximum(bg_var, 0.0))
    return bg_mean, bg_std


def confirm_hotspots(cube, pot_fire_mask, window_size, c1, c2, c3, c4):
    """
    Contextual confirmation of potential hotspots (daytime).

    Applies the MODIS Collection 5 confirmation tests:
      C1: MWIR BT > c1 (absolute threshold — automatic confirm)
      C2: MWIR–TIR diff > bg_mean + c2 × MAD
      C3: MWIR–TIR diff > bg_mean + c3
      C4: MWIR BT > bg_mean + c4 × MAD

    A pixel is confirmed if: C1 OR (C2 AND C3 AND C4)

    Parameters
    ----------
    cube : dict
    pot_fire_mask : numpy.ndarray (bool)
    window_size : int
    c1, c2, c3, c4 : float
        Confirmation coefficients.

    Returns
    -------
    numpy.ndarray (bool)
    """
    mwir = cube["MWIR_BT"]
    tir = cube["TIR_11_BT"]
    diff = mwir - tir

    mir_mean, mir_mad = compute_background_stats(mwir, pot_fire_mask, window_size)
    diff_mean, diff_mad = compute_background_stats(
        ma.masked_array(diff if not hasattr(diff, 'filled') else diff,
                        mask=ma.getmaskarray(mwir)),
        pot_fire_mask, window_size,
    )

    mwir_vals = np.where(pot_fire_mask, mwir.filled(0), 0)
    diff_vals = np.where(pot_fire_mask,
                         diff.filled(0) if hasattr(diff, 'filled') else diff, 0)

    mask_c1 = mwir_vals > c1
    mask_c2 = diff_vals > (diff_mean + c2 * diff_mad)
    mask_c3 = diff_vals > (diff_mean + c3)
    mask_c4 = mwir_vals > (mir_mean + c4 * mir_mad)

    confirmed = np.logical_or(mask_c1,
                               np.logical_and(mask_c2, np.logical_and(mask_c3, mask_c4)))
    confirmed = np.logical_and(confirmed, pot_fire_mask)

    # DEBUG: diagnostics at the hottest potential fire pixel
    pf_idx = np.argwhere(pot_fire_mask)
    if len(pf_idx) > 0:
        mwir_filled = mwir.filled(np.nan)
        best = max(range(len(pf_idx)), key=lambda i: mwir_filled[pf_idx[i][0], pf_idx[i][1]])
        r, c = pf_idx[best]
        print(f"    [confirm] hottest pot pixel ({r},{c}): "
              f"MWIR={mwir_vals[r,c]:.1f}, diff={diff_vals[r,c]:.1f}")
        print(f"      mir_mean={mir_mean[r,c]:.1f}, mir_mad={mir_mad[r,c]:.1f}, "
              f"diff_mean={diff_mean[r,c]:.1f}, diff_mad={diff_mad[r,c]:.1f}")
        print(f"      C1({c1})={mask_c1[r,c]}, "
              f"C2(diff>{diff_mean[r,c]+c2*diff_mad[r,c]:.1f})={mask_c2[r,c]}, "
              f"C3(diff>{diff_mean[r,c]+c3:.1f})={mask_c3[r,c]}, "
              f"C4(mwir>{mir_mean[r,c]+c4*mir_mad[r,c]:.1f})={mask_c4[r,c]}")
        print(f"      confirmed={confirmed[r,c]}")

    return confirmed


def compute_frp_modis(mwir_bt_fire, mwir_bt_bg, pixel_area_km2):
    """
    Compute Fire Radiative Power using the MODIS equation.

    FRP [MW] = pixel_area [km²] × 4.34×10⁻¹⁹ × (T_fire⁸ – T_bg⁸)

    This relates the 3.8 µm radiance excess to fire radiative power
    via the 8th-power approximation of the Planck function.
    (Wooster et al., 2003; Giglio et al., 2003)

    Parameters
    ----------
    mwir_bt_fire : numpy.ndarray
        MWIR brightness temperature of fire pixels (K).
    mwir_bt_bg : numpy.ndarray
        MWIR background brightness temperature (K).
    pixel_area_km2 : numpy.ndarray
        Area of each pixel (km²).

    Returns
    -------
    numpy.ndarray
        FRP in MW per pixel.
    """
    return pixel_area_km2 * 4.34e-19 * (mwir_bt_fire**8 - mwir_bt_bg**8)


# %% =========================================================================
#    SINGLE-TIMESTAMP PROCESSING
# =============================================================================

def process_single_datetime(dtime, pixel_area_cache=None):
    """
    Run the full fire detection pipeline for one observation timestamp.

    Parameters
    ----------
    dtime : datetime.datetime
        Observation time (UTC).
    pixel_area_cache : str or Path, optional
        Path to pixel-area .npy cache file (shared across timestamps).

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Fire detections with FRP and brightness temperatures.
        Empty GeoDataFrame if no fires were confirmed.
    cube : dict
        Data cube (useful for visualizations of a selected timestamp).
    confirmed : numpy.ndarray (bool)
        Confirmed fire mask.
    frp : numpy.ndarray
        FRP values (MW), zero outside fire pixels.
    lons, lats : numpy.ndarray
        Coordinate grids.
    """
    dtstr = dtime.strftime("%Y%m%d%H%M")
    fdhsi_zip = DATA_DIR / f"MTG_FDHSI_POL_{dtstr}.zip"
    hrfi_zip = DATA_DIR / f"MTG_HRFI_POL_{dtstr}.zip"

    # Per-timestamp temp subdirectory avoids filename collisions
    ts_temp = TEMP_DIR / dtstr
    ts_temp.mkdir(exist_ok=True)

    # 1. Extract bands
    fdhsi_files = extract_bands_from_zip(fdhsi_zip, list(FDHSI_BANDS.keys()), ts_temp)
    hrfi_files = extract_bands_from_zip(hrfi_zip, list(HRFI_BANDS.keys()), ts_temp)

    ref_file = list(hrfi_files.values())[0]
    target_shape = read_band_as_masked_array(ref_file).shape
    out_bounds = get_raster_extent(ref_file)

    # 2. Build data cube
    cube = build_data_cube(fdhsi_files, hrfi_files, target_shape, out_bounds)

    # 3. Solar zenith angle → day/night mask
    lons, lats = get_lat_lon_grids(ref_file, pixel_center=True)
    sza = sun_zenith_angle(dtime, lons, lats)
    mask_night = sza > 85.0

    # 4. Cloud mask (threshold-based, derived from the data cube)
    cloud_mask = compute_cloud_mask(cube)
    print(f"  Cloud mask: {100*np.mean(cloud_mask):.1f}% cloudy, "
          f"night: {100*np.mean(mask_night):.1f}%, "
          f"MWIR max (unmasked): {float(np.nanmax(cube['MWIR_BT'].filled(np.nan))):.1f} K")

    # 5. Apply exclusion mask (night + cloud)
    exclusion = np.logical_or(mask_night, cloud_mask)
    cube_day = apply_exclusion_mask(cube, exclusion)

    # 6. Potential hotspots
    pot_fires = detect_potential_hotspots(cube_day, MWIR_THRESHOLD, MWIR_TIR_DIFF_THRESHOLD)
    n_pot = int(np.sum(pot_fires))
    mwir_arr = cube_day["MWIR_BT"].filled(np.nan)
    tir_arr = cube_day["TIR_11_BT"].filled(np.nan)
    diff_arr = mwir_arr - tir_arr
    idx = np.unravel_index(np.nanargmax(mwir_arr), mwir_arr.shape)
    print(f"    MWIR max (after excl): {np.nanmax(mwir_arr):.1f} K at {idx}, "
          f"TIR there: {tir_arr[idx]:.1f} K, diff: {diff_arr[idx]:.1f} K")
    print(f"    Max MWIR-TIR diff: {np.nanmax(diff_arr):.1f} K")
    print(f"    pot_fires type: {type(pot_fires)}, sum: {np.sum(pot_fires)}, "
          f"at hot pixel: {pot_fires[idx]}")
    # Direct check at the hot pixel
    mwir_val = cube_day["MWIR_BT"][idx]
    tir_val = cube_day["TIR_11_BT"][idx]
    print(f"    Direct check: MWIR[{idx}]={mwir_val} (masked={np.ma.is_masked(mwir_val)}), "
          f"TIR={tir_val} (masked={np.ma.is_masked(tir_val)})")

    empty_gdf = gpd.GeoDataFrame(
        columns=["LATITUDE", "LONGITUDE", "ACQ_DATE", "ACQ_TIME", "DATETIME",
                 "SATELLITE", "INSTRUMENT", "BRIGHT_MIR", "BRIGHT_TIR",
                 "MIR_TIR_DIFF", "FRP_MW", "geometry"],
        geometry="geometry", crs="EPSG:4326",
    )

    if n_pot == 0:
        _cleanup_temp(ts_temp)
        return empty_gdf, cube, np.zeros(target_shape, dtype=bool), np.zeros(target_shape), lons, lats

    # 7. Confirm hotspots
    confirmed = confirm_hotspots(
        cube_day, pot_fires, CONTEXT_WINDOW,
        c1=C1_ABSOLUTE_MWIR, c2=C2_DIFF_MAD_MULT,
        c3=C3_DIFF_ABSOLUTE, c4=C4_MWIR_MAD_MULT,
    )
    n_confirmed = int(np.sum(confirmed))

    if n_confirmed == 0:
        _cleanup_temp(ts_temp)
        return empty_gdf, cube, confirmed, np.zeros(target_shape), lons, lats

    # 8. Compute FRP
    gt = get_geotransform(ref_file)
    proj_wkt = get_projection_wkt(ref_file)
    pixel_area = get_pixel_area_grid(gt, target_shape, proj_wkt, cache_file=pixel_area_cache)

    mir_bg_mean, _ = compute_background_stats(cube_day["MWIR_BT"], confirmed, CONTEXT_WINDOW)
    frp = np.where(
        confirmed,
        compute_frp_modis(cube_day["MWIR_BT"].filled(0), mir_bg_mean, pixel_area.filled(0)),
        0.0,
    )

    # 9. Build GeoDataFrame
    fire_lats = lats[confirmed]
    fire_lons = lons[confirmed]
    fire_frp = frp[confirmed]
    fire_mwir = cube_day["MWIR_BT"].filled(0)[confirmed]
    fire_tir = cube_day["TIR_11_BT"].filled(0)[confirmed]

    df = pd.DataFrame({
        "LATITUDE": fire_lats,
        "LONGITUDE": fire_lons,
        "ACQ_DATE": dtime.strftime("%Y-%m-%d"),
        "ACQ_TIME": dtime.strftime("%H:%M"),
        "DATETIME": dtime,
        "SATELLITE": "MTG-1",
        "INSTRUMENT": "FCI",
        "BRIGHT_MIR": np.round(fire_mwir, 2),
        "BRIGHT_TIR": np.round(fire_tir, 2),
        "MIR_TIR_DIFF": np.round(fire_mwir - fire_tir, 2),
        "FRP_MW": np.round(fire_frp, 2),
    })
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE), crs="EPSG:4326"
    )

    _cleanup_temp(ts_temp)
    return gdf, cube, confirmed, frp, lons, lats


def _cleanup_temp(temp_dir):
    """Remove extracted .tif files from a per-timestamp temp subdirectory."""
    for f in temp_dir.glob("*.tif"):
        f.unlink()
    try:
        temp_dir.rmdir()
    except OSError:
        pass


# %% =========================================================================
#    VISUALIZATION FUNCTIONS
# =============================================================================

def plot_band(arr, title, cmap="gray", vmin=None, vmax=None,
              cbar_label=None, figsize=(10, 8)):
    """Plot a 2D array with colorbar."""
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.axis("off")
    plt.colorbar(im, ax=ax, label=cbar_label or "", shrink=0.7)
    plt.tight_layout()
    plt.show()
    return fig


def plot_fire_overlay(background, fire_mask, title, frp_values=None,
                      vmin=270, vmax=320, figsize=(12, 10)):
    """Overlay fire detections (optionally coloured by FRP) on a background image."""
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.imshow(background, cmap="gray", vmin=vmin, vmax=vmax)

    if frp_values is not None and np.any(fire_mask):
        fire_display = np.where(fire_mask, frp_values, np.nan)
        im = ax.imshow(
            np.ma.masked_invalid(fire_display), cmap="YlOrRd", vmin=0,
            vmax=max(float(np.nanpercentile(frp_values[fire_mask], 95)), 1.0),
        )
        plt.colorbar(im, ax=ax, label="FRP [MW]", shrink=0.7)
    else:
        overlay = np.ma.masked_where(~fire_mask, np.ones_like(fire_mask, float))
        ax.imshow(overlay, cmap="autumn_r", alpha=0.85)

    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    return fig


def plot_ncc(ncc_arr, title, figsize=(12, 10)):
    """Plot natural colour composite with contrast stretching."""
    ncc = ncc_arr.copy().astype(float)
    for i in range(3):
        band = ncc[:, :, i]
        valid = band[band > 0]
        if valid.size:
            p2, p98 = np.nanpercentile(valid, [2, 98])
            ncc[:, :, i] = np.clip((band - p2) / max(p98 - p2, 1e-9), 0, 1)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.imshow(ncc)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    return fig


def plot_frp_timeseries(timestamps, frp_total, frp_biebrza, figsize=(13, 5)):
    """
    Plot total scene FRP and Biebrza National Park FRP over time.

    Parameters
    ----------
    timestamps : list of datetime.datetime
    frp_total : list of float
        Total FRP (MW) across the whole scene at each timestamp.
    frp_biebrza : list of float
        FRP (MW) within the Biebrza bounding box at each timestamp.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150)

    for ax, vals, label, color, title in [
        (axes[0], frp_total, "Total scene FRP [MW]", "#d62728",
         "Total Scene Fire Radiative Power"),
        (axes[1], frp_biebrza, "Biebrza NP FRP [MW]", "#1f77b4",
         "Biebrza National Park FRP"),
    ]:
        ax.plot(timestamps, vals, marker="o", linewidth=2,
                color=color, markerfacecolor="white", markeredgewidth=2)
        ax.fill_between(timestamps, vals, alpha=0.15, color=color)
        ax.set_xlabel("Time (UTC)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_ylim(bottom=0)

    fig.suptitle("MTG-FCI Fire Radiative Power — Poland, 2025-04-20",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
    return fig


def plot_cumulative_frp(timestamps, frp_biebrza, figsize=(10, 5)):
    """
    Plot cumulative FRP (sum over time) for the Biebrza National Park.

    Each 10-minute FRP value is multiplied by the time step (converted to
    seconds) to obtain cumulative Fire Radiative Energy (FRE) in MJ.

    Parameters
    ----------
    timestamps : list of datetime.datetime
    frp_biebrza : list of float
        FRP (MW) within the Biebrza bounding box at each timestamp.
    """
    # Compute Fire Radiative Energy: FRE [MJ] = FRP [MW] × Δt [s]
    dt_seconds = 10 * 60   # 10-minute revisit → 600 s
    fre = [frp * dt_seconds for frp in frp_biebrza]   # MJ per step
    cumulative_fre = np.cumsum(fre)                    # cumulative MJ

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.plot(timestamps, cumulative_fre / 1e3, marker="o", linewidth=2,
            color="#2ca02c", markerfacecolor="white", markeredgewidth=2)
    ax.fill_between(timestamps, cumulative_fre / 1e3, alpha=0.15, color="#2ca02c")
    ax.set_xlabel("Time (UTC)", fontsize=11)
    ax.set_ylabel("Cumulative FRE [GJ]", fontsize=11)
    ax.set_title("Cumulative Fire Radiative Energy — Biebrza National Park\n"
                 "MTG-FCI, 2025-04-20",
                 fontweight="bold", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()
    return fig


def build_interactive_map(gdf_all):
    """
    Build a leafmap interactive map of all fire detections across all timestamps.

    Fires are colour-coded by acquisition time and sized by FRP.

    Parameters
    ----------
    gdf_all : geopandas.GeoDataFrame
        All confirmed fire detections (all timestamps combined).

    Returns
    -------
    leafmap.Map
    """
    from shapely.geometry import box

    m = leafmap.Map(center=[52.5, 20.5], zoom=7)
    m.add_basemap("Esri.WorldImagery")

    # All fires
    if not gdf_all.empty:
        m.add_gdf(
            gdf_all,
            layer_name="Fire Detections (all timestamps)",
            style={"color": "red", "fillColor": "orange",
                   "fillOpacity": 0.65, "radius": 4, "weight": 1},
        )

    # Biebrza subset highlighted
    gdf_biebrza = gdf_all[
        (gdf_all.LONGITUDE >= BIEBRZA_BBOX["lon_min"]) &
        (gdf_all.LONGITUDE <= BIEBRZA_BBOX["lon_max"]) &
        (gdf_all.LATITUDE >= BIEBRZA_BBOX["lat_min"]) &
        (gdf_all.LATITUDE <= BIEBRZA_BBOX["lat_max"])
    ]
    if not gdf_biebrza.empty:
        m.add_gdf(
            gdf_biebrza,
            layer_name="Biebrza National Park Fires",
            style={"color": "darkred", "fillColor": "red",
                   "fillOpacity": 0.9, "radius": 6, "weight": 2},
        )

    # Biebrza bounding box
    biebrza_poly = gpd.GeoDataFrame(
        geometry=[box(BIEBRZA_BBOX["lon_min"], BIEBRZA_BBOX["lat_min"],
                      BIEBRZA_BBOX["lon_max"], BIEBRZA_BBOX["lat_max"])],
        crs="EPSG:4326",
    )
    m.add_gdf(
        biebrza_poly,
        layer_name="Biebrza National Park (AOI)",
        style={"color": "#2ca02c", "fillOpacity": 0.04, "weight": 2},
    )

    return m


# %% =========================================================================
#    MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  MTG-FCI Fire Detection — Poland, 2025-04-20")
    print("=" * 70)

    # ---- Discover all available timestamps ----
    timestamps = discover_timestamps(DATA_DIR)
    if not timestamps:
        print("No complete timestamp sets found in data directory.")
        return
    print(f"\n  Found {len(timestamps)} timestamps:")
    for t in timestamps:
        print(f"    {t.strftime('%Y-%m-%d %H:%M UTC')}")

    # Shared pixel-area cache (grid is the same for all timestamps)
    pixel_area_cache = str(TEMP_DIR / "pixel_area_POL_MTG.npy")

    # ---- Process each timestamp ----
    all_gdfs = []
    frp_total_ts = []       # total scene FRP per timestamp
    frp_biebrza_ts = []     # Biebrza FRP per timestamp

    # Keep results of the first detected fire timestamp for visualizations
    vis_dtime = None
    vis_cube = None
    vis_confirmed = None
    vis_frp = None
    vis_lons = None
    vis_lats = None

    print()
    for dtime in timestamps:
        print(f"[{dtime.strftime('%H:%M')}] Processing...", end=" ", flush=True)

        gdf, cube, confirmed, frp, lons, lats = process_single_datetime(
            dtime, pixel_area_cache=pixel_area_cache
        )

        n_fires = len(gdf)
        total_frp = float(np.nansum(frp[confirmed])) if n_fires > 0 else 0.0

        biebrza_mask = (
            (lons >= BIEBRZA_BBOX["lon_min"]) & (lons <= BIEBRZA_BBOX["lon_max"]) &
            (lats >= BIEBRZA_BBOX["lat_min"]) & (lats <= BIEBRZA_BBOX["lat_max"])
        )
        biebrza_fires = np.logical_and(confirmed, biebrza_mask)
        biebrza_frp = float(np.nansum(frp[biebrza_fires])) if n_fires > 0 else 0.0

        print(f"{n_fires:4d} fires | "
              f"Total FRP: {total_frp:8.1f} MW | "
              f"Biebrza FRP: {biebrza_frp:8.1f} MW")

        frp_total_ts.append(total_frp)
        frp_biebrza_ts.append(biebrza_frp)

        if not gdf.empty:
            all_gdfs.append(gdf)

        # Keep first timestamp that has fires for static visualizations
        if vis_dtime is None and n_fires > 0:
            vis_dtime = dtime
            vis_cube = cube
            vis_confirmed = confirmed
            vis_frp = frp
            vis_lons = lons
            vis_lats = lats

    # ---- Combine all detections ----
    if all_gdfs:
        gdf_all = gpd.GeoDataFrame(
            pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326"
        )
    else:
        print("\nNo fire detections across all timestamps.")
        return

    total_fires = len(gdf_all)
    print(f"\n  Total fire detections (all timestamps): {total_fires}")
    print(f"  Peak scene FRP: {max(frp_total_ts):.1f} MW "
          f"at {timestamps[frp_total_ts.index(max(frp_total_ts))].strftime('%H:%M UTC')}")
    print(f"  Peak Biebrza FRP: {max(frp_biebrza_ts):.1f} MW "
          f"at {timestamps[frp_biebrza_ts.index(max(frp_biebrza_ts))].strftime('%H:%M UTC')}")

    # ---- Save combined GeoJSON ----
    out_geojson = OUTPUT_DIR / "fire_detections_all_timestamps.geojson"
    gdf_all.to_file(out_geojson)
    print(f"\n  Saved combined detections to: {out_geojson}")

    # ================================================================
    #  VISUALIZATION
    # ================================================================
    print("\n" + "-" * 70)
    print("  STATIC VISUALIZATIONS")
    print(f"  (showing first fire timestamp: {vis_dtime.strftime('%Y-%m-%d %H:%M UTC')})")
    print("-" * 70)

    ts_label = vis_dtime.strftime("%Y-%m-%d %H:%M UTC")

    # A. MWIR Brightness Temperature
    plot_band(
        vis_cube["MWIR_BT"].filled(np.nan),
        f"MWIR Brightness Temperature (3.8 µm) — {ts_label}",
        cmap="inferno", vmin=270, vmax=340,
        cbar_label="Brightness Temperature [K]",
    )

    # B. MWIR–TIR difference
    diff = (vis_cube["MWIR_BT"] - vis_cube["TIR_11_BT"]).filled(np.nan)
    plot_band(
        diff,
        f"MWIR – TIR Brightness Temperature Difference — {ts_label}",
        cmap="RdBu_r", vmin=-15, vmax=30,
        cbar_label="ΔT [K]",
    )

    # C. Confirmed fires with FRP
    plot_fire_overlay(
        vis_cube["MWIR_BT"].filled(np.nan), vis_confirmed,
        f"Confirmed Fire Pixels (n={int(np.sum(vis_confirmed))}) — {ts_label}",
        frp_values=vis_frp,
    )

    # D. Natural colour composite
    if "NCC" in vis_cube:
        plot_ncc(
            vis_cube["NCC"],
            f"Natural Colour Composite — {ts_label}",
        )

    # E. FRP timeseries
    print("\n  FRP timeseries...")
    plot_frp_timeseries(timestamps, frp_total_ts, frp_biebrza_ts)

    # F. Cumulative FRE for Biebrza
    print("  Cumulative FRE for Biebrza National Park...")
    plot_cumulative_frp(timestamps, frp_biebrza_ts)

    # G. Interactive leafmap
    print("  Interactive fire map...")
    m = build_interactive_map(gdf_all)
    map_path = OUTPUT_DIR / "fire_map_all_timestamps.html"
    m.to_html(str(map_path))
    print(f"  Interactive map saved to: {map_path}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)

    return gdf_all, timestamps, frp_total_ts, frp_biebrza_ts


# %% Run
if __name__ == "__main__":
    results = main()
