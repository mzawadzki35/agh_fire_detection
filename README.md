# Fire Detection with MTG-FCI Satellite Imagery

**AGH University of Krakow — Aerospace Engineering Challenge**
*Author: Valerio Pampanoni — [EOSIAL](https://eosial.uniroma1.it/), Sapienza University of Rome*

---

## Overview

This repository implements an **active fire detection pipeline** applied to data from the **MTG-FCI** (Meteosat Third Generation — Flexible Combined Imager) satellite, Europe's newest generation of geostationary meteorological imagers.

The case study is the **Biebrza National Park wildfire** (Poland, 20 April 2025), one of the most significant wildfire events in recent Polish history. Thirteen consecutive observations at 10-minute intervals (12:00–14:00 UTC) are processed to track fire radiative power over time.

The algorithm is a simplified adaptation of the **MODIS Collection 5 contextual fire detection** approach (Giglio et al., 2003), reimplemented for MTG-FCI bands.

---

## Key Results

| Metric | Value |
|---|---|
| Timestamps processed | 13 (12:00 – 14:00 UTC, 10 min cadence) |
| Total fire detections | 28 pixels |
| Peak scene FRP | 198.8 MW (13:20 UTC) |
| Peak Biebrza NP FRP | 198.8 MW (13:20 UTC) |

---

## Algorithm

### Detection

1. **MWIR threshold** — flag pixels with 3.8 µm brightness temperature > 310 K
2. **MWIR–TIR difference** — flag pixels where BT(3.8 µm) − BT(10.5 µm) > 10 K
3. **Solar zenith mask** — exclude nighttime pixels (SZA > 85°)
4. **Cloud mask** — exclude cloudy pixels using red reflectance, NIR 1.3 µm, and TIR 12.3 µm thresholds

### Contextual Confirmation

Potential fire pixels are confirmed against local background statistics within a 21×21 pixel window, using four criteria adapted from the MODIS C5 algorithm:

- **C1** — absolute MWIR BT > 360 K
- **C2** — MWIR–TIR difference > background mean + 3 σ
- **C3** — MWIR–TIR difference > background mean + 5.5 K
- **C4** — MWIR BT > background mean + 2.5 σ

### Fire Radiative Power (FRP)

FRP is estimated per pixel using the MODIS radiance-based equation (Wooster et al., 2005):

$$\text{FRP} = A_{\text{pixel}} \cdot \sigma \cdot \left( T_{\text{fire}}^4 - T_{\text{bg}}^4 \right)$$

where $A_{\text{pixel}}$ is the pixel area in m², $\sigma$ is the Stefan–Boltzmann constant, and $T_{\text{bg}}$ is the local background brightness temperature.

---

## MTG-FCI Bands Used

| Band | Wavelength | Tag | Resolution | Use |
|---|---|---|---|---|
| `vis_06` | 0.64 µm | `RED_REF` | ~2 km | Cloud masking |
| `vis_08` | 0.86 µm | `NIR_08_REF` | ~2 km | Cloud masking |
| `nir_13` | 1.3 µm | `NIR_13_REF` | ~2 km | Cirrus / cloud |
| `nir_22` | 2.2 µm | `SWIR_22_REF` | ~2 km | Fire/smoke |
| `ir_123` | 12.3 µm | `TIR_12_BT` | ~2 km | Cloud tops |
| `natural_color` | RGB | `NCC` | ~2 km | Visualization |
| `ir_38` | 3.8 µm | `MWIR_BT` | ~1 km | **Primary fire band** |
| `ir_38_rad` | 3.8 µm | `MWIR_RAD` | ~1 km | FRP estimation |
| `ir_105` | 10.5 µm | `TIR_11_BT` | ~1 km | Background / cloud |

FDHSI bands (~2 km) are resampled to the HRFI grid (~1 km) using nearest-neighbour interpolation. Detection and FRP estimation run at the full HRFI resolution.

---

## Repository Structure

```
agh_fire_detection/
├── notebooks/
│   ├── fire_detection.ipynb          # Main solution notebook (interactive)
│   ├── fire_detection_challenge.ipynb # Student challenge template
│   └── logos/                         # Institution logos for notebook header
├── aoi/
│   └── biebrza_national_park.geojson  # Area of interest boundary
├── data/                              # MTG-FCI zip archives (not tracked — see below)
│   ├── MTG_FDHSI_POL_<timestamp>.zip
│   ├── MTG_HRFI_POL_<timestamp>.zip
│   └── MTG_CLM_POL_<timestamp>.zip
├── output/                            # Generated outputs (not tracked)
│   └── fire_detections_all_timestamps.geojson
├── temp/                              # Temporary extracted GeoTIFFs (not tracked)
├── fire_detection.py                  # Standalone Python script (same pipeline)
├── environment.yml                    # Conda environment specification
├── requirements.txt                   # pip requirements (GDAL via conda only)
└── README.md
```

---

## Installation

### Recommended: Conda (ensures GDAL C bindings)

```bash
conda env create -f environment.yml
conda activate fire_detection
```

### Alternative: pip

GDAL **must** be installed via conda or system package manager — pip wheels are unreliable across platforms.

```bash
conda install -c conda-forge gdal
pip install -r requirements.txt
```

---

## Data

The `data/` directory is not tracked in this repository due to file size (~150 MB). It must contain three zip archives per timestamp, named:

```
MTG_FDHSI_POL_YYYYMMDDHHMM.zip   # Full-Disc High Spectral resolution Imagery
MTG_HRFI_POL_YYYYMMDDHHMM.zip    # High Resolution Fast Imagery
MTG_CLM_POL_YYYYMMDDHHMM.zip     # Cloud Mask
```

Each zip is expected to contain band-specific GeoTIFF files following the naming convention produced by the **EUMETSAT Data Store** MTG-FCI Level 1c processor.

Data for this case study (13 timestamps, 20 April 2025, 12:00–14:00 UTC, Poland subset) was provided as part of the AGH University of Krakow Aerospace Engineering challenge.

---

## Usage

### Jupyter Notebook (recommended)

```bash
conda activate fire_detection
jupyter lab notebooks/fire_detection.ipynb
```

The notebook is fully interactive:

- **Section 6 — Static Visualizations**: dropdown to select any processed timestamp, radio button to switch between full-image and Biebrza NP zoom, checkboxes to choose which band plots to display.
- **Section 8 — Interactive Fire Map**: dropdown to filter fire detections by timestamp on a Leaflet map (Esri World Imagery basemap), zoomed to Biebrza NP.

### Standalone Script

```bash
conda activate fire_detection
python fire_detection.py
```

Outputs are written to `output/` and displayed inline via matplotlib.

---

## References

- Giglio, L., Descloitres, J., Justice, C. O., & Kaufman, Y. J. (2003). An enhanced contextual fire detection algorithm for MODIS. *Remote Sensing of Environment*, 87(2–3), 273–282.
- Wooster, M. J., Roberts, G., Perry, G. L. W., & Kaufman, Y. J. (2005). Retrieval of biomass combustion rates and totals from fire radiative power observations: FRP derivation and calibration relationships between biomass consumption and fire radiative energy release. *Journal of Geophysical Research: Atmospheres*, 110(D24).
- EUMETSAT (2024). *MTG-FCI Level 1c Product User Guide*. EUM/MTG/USR/21/1198545.
