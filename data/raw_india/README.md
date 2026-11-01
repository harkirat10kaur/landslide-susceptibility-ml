# Landslide Inventory and Environmental Feature Dataset for Himachal Pradesh, India (Version 2)

## Overview

This dataset contains a georeferenced landslide inventory and extracted environmental features for Himachal Pradesh, India. It covers 6,539 landslide records sourced from the Geological Survey of India (GSI), with 40 environmental features extracted using Google Earth Engine (GEE) at 30 m resolution, and a modelling-ready dataset with 12,409 samples (landslide and non-landslide) for landslide susceptibility mapping.

**Version 2 updates:**
- Added `landslide_features_master_6503.csv` with coordinates and explicit `sno` key (replaces `master_landslides_clean.csv` from Version 1, which had no coordinates or record ID).
- Added `landslide_modelling_dataset_12409.csv` containing both landslide and non-landslide samples with 7 standardised predictors and land-cover classes.
- Cleaned the inventory (`hp_landslide_inventory_clean.csv`): normalised district spellings, removed corrupted year values, fixed shifted columns, added cleaning notes per row.
- Added `data_dictionary.csv` documenting every column across all files.
- Added `inventory_records_without_features_36.csv` listing the 36 inventory records absent from the feature table.

**Coordinate reference system:** WGS84 (EPSG:4326), decimal degrees.

**Study area:** Himachal Pradesh, India (approx. 30.00–32.99°N, 75.74–79.80°E), covering 12 districts.

---

## Files

| File | Rows | Columns | Description |
|---|---|---|---|
| `hp_landslide_inventory_clean.csv` | 6,539 | 14 | Landslide inventory with location, district, material type, movement type, year of occurrence. All records are landslides (label = 1). Cleaned from the original GSI data. |
| `landslide_features_master_6503.csv` | 6,503 | 46 | 40 environmental features extracted at each landslide location. Includes terrain, climate, soil, vegetation, and proximity variables. Landslides only (no non-landslide samples). |
| `landslide_modelling_dataset_12409.csv` | 12,409 | 30 | Modelling-ready dataset with 6,531 landslide and 5,878 non-landslide (pseudo-absence) samples. Contains 7 standardised predictors (z-scores), 3 raw terrain variables, and ESA WorldCover land-cover classes. |
| `inventory_records_without_features_36.csv` | 36 | 14 | Inventory records for which feature extraction failed. Same columns as the inventory file. |
| `data_dictionary.csv` | 90 entries | 6 | Column-level documentation for all three main files: column name, description, units, data source, and observed value range. |

---

## How the files link

The primary key across files is `sno` (the GSI inventory serial number).

- `landslide_features_master_6503.sno` → `hp_landslide_inventory_clean.sno` (one-to-one; coordinates verified identical).
- `landslide_modelling_dataset_12409.nearest_inventory_sno` → `hp_landslide_inventory_clean.sno` (nearest-neighbour link for landslide rows; non-landslide rows have this field blank). The modelling dataset coordinates are 30 m SRTM pixel centres, not the original inventory coordinates, so the link is by proximity (all within 20.9 m).
- `inventory_records_without_features_36.sno` → the 36 `hp_landslide_inventory_clean` records where `in_feature_table = False`.

---

## Data description

### Inventory (`hp_landslide_inventory_clean.csv`)

| Column | Description |
|---|---|
| `sno` | Record ID (primary key) |
| `latitude` | Latitude (WGS84) |
| `longitude` | Longitude (WGS84) |
| `slide_name` | Name given in inventory (mostly blank) |
| `district` | District, spellings normalised (see Cleaning below) |
| `subdivision` | Sub-division / tehsil |
| `material` | Debris, Rock, Soil, or Glaciofluvial |
| `movement` | Slide, Fall, Topple, Flow, Creep, Subsidence, Composite, etc. (13 categories) |
| `year` | Year of occurrence; blank when unknown |
| `event_note` | Free-text date note from source, where available |
| `n_records_at_coord` | Number of inventory records at this exact coordinate |
| `in_feature_table` | Whether this record has a row in `landslide_features_master_6503.csv` |
| `label` | 1 (all records are landslides) |
| `flag_note` | What was changed during cleaning, if anything |

### Master features (`landslide_features_master_6503.csv`)

40 environmental features extracted at 30 m resolution using Google Earth Engine. Feature categories:

- **Terrain (SRTM 30 m):** aspect, curvature (plan, profile), elevation, hillshade, roughness, slope, TPI, TRI, valley depth
- **Hydrology (DEM-derived):** distance to river, flow accumulation (log), SPI, TWI
- **Climate:** annual rainfall, monsoon rainfall, 95th-percentile daily rainfall, max 1-day rainfall, mean annual temperature, temperature seasonality, snow cover fraction
- **Soil:** bulk density, clay content, sand content, organic carbon, soil moisture
- **Vegetation:** NDVI median, NDVI std. dev., EVI median, BSI, forest loss flag
- **Land use / proximity:** LULC class, distance to agriculture, distance to urban, LST, nightlights, population density
- **Seismic proxy:** Vs30
- **Geomorphology:** geomorphon class, erosion risk index

Added columns: `n_records_at_coord` (co-located record count) and `duplicate_feature_vector` (True if another row has an identical 40-feature vector).

Full column details including units, sources, and value ranges are in `data_dictionary.csv`.

### Modelling dataset (`landslide_modelling_dataset_12409.csv`)

| Column | Description |
|---|---|
| `sample_id` | Row number (1–12,409) |
| `gee_system_index` | Original GEE system:index |
| `longitude`, `latitude` | Coordinates of the sampled 30 m pixel centre (WGS84) |
| `class` | 1 = landslide, 0 = non-landslide (pseudo-absence) |
| `Aspect_z` to `Slope_z` | 7 predictors, z-score standardised (mean = 0, SD = 1 over all 12,409 rows) |
| `aspect_deg`, `elevation_m`, `slope_deg` | Raw values of 3 terrain predictors |
| `LULC_10` to `LULC_100` | One-hot encoded ESA WorldCover v100 (2020) land-cover classes |
| `lulc_code`, `lulc_name` | Land-cover class code and name |
| `nearest_inventory_sno` | For landslide rows: `sno` of the nearest inventory record |
| `match_dist_m` | For landslide rows: distance to that inventory record (max 20.9 m) |
| `n_samples_at_coord` | Number of rows in this file at the same coordinate |

**Predictors (z-scored):** Aspect, Distance to River, Elevation, Flow Accumulation, NDVI, Rainfall, Slope.

**Land-cover classes (ESA WorldCover v100, 2020):** Tree cover (10), Shrubland (20), Grassland (30), Cropland (40), Built-up (50), Bare / sparse vegetation (60), Snow and ice (70), Permanent water bodies (80), Herbaceous wetland (90), Moss and lichen (100).

**Non-landslide sampling:** Random points generated inside the convex hull of landslide points buffered by 5 km, using Google Earth Engine `ee.FeatureCollection.randomPoints` with seed 42. No exclusion buffer was applied around known landslide locations.

**Feature extraction:** GEE `sampleRegions` at 30 m scale. Data sources: SRTM (`USGS/SRTMGL1_003`), CHIRPS (`UCSB-CHG/CHIRPS/DAILY`), Landsat NDVI, HydroSHEDS (distance to river, flow accumulation), ESA WorldCover v100 (`ESA/WorldCover/v100/2020`).

---

## Cleaning applied to the inventory

No rows were removed. All changes are documented in the `flag_note` column (16 rows have notes).

- **District spellings normalised:** Sirmur/Sirmour → Sirmaur; SHIMLA → Shimla; Lahul & Spiti/Lahaul & Spiti → Lahaul and Spiti. One record with district "Nathpa" set to Kinnaur based on subdivision Nichar; one record with district "." set to blank.
- **Year field:** The source used 0 for "unknown" in 5,886 rows; these are now blank. Five corrupted Excel dates (e.g. "7/15/1905") were blanked. Where a 4-digit year appeared in the event note, it was copied to `year`. Only 483 records have a known year (range: 1970–2023).
- **Material and movement:** Capitalisation unified. About 15 rows had shifted or merged columns in the source; these were realigned and each is described in `flag_note`.
- **Dropped column:** The constant `state` column (all rows = "Himachal Pradesh") was removed.

---

## Known limitations

1. **Duplicate and co-located records.** 992 inventory records share coordinates with at least one other record (up to 15 records at a single coordinate). In the master features file, 986 rows share coordinates and 1,024 rows have identical feature vectors. In the modelling dataset, 1,081 landslide rows share a pixel with another landslide row. These are not removed. Users should de-duplicate or use spatial grouping strategies as appropriate.
2. **36 inventory records have no features** (listed in `inventory_records_without_features_36.csv`). The reason is not recorded in the original processing (likely failed or masked GEE extraction).
3. **Non-landslide points are pseudo-absences**, not verified stable ground. Some may coincide with unrecorded landslide locations.
4. **Column units in the master file** follow the column name suffixes (`_mm`, `_pct`, `_g_kg`, `_c`, `_k`, `_m`). Data sources marked "(likely)" in `data_dictionary.csv` are inferred from column names and should be verified by the user against the original extraction scripts.
5. **SPI (stream power index)** is extremely right-skewed (max ≈ 495,757). Log-transformation is recommended before use.
6. **No lithology, fault distance, or seismic intensity layers** are included.
7. **The modelling dataset contains only z-scored values** for 4 of the 7 predictors (Distance to River, Flow Accumulation, NDVI, Rainfall). Raw values are provided only for Aspect, Elevation, and Slope.

---

## Version history

| Version | Date | Changes |
|---|---|---|
| 1 | July 2026 | Initial release: `hp_landslides.csv` and `master_landslides_clean.csv` |
| 2 | October 2026 | Cleaned inventory; added coordinates and `sno` key to master features; added modelling dataset with non-landslide samples; added data dictionary and README |

---