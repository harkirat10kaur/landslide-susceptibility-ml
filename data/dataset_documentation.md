# Dataset Documentation: Himachal Pradesh Landslide Susceptibility

## 1. Dataset Overview & Provenance
- **Dataset Title:** A Geo-Environmental Feature Dataset for Landslide Susceptibility Modelling in Himachal Pradesh, India (Version 2)
- **Source Platform:** Mendeley Data
- **DOI:** 10.17632/mxpg845z62.2
- **URL:** https://data.mendeley.com/datasets/mxpg845z62/2
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Data Provenance:** Geological Survey of India (GSI) official landslide inventory, complemented by 30 m resolution geo-environmental variables extracted via Google Earth Engine (GEE).
- **Study Region:** Himachal Pradesh, Western Himalayas, India (approx. 30.00°–33.01°N, 75.71°–79.83°E), spanning 12 districts: Kangra, Mandi, Shimla, Kullu, Kinnaur, Solan, Sirmaur, Chamba, Bilaspur, Hamirpur, Una, and Lahaul & Spiti.
- **Coordinate Reference System:** WGS84 (EPSG:4326), decimal degrees.

## 2. Sample Counts & Target Definition
- **Raw Packaging Count:** 12,409 samples (6,531 landslides, 5,878 non-landslides).
- **Cleaned Deduplicated Count:** **11,737 unique spatial samples** (672 co-located 30 m pixel duplicates removed to prevent pixel-level leakage across folds).
  - `class = 1` (Confirmed Historical Landslide): 5,859 samples (49.92%)
  - `class = 0` (Non-Landslide Pseudo-Absence): 5,878 samples (50.08%)
- **Target Distribution:** Perfectly balanced 50:50 ratio.

## 3. Pseudo-Absence Construction & Spatial Leakage Audit
- **Methodology:** Random pseudo-absence points were generated inside the 5 km buffered convex hull of known landslides using GEE `ee.FeatureCollection.randomPoints(seed=42)`.
- **Exclusion Buffer Audit:** The dataset authors did not enforce an explicit minimum distance exclusion buffer. Our spatial distance audit reveals:
  - Median distance to nearest known landslide: **11.27 km**.
  - Mean distance: **21.27 km**.
  - Points within 100 m: **13 samples (0.22%)**.
  - Points within 500 m: **188 samples (3.20%)**.
  - Points > 1,000 m away: **5,398 samples (91.83%)**.
  - *Risk Assessment:* While 91.8% of pseudo-absences sit safely > 1 km from known landslide ruptures, approximately 3.2% fall within 500 m, presenting a minor label noise risk that is controlled through spatial block validation.

## 4. Conditioning Factors Schema
| Predictor Name | Feature Type | Physical Description | Raw Units / Standardisation |
|---|---|---|---|
| `slope_deg` | Continuous | Terrain slope gradient from SRTM 30 m | Degrees [0.0° - 76.8°] |
| `elevation_m` | Continuous | Digital elevation model from SRTM 30 m | Meters above sea level [246 m - 6,430 m] |
| `aspect_deg` | Continuous | Slope aspect from SRTM 30 m | Degrees [0° - 359°] |
| `Aspect_cos` | Continuous | Northness component cos(Aspect) | Unitless [-1.0 North, +1.0 South] |
| `Aspect_sin` | Continuous | Eastness component sin(Aspect) | Unitless [-1.0 West, +1.0 East] |
| `Dist_River_z` | Continuous | Distance to river network from HydroSHEDS | z-score [min: -0.13, max: +8.04] |
| `FlowAccumulation_z` | Continuous | Catchment flow accumulation from HydroSHEDS | z-score [min: -0.58, max: +5.01] |
| `NDVI_z` | Continuous | Median vegetation index from Landsat | z-score [min: -5.97, max: +1.80] |
| `Rainfall_z` | Continuous | Annual precipitation from CHIRPS | z-score [min: -2.33, max: +3.92] |
| `lulc_name` | Categorical | Land use / land cover from ESA WorldCover 2020 | 10 classes (Tree cover, Grassland, Bare/sparse, Cropland, Built-up, Shrubland, Snow/ice, Water, Wetland, Moss/lichen) |

## 5. Preprocessing & Leakage Safeguards
- **Re-Standardisation on Training Folds Only:** Although the raw table provided pre-standardized z-scores, all numeric predictors are passed through a scikit-learn `StandardScaler` fitted strictly on training partitions inside each cross-validation fold.
- **Categorical Encoding:** `lulc_name` is encoded via `OneHotEncoder(handle_unknown='ignore')` fitted strictly on training data.
- **Spatial Block Partitioning:** 5 geographic sectors defined via K-Means on coordinates `(longitude, latitude)` to enforce leakage-free Spatial GroupKFold validation.
