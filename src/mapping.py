"""
Susceptibility Mapping and Five-Tier Classification Module
Project: AI-Powered Prediction of Landslide Susceptibility Using Machine Learning
Dataset: Himachal Pradesh Geo-Environmental Landslide Dataset (Mendeley Data DOI: 10.17632/mxpg845z62.2)
Calculates S(x) across Random Forest trees, partitions into 5 relative tiers,
generates publication-quality 300 DPI maps, and exports GeoTIFF raster.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.interpolate import Rbf
import tifffile

# Styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Standard 5-tier landslide susceptibility palette
CLASS_COLORS = {
    'Very Low': '#1a9641',   # Dark Green
    'Low': '#a6d96a',        # Light Green
    'Moderate': '#ffffbf',   # Soft Yellow
    'High': '#fdae61',       # Orange
    'Very High': '#d7191c'   # Red
}

def generate_susceptibility_outputs():
    print("="*60)
    print("  HIMACHAL PRADESH SUSCEPTIBILITY ZONING & GEOSPATIAL MAPS")
    print("="*60)
    
    os.makedirs("figures", exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("maps", exist_ok=True)
    
    # 1. Load Data & Fitted Model
    df = pd.read_csv("data/cleaned_landslide_data.csv")
    pipe = joblib.load("models/best_landslide_model.joblib")
    
    with open("models/feature_metadata.json", "r") as f:
        meta = json.load(f)
    feature_cols = meta['all_features']
    target_col = meta.get('target_col', 'class')
    
    X = df[feature_cols]
    y = df[target_col].values
    
    # 2. Compute S(x) = mean probability across Random Forest trees
    rf_probs = pipe.predict_proba(X)[:, 1]
    df['susceptibility_score'] = rf_probs
    
    # 3. Establish 5-Class Susceptibility Thresholds using Quantiles (Equal-frequency relative risk tiers)
    q20, q40, q60, q80 = np.percentile(rf_probs, [20, 40, 60, 80])
    thresholds = {
        'Very Low': [0.0, float(q20)],
        'Low': [float(q20), float(q40)],
        'Moderate': [float(q40), float(q60)],
        'High': [float(q60), float(q80)],
        'Very High': [float(q80), 1.0]
    }
    
    with open("models/susceptibility_thresholds.json", "w") as f:
        json.dump({
            'region': 'Himachal Pradesh, India',
            'method': 'Quantile Natural Intervals (Quintiles 20%, 40%, 60%, 80%)',
            'threshold_cutoffs': {
                'Q20': float(q20), 'Q40': float(q40),
                'Q60': float(q60), 'Q80': float(q80)
            },
            'classes': thresholds,
            'note': 'These tiers represent relative susceptibility across Himachal Pradesh, not absolute deterministic collapse probabilities.'
        }, f, indent=2)
    print(f"[+] Established Quantile Cutoffs: Q20={q20:.4f}, Q40={q40:.4f}, Q60={q60:.4f}, Q80={q80:.4f}")
    
    # Assign Susceptibility Tier to each point
    def assign_tier(val):
        if val < q20:
            return 'Very Low'
        elif val < q40:
            return 'Low'
        elif val < q60:
            return 'Moderate'
        elif val < q80:
            return 'High'
        else:
            return 'Very High'
            
    df['susceptibility_class'] = df['susceptibility_score'].apply(assign_tier)
    
    # 4. Class Distribution & Known Landslide Share Table
    tier_order = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    stats_records = []
    total_landslides = (y == 1).sum()
    total_points = len(df)
    
    for tier in tier_order:
        sub = df[df['susceptibility_class'] == tier]
        pt_count = len(sub)
        pt_pct = (pt_count / total_points) * 100
        ls_count = (sub[target_col] == 1).sum()
        non_ls_count = (sub[target_col] == 0).sum()
        ls_share = (ls_count / total_landslides) * 100
        ls_density = (ls_count / pt_count) * 100 if pt_count > 0 else 0.0
        
        low_b, high_b = thresholds[tier]
        stats_records.append({
            'Susceptibility_Class': tier,
            'Score_Range': f"[{low_b:.3f} - {high_b:.3f}]",
            'Total_Samples': pt_count,
            'Percentage_of_Total_%': pt_pct,
            'Known_Landslides': ls_count,
            'Non_Landslides': non_ls_count,
            'Share_of_Known_Landslides_%': ls_share,
            'Within_Class_Landslide_Rate_%': ls_density
        })
        
    stats_df = pd.DataFrame(stats_records)
    stats_df.to_csv("tables/susceptibility_class_distribution.csv", index=False)
    print("[+] Saved class distribution table to tables/susceptibility_class_distribution.csv")
    print("\n=== Susceptibility Class Distribution (Himachal Pradesh) ===")
    print(stats_df.to_string(index=False))
    
    df.to_csv("data/landslide_points_with_susceptibility.csv", index=False)
    
    # 5. Susceptibility Score Distribution Histogram (300 DPI)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    band_spans = [
        (0.0, q20, CLASS_COLORS['Very Low'], 'Very Low'),
        (q20, q40, CLASS_COLORS['Low'], 'Low'),
        (q40, q60, CLASS_COLORS['Moderate'], 'Moderate'),
        (q60, q80, CLASS_COLORS['High'], 'High'),
        (q80, 1.0, CLASS_COLORS['Very High'], 'Very High')
    ]
    for low, high, col, name in band_spans:
        ax.axvspan(low, high, color=col, alpha=0.35, label=f"{name}: [{low:.2f}-{high:.2f}]")
        
    sns.histplot(df['susceptibility_score'], bins=50, kde=True, ax=ax,
                 color='#1f77b4', edgecolor='black', alpha=0.75, stat='density', linewidth=0.8)
    
    for cutoff, label in zip([q20, q40, q60, q80], ['Q20', 'Q40', 'Q60', 'Q80']):
        ax.axvline(cutoff, color='black', linestyle='--', linewidth=1.2)
        ax.text(cutoff, ax.get_ylim()[1]*0.88, f" {label}\n {cutoff:.2f}",
                fontsize=8.5, fontweight='bold', va='bottom', rotation=0)
        
    ax.set_title("Random Forest Landslide Susceptibility Score Distribution S(x)\nHimachal Pradesh, India (N=11,737)",
                 fontsize=11.5, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Landslide Susceptibility Score S(x)", fontsize=10)
    ax.set_ylabel("Empirical Density", fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.legend(loc='upper right', frameon=True, fontsize=8.5)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()
    fig.savefig("figures/susceptibility_score_distribution.png", dpi=300)
    plt.close(fig)
    print("[+] Saved score distribution plot to figures/susceptibility_score_distribution.png")
    
    # 6. High-Resolution Susceptibility Point Map of Himachal Pradesh
    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)
    for tier in tier_order:
        sub = df[df['susceptibility_class'] == tier]
        col = CLASS_COLORS[tier]
        ax.scatter(sub['longitude'], sub['latitude'], c=col, label=f"{tier} ({len(sub):,} pts, {len(sub)/total_points*100:.1f}%)",
                   s=12, alpha=0.75, edgecolors='none')
        
    ax.set_title("Landslide Susceptibility Point Zonation Map — Himachal Pradesh, India\nModel: Tuned Random Forest (Test ROC-AUC: 0.9009, N=11,737 points)",
                 fontsize=11.5, fontweight='bold', pad=12)
    ax.set_xlabel("Longitude (°E) [WGS84 / EPSG:4326]", fontsize=10.5)
    ax.set_ylabel("Latitude (°N) [WGS84 / EPSG:4326]", fontsize=10.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title="Susceptibility Tier", loc='upper left', frameon=True, fontsize=8.5, title_fontsize=9)
    
    ax.text(0.98, 0.98, "Region: Himachal Pradesh, India\nCoverage: 12 Districts (Kangra, Mandi, Shimla, etc.)\nCRS: WGS84 (EPSG:4326)\nRelative Quantile Classification",
            transform=ax.transAxes, fontsize=8.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))
    plt.tight_layout()
    fig.savefig("figures/susceptibility_map_points.png", dpi=300)
    plt.close(fig)
    print("[+] Saved point susceptibility map to figures/susceptibility_map_points.png")
    
    # 7. Spatial Interpolation & Continuous Susceptibility Surface
    print("\n[*] Interpolating Continuous Susceptibility Surface Across Himachal Pradesh Grid...")
    xi = np.linspace(df['longitude'].min() - 0.05, df['longitude'].max() + 0.05, 350)
    yi = np.linspace(df['latitude'].min() - 0.05, df['latitude'].max() + 0.05, 350)
    XI, YI = np.meshgrid(xi, yi)
    
    # Subsample 3,000 points for fast, smooth RBF interpolation
    np.random.seed(42)
    sample_sub = df.sample(min(3000, len(df)), random_state=42)
    rbf = Rbf(sample_sub['longitude'], sample_sub['latitude'], sample_sub['susceptibility_score'],
              function='multiquadric', smooth=0.15)
    ZI = rbf(XI, YI)
    ZI = np.clip(ZI, 0.0, 1.0)
    
    # Continuous Map Plot
    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)
    im = ax.imshow(ZI, extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                   origin='lower', cmap='RdYlGn_r', vmin=0, vmax=1, aspect='equal')
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Predicted Landslide Susceptibility Index S(x)", fontsize=10)
    
    # Overlay actual known landslide locations as small black dots
    ls_pts = df[df[target_col] == 1]
    ax.scatter(ls_pts['longitude'], ls_pts['latitude'], c='black', s=3, alpha=0.35, label=f'Known GSI Landslide Events (N={len(ls_pts):,})')
    
    ax.set_title("Continuous Landslide Susceptibility Surface: Himachal Pradesh, India\nSpatial Generalization of Random Forest Predictions",
                 fontsize=11.5, fontweight='bold', pad=12)
    ax.set_xlabel("Longitude (°E) [WGS84 / EPSG:4326]", fontsize=10.5)
    ax.set_ylabel("Latitude (°N) [WGS84 / EPSG:4326]", fontsize=10.5)
    ax.legend(loc='lower left', frameon=True, fontsize=8.5)
    ax.grid(True, linestyle=':', alpha=0.4, color='white')
    plt.tight_layout()
    fig.savefig("figures/susceptibility_map_continuous.png", dpi=300)
    plt.close(fig)
    print("[+] Saved continuous susceptibility map to figures/susceptibility_map_continuous.png")
    
    # 8. 5-Class Categorical Zoned Map
    ZI_5class = np.zeros_like(ZI, dtype=int)
    ZI_5class[(ZI >= q20) & (ZI < q40)] = 1
    ZI_5class[(ZI >= q40) & (ZI < q60)] = 2
    ZI_5class[(ZI >= q60) & (ZI < q80)] = 3
    ZI_5class[ZI >= q80] = 4
    
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap([CLASS_COLORS[t] for t in tier_order])
    
    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)
    im2 = ax.imshow(ZI_5class, extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                    origin='lower', cmap=custom_cmap, vmin=0, vmax=4, aspect='equal')
    
    patches = [mpatches.Patch(color=CLASS_COLORS[t], label=f"{t} [{thresholds[t][0]:.2f}-{thresholds[t][1]:.2f}]") for t in tier_order]
    ax.legend(handles=patches, title="Susceptibility Classes", loc='lower left', frameon=True, fontsize=8.5, title_fontsize=9)
    
    ax.set_title("Five-Class Landslide Hazard Susceptibility Zonation Map\nHimachal Pradesh, Western Himalayas, India",
                 fontsize=11.5, fontweight='bold', pad=12)
    ax.set_xlabel("Longitude (°E) [WGS84 / EPSG:4326]", fontsize=10.5)
    ax.set_ylabel("Latitude (°N) [WGS84 / EPSG:4326]", fontsize=10.5)
    ax.grid(True, linestyle=':', alpha=0.4, color='white')
    
    # North Arrow
    ax.annotate('N', xy=(0.06, 0.94), xytext=(0.06, 0.86), xycoords='axes fraction',
                arrowprops=dict(facecolor='black', width=3, headwidth=9),
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig("figures/susceptibility_map_5class.png", dpi=300)
    plt.close(fig)
    print("[+] Saved 5-class susceptibility map to figures/susceptibility_map_5class.png")
    
    # 9. Export Susceptibility GeoTIFF Raster using tifffile
    print("[*] Exporting GeoTIFF raster to maps/susceptibility_surface.tif...")
    raster_data = ZI.astype(np.float32)
    dx = (xi.max() - xi.min()) / (len(xi) - 1)
    
    tifffile.imwrite(
        "maps/susceptibility_surface.tif",
        raster_data,
        description=f"Landslide Susceptibility Index S(x) for Himachal Pradesh. Bounds: [{xi.min()}, {yi.min()}, {xi.max()}, {yi.max()}], WGS84 EPSG:4326",
        metadata={
            'crs': 'EPSG:4326',
            'min_lon': float(xi.min()), 'max_lon': float(xi.max()),
            'min_lat': float(yi.min()), 'max_lat': float(yi.max()),
            'resolution_deg': float(dx)
        }
    )
    print("[+] Exported GeoTIFF to maps/susceptibility_surface.tif")
    print("[+] Mapping and zonation complete!")

if __name__ == "__main__":
    generate_susceptibility_outputs()
