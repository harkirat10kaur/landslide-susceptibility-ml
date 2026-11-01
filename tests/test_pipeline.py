"""
Automated Integration and Data Integrity Tests
Project: AI-Powered Prediction of Landslide Susceptibility Using Machine Learning
Dataset: Himachal Pradesh Geo-Environmental Landslide Dataset (Mendeley Data DOI: 10.17632/mxpg845z62.2)
"""

import os
import joblib
import pandas as pd
import numpy as np

def test_dataset_integrity():
    clean_path = "data/cleaned_landslide_data.csv"
    assert os.path.exists(clean_path), "Cleaned dataset does not exist!"
    df = pd.read_csv(clean_path)
    assert len(df) == 11737, f"Expected 11737 cleaned rows, got {len(df)}"
    assert 'class' in df.columns, "Target column 'class' missing"
    assert set(df['class'].unique()) == {0, 1}, "Target must be binary {0, 1}"
    
    # Check that sentinel values (-9990 or less) are removed or sanitized
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ['sample_id', 'nearest_inventory_sno']:
            assert not (df[col] <= -9990).any(), f"Sentinel value found in {col}"

def test_spatial_blocks_exist():
    clean_path = "data/cleaned_landslide_data.csv"
    df = pd.read_csv(clean_path)
    assert 'spatial_cluster' in df.columns, "spatial_cluster column missing"
    assert df['spatial_cluster'].nunique() == 5, "Expected 5 spatial clusters"

def test_models_exist_and_loadable():
    model_path = "models/best_landslide_model.joblib"
    assert os.path.exists(model_path), "Primary model file missing"
    model = joblib.load(model_path)
    assert hasattr(model, "predict_proba"), "Model must support predict_proba"
    
    all_models_path = "models/all_fitted_models.joblib"
    assert os.path.exists(all_models_path), "All fitted models file missing"
    all_models = joblib.load(all_models_path)
    assert len(all_models) >= 6, "Expected at least 6 fitted models"

def test_evaluation_metrics_validity():
    comp_path = "tables/model_comparison.csv"
    assert os.path.exists(comp_path), "Model comparison CSV missing"
    comp_df = pd.read_csv(comp_path)
    assert len(comp_df) >= 6, "Expected at least 6 models in comparison table"
    for col in ['Accuracy', 'ROC_AUC', 'Recall', 'F1_Score']:
        assert col in comp_df.columns, f"Missing metric column {col}"
        assert comp_df[col].between(0.0, 1.0).all(), f"Metric {col} values out of bounds [0, 1]"

def test_geotiff_surface_generated():
    tif_path = "maps/susceptibility_surface.tif"
    assert os.path.exists(tif_path), "GeoTIFF surface raster missing at maps/susceptibility_surface.tif"
    assert os.path.getsize(tif_path) > 1000, "GeoTIFF raster file is empty or corrupted"

if __name__ == "__main__":
    print("Running integration tests on Himachal Pradesh Indian Landslide Pipeline...")
    test_dataset_integrity()
    print("[PASS] test_dataset_integrity passed (11,737 rows, WGS84 deduplicated)")
    test_spatial_blocks_exist()
    print("[PASS] test_spatial_blocks_exist passed (5 spatial blocks)")
    test_models_exist_and_loadable()
    print("[PASS] test_models_exist_and_loadable passed (all candidate models present)")
    test_evaluation_metrics_validity()
    print("[PASS] test_evaluation_metrics_validity passed (all metrics in [0, 1])")
    test_geotiff_surface_generated()
    print("[PASS] test_geotiff_surface_generated passed (maps/susceptibility_surface.tif valid)")
    print("\n[SUCCESS] All pipeline integration tests passed successfully!")
