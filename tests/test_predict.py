"""
Automated Inference and Schema Validation Tests
Project: AI-Powered Prediction of Landslide Susceptibility Using Machine Learning
Dataset: Himachal Pradesh Geo-Environmental Landslide Dataset (Mendeley Data DOI: 10.17632/mxpg845z62.2)
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath("."))
from predict import LandslidePredictor

def test_predictor_initialization():
    predictor = LandslidePredictor()
    assert predictor.model is not None, "Model failed to load"
    assert len(predictor.all_features) == 10, f"Expected 10 features, got {len(predictor.all_features)}"
    assert 0.0 < predictor.operational_threshold < 1.0, "Operational threshold out of bounds"
    assert predictor.region == "Himachal Pradesh, India"

def test_single_prediction_valid_schema():
    predictor = LandslidePredictor()
    test_df = pd.read_csv("data/test_dataset_untouched.csv")
    sample = test_df.iloc[[0]].copy()
    
    preds = predictor.predict(sample)
    assert 'susceptibility_score' in preds.columns
    assert 'susceptibility_tier' in preds.columns
    assert 'hazard_alert_operational' in preds.columns
    assert 0.0 <= preds['susceptibility_score'].iloc[0] <= 1.0
    assert preds['susceptibility_tier'].iloc[0] in ['Very Low', 'Low', 'Moderate', 'High', 'Very High']

def test_missing_features_raises_error():
    predictor = LandslidePredictor()
    incomplete_df = pd.DataFrame({'slope_deg': [25.0], 'elevation_m': [1200.0]})
    try:
        predictor.predict(incomplete_df)
        assert False, "Should have raised ValueError for missing features"
    except ValueError as e:
        assert "missing required feature" in str(e).lower()

def test_aspect_autocomputation():
    predictor = LandslidePredictor()
    test_df = pd.read_csv("data/test_dataset_untouched.csv")
    sample = test_df.iloc[[0]].copy()
    # Drop cos and sin aspect, should be auto-derived from aspect_deg
    sample_no_trig = sample.drop(columns=['Aspect_cos', 'Aspect_sin'])
    preds = predictor.predict(sample_no_trig)
    assert 'susceptibility_score' in preds.columns
    assert not np.isnan(preds['susceptibility_score'].iloc[0])

def test_sentinel_nodata_imputation():
    predictor = LandslidePredictor()
    test_df = pd.read_csv("data/test_dataset_untouched.csv")
    sample = test_df.iloc[[0]].copy()
    # Inject sentinel value
    sample['NDVI_z'] = -9999.0
    sample['Rainfall_z'] = -9999.0
    preds = predictor.predict(sample)
    assert not np.isnan(preds['susceptibility_score'].iloc[0]), "Imputation failed for sentinel value"

if __name__ == "__main__":
    print("Running inference tests on Himachal Pradesh Predictor...")
    test_predictor_initialization()
    print("[PASS] test_predictor_initialization passed")
    test_single_prediction_valid_schema()
    print("[PASS] test_single_prediction_valid_schema passed")
    test_missing_features_raises_error()
    print("[PASS] test_missing_features_raises_error passed")
    test_aspect_autocomputation()
    print("[PASS] test_aspect_autocomputation passed (aspect_deg -> cos/sin aspect auto-computed)")
    test_sentinel_nodata_imputation()
    print("[PASS] test_sentinel_nodata_imputation passed (sentinels handled by pipeline imputer)")
    print("\n[SUCCESS] All predictor tests passed successfully!")
