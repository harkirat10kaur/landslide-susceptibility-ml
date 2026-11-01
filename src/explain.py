"""
Feature Importance, SHAP Explainability and Feature-Group Ablation Module
Project: AI-Powered Prediction of Landslide Susceptibility Using Machine Learning
Dataset: Himachal Pradesh Geo-Environmental Landslide Dataset (Mendeley Data DOI: 10.17632/mxpg845z62.2)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import shap

# Styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def run_explainability():
    print("="*60)
    print("  EXPLAINABILITY, SHAP & ABLATION (HIMACHAL PRADESH)")
    print("="*60)
    
    # 1. Load Data, Models & Metadata
    test_df = pd.read_csv("data/test_dataset_untouched.csv")
    train_df = pd.read_csv("data/train_dataset_pool.csv")
    
    with open("models/feature_metadata.json", "r") as f:
        meta = json.load(f)
    feature_cols = meta['all_features']
    target_col = meta.get('target_col', 'class')
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values
    
    pipe = joblib.load("models/best_landslide_model.joblib")
    prep = pipe.named_steps['prep']
    rf = pipe.named_steps['clf']
    
    # Get transformed feature names
    cat_names = prep.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(meta['categorical_features'])
    transformed_feature_names = list(meta['numeric_features']) + list(cat_names)
    
    # 2. Built-in Gini / MDI Feature Importance
    print("\n[*] Computing Random Forest Gini Feature Importance...")
    mdi_importances = rf.feature_importances_
    
    # Aggregate one-hot categories back to base feature names
    feat_agg = {}
    for feat, imp in zip(transformed_feature_names, mdi_importances):
        base_name = feat.split('_')[0] if feat.startswith('lulc_name') else feat
        feat_agg[base_name] = feat_agg.get(base_name, 0.0) + imp
        
    orig_mdi_df = pd.DataFrame([
        {'Feature': k, 'Importance': v} for k, v in feat_agg.items()
    ]).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    orig_mdi_df.to_csv("tables/feature_importance_ranked.csv", index=False)
    print("[+] Saved ranked feature importance to tables/feature_importance_ranked.csv")
    print(orig_mdi_df.to_string(index=False))
    
    # Plot MDI Feature Importance Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    sorted_mdi = orig_mdi_df.sort_values(by="Importance", ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(sorted_mdi)))
    bars = ax.barh(sorted_mdi['Feature'], sorted_mdi['Importance'], color=colors, edgecolor='black', alpha=0.85)
    ax.set_title("Random Forest Gini Feature Importance (Himachal Pradesh Inventory)", fontsize=11.5, fontweight='bold', pad=12)
    ax.set_xlabel("Relative Importance Weight (Mean Decrease in Impurity)", fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f'{w:.3f}', xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", ha='left', va='center', fontsize=8.5)
    plt.tight_layout()
    fig.savefig("figures/rf_feature_importance.png", dpi=300)
    plt.close(fig)
    print("[+] Saved MDI chart to figures/rf_feature_importance.png")
    
    # 3. Permutation Importance on Untouched Test Set (50 repeats)
    print("\n[*] Computing Permutation Importance on Untouched Test Set (50 repeats)...")
    perm_res = permutation_importance(
        pipe, X_test, y_test, scoring='roc_auc', n_repeats=50, random_state=42, n_jobs=-1
    )
    perm_df = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_Importance_Drop': perm_res.importances_mean,
        'Std_Importance_Drop': perm_res.importances_std
    }).sort_values(by="Mean_Importance_Drop", ascending=False).reset_index(drop=True)
    perm_df.to_csv("tables/permutation_importance.csv", index=False)
    print("[+] Saved permutation importance to tables/permutation_importance.csv")
    print(perm_df.to_string(index=False))
    
    # Plot Permutation Importance Chart
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    sorted_perm = perm_df.sort_values(by="Mean_Importance_Drop", ascending=True)
    ax.barh(sorted_perm['Feature'], sorted_perm['Mean_Importance_Drop'],
            xerr=sorted_perm['Std_Importance_Drop'], color='#2b83ba', edgecolor='black', alpha=0.85, capsize=3)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title("Test-Set Permutation Feature Importance (Himachal Pradesh)", fontsize=11.5, fontweight='bold', pad=12)
    ax.set_xlabel("Mean Test ROC-AUC Decrease Upon Feature Shuffling", fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig("figures/rf_permutation_importance.png", dpi=300)
    plt.close(fig)
    print("[+] Saved permutation chart to figures/rf_permutation_importance.png")
    
    # 4. SHAP Explainability (TreeExplainer on Subsample for Speed & Stability)
    print("\n[*] Calculating SHAP Explanations on Test Set...")
    # Use 500 representative test samples for clean SHAP calculation
    sample_size = min(500, len(X_test))
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=sample_size, replace=False)
    X_test_sub = X_test.iloc[sample_indices]
    
    X_test_trans = prep.transform(X_test_sub)
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_test_trans)
    
    if isinstance(shap_vals, list):
        shap_vals_class1 = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_vals_class1 = shap_vals[:, :, 1]
    else:
        shap_vals_class1 = shap_vals
        
    # SHAP Summary Beeswarm Plot
    fig = plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(
        shap_vals_class1, X_test_trans, feature_names=transformed_feature_names,
        max_display=12, show=False
    )
    plt.title("SHAP Beeswarm Summary Plot: Conditioning Factors Impact (Himachal Pradesh)", fontsize=11.5, fontweight='bold', pad=14)
    plt.tight_layout()
    fig.savefig("figures/shap_summary_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[+] Saved SHAP summary plot to figures/shap_summary_beeswarm.png")
    
    # SHAP Dependence Plots for Top Predictors
    key_dep_feats = ['slope_deg', 'elevation_m', 'Rainfall_z', 'NDVI_z']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()
    for idx, feat_name in enumerate(key_dep_feats):
        ax = axes[idx]
        if feat_name in transformed_feature_names:
            feat_idx = transformed_feature_names.index(feat_name)
            feat_vals = X_test_trans[:, feat_idx]
            feat_shaps = shap_vals_class1[:, feat_idx]
            scatter = ax.scatter(feat_vals, feat_shaps, c=X_test_sub['slope_deg'], cmap='coolwarm', alpha=0.7, s=25)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.set_title(f"SHAP Dependence: {feat_name}", fontsize=11, fontweight='bold')
            ax.set_xlabel(f"{feat_name} (Standardized)", fontsize=9.5)
            ax.set_ylabel("SHAP Value (Log-Odds Contribution)", fontsize=9.5)
            ax.grid(True, linestyle=':', alpha=0.4)
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label("Slope (°)", fontsize=8.5)
    plt.suptitle("SHAP Partial Dependence Plots of Top Landslide Drivers (Himachal Pradesh)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig("figures/shap_dependence_plots.png", dpi=300)
    plt.close(fig)
    print("[+] Saved SHAP dependence plots to figures/shap_dependence_plots.png")
    
    # 5. Feature-Group Ablation Experiment for Himachal Pradesh
    print("\n[*] Performing Feature-Group Ablation Experiment...")
    from train import build_preprocessor
    
    feature_groups = {
        'Topographic / Morphometric': ['slope_deg', 'elevation_m', 'aspect_deg', 'Aspect_cos', 'Aspect_sin'],
        'Hydrologic Factors': ['Dist_River_z', 'FlowAccumulation_z'],
        'Climate / Rainfall': ['Rainfall_z'],
        'Ecological & Land Cover': ['NDVI_z', 'lulc_name']
    }
    
    baseline_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    ablation_records = [{
        'Ablation_Setting': 'Full Baseline (All Conditioning Factors)',
        'Features_Removed': 'None (Baseline)',
        'Num_Features_Used': len(feature_cols),
        'Test_ROC_AUC': baseline_auc,
        'AUC_Drop': 0.0
    }]
    
    for grp_name, remove_cols in feature_groups.items():
        retained_features = [c for c in feature_cols if c not in remove_cols]
        ret_num = [c for c in meta['numeric_features'] if c in retained_features]
        ret_cat = [c for c in meta['categorical_features'] if c in retained_features]
        
        ablated_prep = build_preprocessor(ret_cat, ret_num)
        ablated_rf = RandomForestClassifier(
            n_estimators=300, max_depth=16, min_samples_leaf=4, min_samples_split=10,
            max_features='sqrt', class_weight='balanced_subsample', bootstrap=True, random_state=42, n_jobs=-1
        )
        ablated_pipe = Pipeline(steps=[
            ('prep', ablated_prep),
            ('clf', ablated_rf)
        ])
        ablated_pipe.fit(X_train[retained_features], y_train)
        ablated_probs = ablated_pipe.predict_proba(X_test[retained_features])[:, 1]
        ablated_auc = roc_auc_score(y_test, ablated_probs)
        auc_drop = baseline_auc - ablated_auc
        
        ablation_records.append({
            'Ablation_Setting': f"Without {grp_name}",
            'Features_Removed': ", ".join(remove_cols),
            'Num_Features_Used': len(retained_features),
            'Test_ROC_AUC': ablated_auc,
            'AUC_Drop': auc_drop
        })
        print(f"    Ablated {grp_name}: Test ROC-AUC = {ablated_auc:.4f} (Drop = {auc_drop:+.4f})")
        
    ablation_df = pd.DataFrame(ablation_records)
    ablation_df.to_csv("tables/feature_group_ablation.csv", index=False)
    print("[+] Saved feature-group ablation to tables/feature_group_ablation.csv")
    print(ablation_df[['Ablation_Setting', 'Num_Features_Used', 'Test_ROC_AUC', 'AUC_Drop']].to_string(index=False))
    
    # Plot Feature-Group Ablation Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    plot_abl = ablation_df.iloc[1:].sort_values(by="AUC_Drop", ascending=True)
    palette_abl = ['#d7191c' if d > 0.03 else '#fdae61' if d > 0.01 else '#2b83ba' for d in plot_abl['AUC_Drop']]
    bars = ax.barh(plot_abl['Ablation_Setting'], plot_abl['AUC_Drop'], color=palette_abl, edgecolor='black', alpha=0.85)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title("Performance Degradation Upon Environmental Feature-Group Removal\n(Himachal Pradesh Western Himalayas)",
                 fontsize=11.5, fontweight='bold', pad=12)
    ax.set_xlabel("ROC-AUC Score Decrease (Baseline AUC - Ablated AUC)", fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f'-{w:.4f}', xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig("figures/feature_group_ablation.png", dpi=300)
    plt.close(fig)
    print("[+] Saved ablation chart to figures/feature_group_ablation.png")
    print("[+] Explainability module completed successfully!")

if __name__ == "__main__":
    run_explainability()
