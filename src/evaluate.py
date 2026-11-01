"""
Model Evaluation and Diagnostic Module
Project: AI-Powered Prediction of Landslide Susceptibility Using Machine Learning
Dataset: Himachal Pradesh Geo-Environmental Landslide Dataset (Mendeley Data DOI: 10.17632/mxpg845z62.2)
Evaluates on untouched test set (N=2,348), performs threshold tuning on validation,
computes 95% bootstrap CI, and generates 300 DPI diagnostic figures.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve

# Styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def compute_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def evaluate_models_on_test():
    print("="*60)
    print("  FINAL INDEPENDENT EVALUATION ON UNTOUCHED HIMALAYAN TEST SET")
    print("="*60)
    
    # 1. Load test set and models
    test_df = pd.read_csv("data/test_dataset_untouched.csv")
    train_df = pd.read_csv("data/train_dataset_pool.csv")
    fitted_models = joblib.load("models/all_fitted_models.joblib")
    
    with open("models/feature_metadata.json", "r") as f:
        meta = json.load(f)
    feature_cols = meta['all_features']
    target_col = meta.get('target_col', 'class')
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values
    
    print(f"Loaded test set with {len(test_df):,} samples (Landslides: {(y_test == 1).sum():,}, Non-Landslides: {(y_test == 0).sum():,})")
    
    # 2. Threshold Optimization on Validation Data (Train Pool)
    print("\n[*] Performing Operating Threshold Analysis on Validation Data...")
    rf_pipe = fitted_models['Random Forest (Primary)']
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_probs = np.zeros(len(y_train))
    
    for tr_idx, v_idx in skf.split(X_train, y_train):
        clone_pipe = joblib.load("models/best_landslide_model.joblib")
        clone_pipe.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        val_probs[v_idx] = clone_pipe.predict_proba(X_train.iloc[v_idx])[:, 1]
        
    thresholds_sweep = np.linspace(0.10, 0.90, 81)
    thresh_records = []
    for th in thresholds_sweep:
        y_val_pred_th = (val_probs >= th).astype(int)
        rec = recall_score(y_train, y_val_pred_th, zero_division=0)
        prec = precision_score(y_train, y_val_pred_th, zero_division=0)
        f1 = f1_score(y_train, y_val_pred_th, zero_division=0)
        f2 = fbeta_score(y_train, y_val_pred_th, beta=2, zero_division=0)
        spec = compute_specificity(y_train, y_val_pred_th)
        fpr = 1.0 - spec
        thresh_records.append({
            'Threshold': th, 'Precision': prec, 'Recall': rec,
            'F1_Score': f1, 'F2_Score': f2, 'Specificity': spec, 'False_Positive_Rate': fpr
        })
    thresh_df = pd.DataFrame(thresh_records)
    thresh_df.to_csv("tables/threshold_analysis.csv", index=False)
    
    # Choose optimal threshold prioritizing F2 (favoring high recall >= 90% to prevent fatal hazard misses)
    high_recall_candidates = thresh_df[thresh_df['Recall'] >= 0.90]
    if len(high_recall_candidates) > 0:
        best_th_row = high_recall_candidates.sort_values(by="F2_Score", ascending=False).iloc[0]
    else:
        best_th_row = thresh_df.sort_values(by="F2_Score", ascending=False).iloc[0]
        
    selected_threshold = float(best_th_row['Threshold'])
    print(f"[+] Chosen Operational Hazard Threshold: {selected_threshold:.3f}")
    print(f"    Validation Recall: {best_th_row['Recall']:.4f} | Precision: {best_th_row['Precision']:.4f} | FPR: {best_th_row['False_Positive_Rate']:.4f}")
    
    with open("models/threshold_config.json", "w") as f:
        json.dump({
            'selected_threshold': selected_threshold,
            'default_threshold': 0.50,
            'optimization_objective': 'Maximize F2-score with Recall >= 0.90 on validation folds',
            'validation_recall': float(best_th_row['Recall']),
            'validation_precision': float(best_th_row['Precision']),
            'validation_fpr': float(best_th_row['False_Positive_Rate'])
        }, f, indent=2)
        
    # Plot Threshold Trade-off Curve
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    ax.plot(thresh_df['Threshold'], thresh_df['Recall'], label='Recall (Sensitivity)', color='#d7191c', linewidth=2)
    ax.plot(thresh_df['Threshold'], thresh_df['Precision'], label='Precision', color='#2b83ba', linewidth=2)
    ax.plot(thresh_df['Threshold'], thresh_df['F1_Score'], label='F1-Score', color='#4daf4a', linewidth=1.8, linestyle='--')
    ax.plot(thresh_df['Threshold'], thresh_df['F2_Score'], label='F2-Score (Hazard Weight)', color='#984ea3', linewidth=2.2)
    ax.plot(thresh_df['Threshold'], thresh_df['False_Positive_Rate'], label='False Positive Rate (FPR)', color='#fdae61', linewidth=1.5, linestyle=':')
    ax.axvline(x=selected_threshold, color='black', linestyle='--', linewidth=1.5, label=f'Operational Threshold ({selected_threshold:.2f})')
    ax.axvline(x=0.50, color='gray', linestyle=':', linewidth=1.2, label='Standard Threshold (0.50)')
    ax.set_title("Operational Classification Threshold vs Evaluation Metrics (Himachal Pradesh)", fontsize=11.5, fontweight='bold', pad=12)
    ax.set_xlabel("Classification Decision Threshold", fontsize=10.5)
    ax.set_ylabel("Metric Value", fontsize=10.5)
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc='lower left', frameon=True, fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig("figures/threshold_tradeoff_curve.png", dpi=300)
    plt.close(fig)
    print("[+] Saved threshold curve to figures/threshold_tradeoff_curve.png")
    
    # 3. Test Set Evaluation Across All Models
    print("\n[*] Evaluating All Models on Untouched Test Set...")
    eval_records = []
    test_probs_dict = {}
    test_preds_default_dict = {}
    test_preds_opt_dict = {}
    
    for name, pipe in fitted_models.items():
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe, "decision_function"):
            probs = 1 / (1 + np.exp(-pipe.decision_function(X_test)))
        else:
            probs = pipe.predict(X_test)
            
        test_probs_dict[name] = probs
        
        preds_def = (probs >= 0.50).astype(int)
        test_preds_default_dict[name] = preds_def
        
        th_used = selected_threshold if 'Random Forest' in name else 0.50
        preds_opt = (probs >= th_used).astype(int)
        test_preds_opt_dict[name] = preds_opt
        
        acc = accuracy_score(y_test, preds_def)
        prec = precision_score(y_test, preds_def, zero_division=0)
        rec = recall_score(y_test, preds_def, zero_division=0)
        f1 = f1_score(y_test, preds_def, zero_division=0)
        spec = compute_specificity(y_test, preds_def)
        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        mcc = matthews_corrcoef(y_test, preds_def)
        brier = brier_score_loss(y_test, probs)
        
        eval_records.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'Specificity': spec,
            'F1_Score': f1,
            'ROC_AUC': roc_auc,
            'PR_AUC': pr_auc,
            'MCC': mcc,
            'Brier_Score': brier
        })
        
    eval_df = pd.DataFrame(eval_records).sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)
    eval_df.to_csv("tables/model_comparison.csv", index=False)
    print("[+] Saved model comparison table to tables/model_comparison.csv")
    print(eval_df.to_string(index=False))
    
    # 4. Bootstrap 95% Confidence Interval for Best Model's Test ROC-AUC
    best_model_name = 'Random Forest (Primary)'
    best_probs = test_probs_dict[best_model_name]
    print(f"\n[*] Calculating 95% Bootstrap CI for {best_model_name} Test ROC-AUC (1,000 iterations)...")
    np.random.seed(42)
    n_bootstraps = 1000
    bootstrapped_scores = []
    n_samples = len(y_test)
    
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n_samples, n_samples)
        if len(np.unique(y_test[indices])) < 2:
            continue
        score = roc_auc_score(y_test[indices], best_probs[indices])
        bootstrapped_scores.append(score)
        
    ci_lower = np.percentile(bootstrapped_scores, 2.5)
    ci_upper = np.percentile(bootstrapped_scores, 97.5)
    print(f"[+] {best_model_name} Test ROC-AUC: {roc_auc_score(y_test, best_probs):.4f} [95% CI: {ci_lower:.4f} - {ci_upper:.4f}]")
    
    with open("tables/bootstrap_ci_best_model.json", "w") as f:
        json.dump({
            'model': best_model_name,
            'region': 'Himachal Pradesh, India',
            'test_roc_auc': float(roc_auc_score(y_test, best_probs)),
            'ci_lower_95': float(ci_lower),
            'ci_upper_95': float(ci_upper),
            'n_bootstraps': n_bootstraps
        }, f, indent=2)
        
    # 5. Diagnostic Figures (300 DPI)
    # --- Plot A: ROC Curves ---
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    palette = {
        'Random Forest (Primary)': '#d7191c',
        'Voting Ensemble (RF+XGB+LR)': '#e41a1c',
        'XGBoost': '#2b83ba',
        'Support Vector Machine (SVM)': '#4daf4a',
        'Logistic Regression': '#fdae61',
        'Multi-Layer Perceptron (MLP)': '#984ea3',
        'Decision Tree': '#7570b3'
    }
    for name, probs in test_probs_dict.items():
        fpr_vals, tpr_vals, _ = roc_curve(y_test, probs)
        score = roc_auc_score(y_test, probs)
        color = palette.get(name, '#333333')
        lw = 2.5 if 'Random Forest' in name or 'Ensemble' in name else 1.6
        ax.plot(fpr_vals, tpr_vals, label=f"{name} (AUC = {score:.4f})", color=color, linewidth=lw)
        
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Random Chance (AUC = 0.5000)')
    ax.set_title("Receiver Operating Characteristic (ROC) — Himachal Pradesh Test Set", fontsize=11.5, fontweight='bold', pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10.5)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10.5)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.legend(loc='lower right', frameon=True, fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig("figures/model_roc_curves.png", dpi=300)
    plt.close(fig)
    print("[+] Saved ROC curves to figures/model_roc_curves.png")
    
    # --- Plot B: Precision-Recall Curves ---
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    for name, probs in test_probs_dict.items():
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, probs)
        score = average_precision_score(y_test, probs)
        color = palette.get(name, '#333333')
        lw = 2.5 if 'Random Forest' in name or 'Ensemble' in name else 1.6
        ax.plot(rec_vals, prec_vals, label=f"{name} (PR-AUC = {score:.4f})", color=color, linewidth=lw)
        
    baseline_pr = (y_test == 1).sum() / len(y_test)
    ax.axhline(y=baseline_pr, color='gray', linestyle='--', linewidth=1.2, label=f'Baseline ({baseline_pr:.4f})')
    ax.set_title("Precision-Recall (PR) Curves — Himachal Pradesh Test Set", fontsize=11.5, fontweight='bold', pad=12)
    ax.set_xlabel("Recall (True Positive Rate)", fontsize=10.5)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=10.5)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(baseline_pr - 0.05, 1.02)
    ax.legend(loc='lower left', frameon=True, fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig("figures/model_pr_curves.png", dpi=300)
    plt.close(fig)
    print("[+] Saved PR curves to figures/model_pr_curves.png")
    
    # --- Plot C: Confusion Matrices Grid ---
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9), dpi=300)
    axes = axes.flatten()
    model_keys = ['Logistic Regression', 'Decision Tree', 'Random Forest (Primary)',
                  'Support Vector Machine (SVM)', 'XGBoost', 'Multi-Layer Perceptron (MLP)']
    
    for idx, name in enumerate(model_keys):
        ax = axes[idx]
        preds = test_preds_default_dict[name]
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    annot_kws={'size': 12, 'weight': 'bold'},
                    xticklabels=['Non-Landslide', 'Landslide'],
                    yticklabels=['Non-Landslide', 'Landslide'])
        ax.set_title(f"{name}\nAcc: {accuracy_score(y_test, preds):.3f} | F1: {f1_score(y_test, preds):.3f}",
                     fontsize=10.5, fontweight='bold')
        ax.set_xlabel("Predicted Label", fontsize=9.5)
        ax.set_ylabel("True Label", fontsize=9.5)
        
    plt.suptitle("Confusion Matrices on Untouched Test Set (Himachal Pradesh, N=2,348)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig("figures/model_confusion_matrices.png", dpi=300)
    plt.close(fig)
    print("[+] Saved confusion matrices to figures/model_confusion_matrices.png")
    
    # --- Plot D: Model Performance Comparison Bar Chart ---
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    bar_df = eval_df[eval_df['Model'].isin(model_keys)].copy()
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    melted = bar_df.melt(id_vars=['Model'], value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    sns.barplot(data=melted, x='Model', y='Score', hue='Metric', ax=ax, palette='Set2', edgecolor='black')
    ax.set_title("Test Set Metric Comparison Across Models (Himachal Pradesh)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=10.5)
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(range(len(model_keys)))
    ax.set_xticklabels(model_keys, rotation=20, ha='right', fontsize=9.5)
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig("figures/model_performance_comparison.png", dpi=300)
    plt.close(fig)
    print("[+] Saved comparison chart to figures/model_performance_comparison.png")
    
    # --- Plot E: Calibration Reliability Curves & Brier Score ---
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration", linewidth=1.5)
    for name in model_keys:
        probs = test_probs_dict[name]
        brier = brier_score_loss(y_test, probs)
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
        color = palette.get(name, '#333333')
        lw = 2.2 if 'Random Forest' in name else 1.5
        ax.plot(prob_pred, prob_true, marker='s', label=f"{name} (Brier = {brier:.4f})", color=color, linewidth=lw)
        
    ax.set_title("Calibration Reliability Diagrams and Brier Scores (Himachal Pradesh)", fontsize=11.5, fontweight='bold', pad=12)
    ax.set_xlabel("Mean Predicted Probability", fontsize=10.5)
    ax.set_ylabel("Fraction of Landslide Positives", fontsize=10.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower right', frameon=True, fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig("figures/model_calibration_curves.png", dpi=300)
    plt.close(fig)
    print("[+] Saved calibration curves to figures/model_calibration_curves.png")
    
    print("[+] All evaluation steps for Himachal Pradesh completed successfully!")
    return eval_df

if __name__ == "__main__":
    evaluate_models_on_test()
