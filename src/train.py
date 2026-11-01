"""
Model Training and Hyperparameter Tuning Module
Project: AI-Powered Prediction of Landslide Susceptibility Using Machine Learning
Dataset: Himachal Pradesh Geo-Environmental Landslide Dataset (Mendeley Data DOI: 10.17632/mxpg845z62.2)
Compares: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, MLP, Voting Ensemble
Evaluates Spatial GroupKFold vs Conventional Random StratifiedKFold
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def build_preprocessor(categorical_features, numeric_features):
    """
    Constructs a ColumnTransformer that imputes and scales numeric features,
    and imputes and one-hot encodes categorical features.
    Fitted strictly on training data only.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def evaluate_cv_folds(models_dict, X, y, cv_splitter, groups=None, use_smote=False, numeric_features=None, categorical_features=None):
    """
    Evaluates models across cross-validation folds.
    Returns summary dataframe of mean +/- std metrics.
    """
    results = {name: {'roc_auc': [], 'pr_auc': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'fit_time': []} for name in models_dict}
    
    folds = list(cv_splitter.split(X, y, groups=groups))
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        for name, clf in models_dict.items():
            preprocessor = build_preprocessor(categorical_features, numeric_features)
            
            if use_smote:
                pipe = ImbPipeline(steps=[
                    ('prep', preprocessor),
                    ('smote', SMOTE(random_state=42)),
                    ('clf', clf)
                ])
            else:
                pipe = Pipeline(steps=[
                    ('prep', preprocessor),
                    ('clf', clf)
                ])
                
            t0 = time.time()
            pipe.fit(X_train, y_train)
            t_fit = time.time() - t0
            
            if hasattr(pipe, "predict_proba"):
                y_prob = pipe.predict_proba(X_val)[:, 1]
            elif hasattr(pipe, "decision_function"):
                df_val = pipe.decision_function(X_val)
                y_prob = 1 / (1 + np.exp(-df_val))
            else:
                y_prob = pipe.predict(X_val)
                
            y_pred = (y_prob >= 0.5).astype(int)
            
            # Check for folds with single class in validation (if any)
            if len(np.unique(y_val)) > 1:
                results[name]['roc_auc'].append(roc_auc_score(y_val, y_prob))
                results[name]['pr_auc'].append(average_precision_score(y_val, y_prob))
            else:
                results[name]['roc_auc'].append(np.nan)
                results[name]['pr_auc'].append(np.nan)
                
            results[name]['accuracy'].append(accuracy_score(y_val, y_pred))
            results[name]['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            results[name]['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            results[name]['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            results[name]['fit_time'].append(t_fit)
            
    summary_rows = []
    for name, m_dict in results.items():
        summary_rows.append({
            'Model': name,
            'ROC_AUC_Mean': float(np.nanmean(m_dict['roc_auc'])),
            'ROC_AUC_Std': float(np.nanstd(m_dict['roc_auc'])),
            'PR_AUC_Mean': float(np.nanmean(m_dict['pr_auc'])),
            'PR_AUC_Std': float(np.nanstd(m_dict['pr_auc'])),
            'Accuracy_Mean': float(np.mean(m_dict['accuracy'])),
            'Accuracy_Std': float(np.std(m_dict['accuracy'])),
            'F1_Mean': float(np.mean(m_dict['f1'])),
            'F1_Std': float(np.std(m_dict['f1'])),
            'Precision_Mean': float(np.mean(m_dict['precision'])),
            'Recall_Mean': float(np.mean(m_dict['recall'])),
            'Fit_Time_Sec': float(np.mean(m_dict['fit_time']))
        })
    return pd.DataFrame(summary_rows)

def train_and_tune_all():
    print("="*60)
    print("  HIMACHAL PRADESH MODEL TRAINING & SPATIAL VALIDATION")
    print("="*60)
    
    # 1. Load Cleaned Dataset
    data_path = "data/cleaned_landslide_data.csv"
    if not os.path.exists(data_path):
        from data_prep import run_data_prep_and_eda
        df = run_data_prep_and_eda()
    else:
        df = pd.read_csv(data_path)
        
    print(f"Loaded {len(df):,} samples from {data_path}")
    
    # Define features for Himachal Pradesh dataset
    numeric_features = [
        'slope_deg', 'elevation_m', 'aspect_deg', 'Aspect_cos', 'Aspect_sin',
        'Dist_River_z', 'FlowAccumulation_z', 'NDVI_z', 'Rainfall_z'
    ]
    categorical_features = ['lulc_name']
    all_feature_cols = numeric_features + categorical_features
    
    X = df[all_feature_cols]
    y = df['class']
    spatial_groups = df['spatial_cluster']
    
    # 2. Holdout Test Split (Untouched Final Evaluation Set: 20%)
    # Use stratified split with fixed seed=42
    X_train_full, X_test, y_train_full, y_test, groups_train, groups_test = train_test_split(
        X, y, spatial_groups, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train/Val pool: {len(X_train_full):,} samples | Untouched Test set: {len(X_test):,} samples")
    
    # Save partitions for independent evaluation
    test_df = df.loc[X_test.index].copy()
    test_df.to_csv("data/test_dataset_untouched.csv", index=False)
    
    train_df = df.loc[X_train_full.index].copy()
    train_df.to_csv("data/train_dataset_pool.csv", index=False)
    
    # 3. Model Definitions with Tuned Hyperparameters for Himalayan Terrain
    candidate_models = {
        'Logistic Regression': LogisticRegression(
            C=0.5, solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, min_samples_split=40, criterion='gini', class_weight='balanced', random_state=42
        ),
        'Random Forest (Primary)': RandomForestClassifier(
            n_estimators=300, max_depth=16, min_samples_leaf=4, min_samples_split=10,
            max_features='sqrt', class_weight='balanced_subsample', bootstrap=True, random_state=42, n_jobs=-1
        ),
        'Support Vector Machine (SVM)': SVC(
            C=1.5, gamma='scale', kernel='rbf', probability=True, class_weight='balanced', random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, eval_metric='logloss', random_state=42, n_jobs=-1
        ),
        'Multi-Layer Perceptron (MLP)': MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', alpha=0.01,
            learning_rate_init=0.003, max_iter=500, early_stopping=True, n_iter_no_change=15, random_state=42
        )
    }
    
    # 4. Compare Cross-Validation Protocols:
    # A) Spatial GroupKFold CV (5 geographic blocks)
    print("\n[*] Running 5-Fold Spatial GroupKFold Cross-Validation across Himalayan Sectors...")
    gkf = GroupKFold(n_splits=5)
    spatial_cv_results = evaluate_cv_folds(
        candidate_models, X_train_full, y_train_full, cv_splitter=gkf,
        groups=groups_train, use_smote=False,
        numeric_features=numeric_features, categorical_features=categorical_features
    )
    spatial_cv_results['Validation_Protocol'] = 'Spatial GroupKFold (Leakage-Free)'
    
    # B) Conventional Random StratifiedKFold CV (5 folds)
    print("[*] Running 5-Fold Conventional StratifiedKFold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_cv_results = evaluate_cv_folds(
        candidate_models, X_train_full, y_train_full, cv_splitter=skf,
        groups=None, use_smote=False,
        numeric_features=numeric_features, categorical_features=categorical_features
    )
    random_cv_results['Validation_Protocol'] = 'Random StratifiedKFold'
    
    # C) Spatial GroupKFold with SMOTE (inside training folds)
    print("[*] Running 5-Fold Spatial GroupKFold with SMOTE Oversampling...")
    smote_spatial_results = evaluate_cv_folds(
        candidate_models, X_train_full, y_train_full, cv_splitter=gkf,
        groups=groups_train, use_smote=True,
        numeric_features=numeric_features, categorical_features=categorical_features
    )
    smote_spatial_results['Validation_Protocol'] = 'Spatial GroupKFold + SMOTE'
    
    # Save combined CV tuning results
    all_cv_table = pd.concat([spatial_cv_results, random_cv_results, smote_spatial_results], ignore_index=True)
    all_cv_table.to_csv("tables/cv_tuning_results.csv", index=False)
    print("[+] Saved cross-validation tuning results to tables/cv_tuning_results.csv")
    
    # Create Side-by-Side Spatial vs Random Leakage Gap Table
    gap_rows = []
    for model_name in candidate_models.keys():
        sp_row = spatial_cv_results[spatial_cv_results['Model'] == model_name].iloc[0]
        rd_row = random_cv_results[random_cv_results['Model'] == model_name].iloc[0]
        gap_rows.append({
            'Model': model_name,
            'Spatial_ROC_AUC': f"{sp_row['ROC_AUC_Mean']:.4f} ± {sp_row['ROC_AUC_Std']:.4f}",
            'Random_ROC_AUC': f"{rd_row['ROC_AUC_Mean']:.4f} ± {rd_row['ROC_AUC_Std']:.4f}",
            'Optimism_Gap_AUC': rd_row['ROC_AUC_Mean'] - sp_row['ROC_AUC_Mean'],
            'Spatial_Accuracy': f"{sp_row['Accuracy_Mean']:.4f} ± {sp_row['Accuracy_Std']:.4f}",
            'Random_Accuracy': f"{rd_row['Accuracy_Mean']:.4f} ± {rd_row['Accuracy_Std']:.4f}",
            'Optimism_Gap_Acc': rd_row['Accuracy_Mean'] - sp_row['Accuracy_Mean']
        })
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv("tables/spatial_vs_random_splits.csv", index=False)
    print("[+] Saved spatial vs random optimism gap to tables/spatial_vs_random_splits.csv")
    
    # 5. Fit Final Selected Pipelines on Full Training Pool (N=9,389)
    fitted_models = {}
    print(f"\n[*] Fitting all candidate models on Full Training Pool (N={len(X_train_full):,})...")
    for name, clf in candidate_models.items():
        preprocessor = build_preprocessor(categorical_features, numeric_features)
        pipe = Pipeline(steps=[
            ('prep', preprocessor),
            ('clf', clf)
        ])
        pipe.fit(X_train_full, y_train_full)
        fitted_models[name] = pipe
        print(f"    Fitted {name}")
        
    # Soft-Voting Ensemble (Random Forest + XGBoost + Logistic Regression)
    print("[*] Creating Soft-Voting Ensemble (Random Forest + XGBoost + Logistic Regression)...")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', candidate_models['Random Forest (Primary)']),
            ('xgb', candidate_models['XGBoost']),
            ('lr', candidate_models['Logistic Regression'])
        ],
        voting='soft'
    )
    voting_pipe = Pipeline(steps=[
        ('prep', build_preprocessor(categorical_features, numeric_features)),
        ('clf', voting_clf)
    ])
    voting_pipe.fit(X_train_full, y_train_full)
    fitted_models['Voting Ensemble (RF+XGB+LR)'] = voting_pipe
    print("    Fitted Voting Ensemble")
    
    # Save the primary Random Forest pipeline and all models
    os.makedirs("models", exist_ok=True)
    joblib.dump(fitted_models['Random Forest (Primary)'], "models/best_landslide_model.joblib")
    joblib.dump(fitted_models['Random Forest (Primary)'].named_steps['prep'], "models/feature_pipeline.joblib")
    joblib.dump(fitted_models, "models/all_fitted_models.joblib")
    
    # Save feature metadata for predict.py
    feature_meta = {
        'dataset_region': 'Himachal Pradesh, India',
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'all_features': all_feature_cols,
        'target_col': 'class',
        'target_classes': [0, 1],
        'class_names': ['Non-Landslide', 'Landslide']
    }
    with open("models/feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2)
        
    print("[+] Saved primary model to models/best_landslide_model.joblib")
    print("[+] Saved all fitted models to models/all_fitted_models.joblib")
    return fitted_models, X_train_full, y_train_full, X_test, y_test

if __name__ == "__main__":
    train_and_tune_all()
