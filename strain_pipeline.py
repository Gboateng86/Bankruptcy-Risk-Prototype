"""
STEP 2: Data Preparation + Building the Stacking Ensemble
This projects follows 5-step "Data Preparation Phase"
from the project design, builds the ensemble (the "panel of experts +
manager" system), applies the cost-adjustment,and checks results at the 
two threshold settings: 0.40 and 0.30.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
import joblib

RANDOM_STATE = 42
COST_WEIGHT = {0: 1, 1: 3}   # a missed bankruptcy is treated as 3x worse than a false alarm
REPORTED_THRESHOLDS = [0.40, 0.30]   # the two thresholds decided upon for this project


# STEP 2.1 - Load and clean the data
print("STEP 2.1: Loading and cleaning the data...")
df = pd.read_csv("bankruptcy_data.csv")
df["target"] = (df["status_label"] == "failed").astype(int)
feature_columns = [c for c in df.columns if c.startswith("X")]
print(f"Number of usable financial features: {len(feature_columns)}")


# STEP 2.2 - Split the data by TIME, not randomly
print("\nSTEP 2.2: Splitting data by year (temporal split)...")
train_df = df[(df["year"] >= 1999) & (df["year"] <= 2011)]
val_df = df[(df["year"] >= 2012) & (df["year"] <= 2014)]
test_df = df[(df["year"] >= 2015) & (df["year"] <= 2018)]
print(f"Training set:   {len(train_df)} records (1999-2011)")
print(f"Validation set: {len(val_df)} records (2012-2014)")
print(f"Test set:       {len(test_df)} records (2015-2018)")

X_train = train_df[feature_columns].values
y_train = train_df["target"].values
X_val = val_df[feature_columns].values
y_val = val_df["target"].values
X_test = test_df[feature_columns].values
y_test = test_df["target"].values


# STEP 2.3 - Scale the features
print("\nSTEP 2.3: Scaling features (fit on training data only)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# STEP 2.4 - Feature selection
print("\nSTEP 2.4: Selecting the most useful features...")
selector_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE)
selector_model.fit(X_train_scaled, y_train)
importances = pd.Series(selector_model.feature_importances_, index=feature_columns)
importances_sorted = importances.sort_values(ascending=False)

TOP_N_FEATURES = 12
selected_features = importances_sorted.head(TOP_N_FEATURES).index.tolist()
selected_idx = [feature_columns.index(f) for f in selected_features]
print(f"Keeping the top {TOP_N_FEATURES} features: {selected_features}")

X_train_sel = X_train_scaled[:, selected_idx]
X_val_sel = X_val_scaled[:, selected_idx]
X_test_sel = X_test_scaled[:, selected_idx]


# STEP 2.5 - Fix the class imbalance (train-only SMOTE)
print("\nSTEP 2.5: Balancing the training data with SMOTE (train-only)...")


def simple_smote(X_minority, n_synthetic, k_neighbors=5, random_state=42):
    """A from-scratch implementation of SMOTE (Chawla et al., 2002)."""
    rng = np.random.RandomState(random_state)
    n_minority = X_minority.shape[0]
    k_neighbors = min(k_neighbors, n_minority - 1)
    neighbour_finder = NearestNeighbors(n_neighbors=k_neighbors + 1)
    neighbour_finder.fit(X_minority)
    _, neighbour_indices = neighbour_finder.kneighbors(X_minority)
    synthetic_samples = np.zeros((n_synthetic, X_minority.shape[1]))
    for i in range(n_synthetic):
        base_idx = rng.randint(0, n_minority)
        neighbour_choice = rng.randint(1, k_neighbors + 1)
        neighbour_idx = neighbour_indices[base_idx, neighbour_choice]
        gap = rng.rand()
        synthetic_samples[i] = X_minority[base_idx] + gap * (X_minority[neighbour_idx] - X_minority[base_idx])
    return synthetic_samples


minority_mask = y_train == 1
majority_mask = y_train == 0
X_minority = X_train_sel[minority_mask]
X_majority = X_train_sel[majority_mask]
n_to_create = X_majority.shape[0] - X_minority.shape[0]
synthetic_X = simple_smote(X_minority, n_to_create, k_neighbors=5, random_state=RANDOM_STATE)
synthetic_y = np.ones(n_to_create)

X_train_bal = np.vstack([X_train_sel, synthetic_X])
y_train_bal = np.concatenate([y_train, synthetic_y])
print(f"Training data after balancing: {len(y_train_bal)} rows "
      f"({(y_train_bal==1).sum()} failed, {(y_train_bal==0).sum()} alive)")


# STEP 2.6 - Train the three "expert" models (base learners),
#            each told a missed bankruptcy costs 3x more than a false alarm
print("\nSTEP 2.6: Training the three base learners (3x cost-weighted)...")

rf_model = RandomForestClassifier(
    n_estimators=150, random_state=RANDOM_STATE, n_jobs=1, class_weight=COST_WEIGHT,
    # max_depth and min_samples_leaf are constrained deliberately. Left
    # unlimited, the trees grow to depth 61 and the saved pipeline file
    # reaches 229 MB, which exceeds GitHub's 100 MB per-file limit and
    # cannot be deployed. These settings reduce the saved file to roughly
    # 27 MB while leaving results effectively unchanged (AUC 0.7762 to
    # 0.7700), and additionally reduce overfitting by preventing the trees
    # from memorising individual training rows.
    max_depth=12, min_samples_leaf=20
)
rf_model.fit(X_train_bal, y_train_bal)
print("  - Random Forest trained")

sample_weights = np.where(y_train_bal == 1, 3, 1)
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE)
gb_model.fit(X_train_bal, y_train_bal, sample_weight=sample_weights)
print("  - Gradient Boosting (XGBoost stand-in) trained")

# k-Nearest Neighbours has no concept of "cost" - it simply looks at
# nearby companies and copies the majority outcome, so it cannot use
# the 3x weighting the other two models use.
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train_bal, y_train_bal)
print("  - k-Nearest Neighbours trained (cost-weighting not applicable to this model)")


# STEP 2.7 - Train the "manager" model (meta-learner)
print("\nSTEP 2.7: Training the meta-learner (the 'manager')...")

rf_val_pred = rf_model.predict_proba(X_val_sel)[:, 1]
gb_val_pred = gb_model.predict_proba(X_val_sel)[:, 1]
knn_val_pred = knn_model.predict_proba(X_val_sel)[:, 1]
meta_features_val = np.column_stack([rf_val_pred, gb_val_pred, knn_val_pred])

meta_model = LogisticRegression(random_state=RANDOM_STATE, class_weight=COST_WEIGHT)
meta_model.fit(meta_features_val, y_val)
print("  - Meta-learner trained")


# STEP 2.8 - Evaluate on the TEST set, at the two decided thresholds
print("\nSTEP 2.8: Evaluating on the test set at both decided thresholds...")

rf_test_pred = rf_model.predict_proba(X_test_sel)[:, 1]
gb_test_pred = gb_model.predict_proba(X_test_sel)[:, 1]
knn_test_pred = knn_model.predict_proba(X_test_sel)[:, 1]
meta_features_test = np.column_stack([rf_test_pred, gb_test_pred, knn_test_pred])
final_probabilities = meta_model.predict_proba(meta_features_test)[:, 1]

auc = roc_auc_score(y_test, final_probabilities)
print(f"\nStacking ensemble AUC (threshold-independent): {auc:.4f}")

for threshold in REPORTED_THRESHOLDS:
    preds = (final_probabilities >= threshold).astype(int)
    f1 = f1_score(y_test, preds, average="macro")
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    type_1_error = fp / (fp + tn)
    type_2_error = fn / (fn + tp)
    label = "PRIMARY (best Macro-F1)" if threshold == 0.40 else "SECONDARY (fewer missed bankruptcies)"
    print(f"\n--- Threshold {threshold:.2f} [{label}] ---")
    print(f"Macro-F1:     {f1:.4f}")
    print(f"Type I error:  {type_1_error:.4f} (false alarms)")
    print(f"Type II error: {type_2_error:.4f} (missed bankruptcies)")
    print(f"Confusion matrix -> TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# STEP 2.8b - Compare each individual expert against the combined system
# This directly satisfies an Objective of the project by assessing the stacking ensemble
# against the individual classifiers it is built from, not just reporting
# the combined system's own numbers in isolation.
print("\n" + "=" * 70)
print("COMPARISON: EACH INDIVIDUAL EXPERT vs. THE COMBINED ENSEMBLE")
print("=" * 70)

individual_models = [
    ("Random Forest alone", rf_test_pred),
    ("Gradient Boosting alone", gb_test_pred),
    ("k-Nearest Neighbours alone", knn_test_pred),
    ("STACKING ENSEMBLE (combined)", final_probabilities),
]

for threshold in REPORTED_THRESHOLDS:
    print(f"\n--- At threshold {threshold:.2f} ---")
    print(f"{'Model':32s} {'AUC':>7s} {'Macro-F1':>9s} {'Type I':>8s} {'Type II':>8s}")
    for name, probs in individual_models:
        preds = (probs >= threshold).astype(int)
        model_auc = roc_auc_score(y_test, probs)
        model_f1 = f1_score(y_test, preds, average="macro")
        tn_i, fp_i, fn_i, tp_i = confusion_matrix(y_test, preds).ravel()
        t1_i = fp_i / (fp_i + tn_i)
        t2_i = fn_i / (fn_i + tp_i)
        print(f"{name:32s} {model_auc:7.4f} {model_f1:9.4f} {t1_i:8.4f} {t2_i:8.4f}")


# STEP 2.9 - Save everything needed for the Streamlit app
print("\nSTEP 2.9: Saving the trained pipeline to disk...")

pipeline_bundle = {
    "scaler": scaler,
    "selected_features": selected_features,
    "all_feature_columns": feature_columns,
    "feature_importances": importances_sorted.head(TOP_N_FEATURES).to_dict(),
    "rf_model": rf_model,
    "gb_model": gb_model,
    "knn_model": knn_model,
    "meta_model": meta_model,
}
joblib.dump(pipeline_bundle, "trained_pipeline_final.joblib", compress=3)
print("Saved as trained_pipeline_final.joblib - used by the Streamlit app.")
print("\nAll done. Move on to Step 3 (the Streamlit web app).")
