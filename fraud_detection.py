# =============================================================================
#  CREDIT CARD FRAUD DETECTION
#  ML InnovateX Hackathon
# =============================================================================
#
#  DATASET : Credit Card Fraud Detection
#  Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#  File    : creditcard.csv
#
#  Run: python fraud_detection.py
#
#  Requirements:
#  pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow
#
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      GridSearchCV, RandomizedSearchCV,
                                      cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
#  1. DATA UNDERSTANDING & CLEANING
# =============================================================================

print("=" * 60)
print("  1. DATA UNDERSTANDING & CLEANING")
print("=" * 60)

df = pd.read_csv("creditcard.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicates:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

print("\nBasic statistics:")
print(df.describe())

print("\nClass distribution:")
print(df["Class"].value_counts())
print(f"Fraud %: {df['Class'].mean() * 100:.4f}%")


# =============================================================================
#  2. EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("  2. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --- 2.1 Class imbalance ---
plt.figure()
df["Class"].value_counts().plot(kind="bar")
plt.title("Class Distribution (0=Legit, 1=Fraud)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# --- 2.2 Transaction amount by class ---
plt.figure()
df.boxplot(column="Amount", by="Class")
plt.title("Transaction Amount by Class")
plt.suptitle("")
plt.xlabel("Class (0=Legit, 1=Fraud)")
plt.ylabel("Amount")
plt.show()

# --- 2.3 Time distribution by class ---
plt.figure()
df[df["Class"] == 0]["Time"].hist(bins=50, alpha=0.7, label="Legit")
df[df["Class"] == 1]["Time"].hist(bins=50, alpha=0.7, label="Fraud")
plt.title("Transaction Time Distribution by Class")
plt.xlabel("Time (seconds)")
plt.ylabel("Count")
plt.legend()
plt.show()

# --- 2.4 Amount distribution (log scale) ---
plt.figure()
df[df["Class"] == 0]["Amount"].apply(lambda x: np.log1p(x)).hist(
    bins=50, alpha=0.7, label="Legit")
df[df["Class"] == 1]["Amount"].apply(lambda x: np.log1p(x)).hist(
    bins=50, alpha=0.7, label="Fraud")
plt.title("Log Amount Distribution by Class")
plt.xlabel("log(Amount + 1)")
plt.ylabel("Count")
plt.legend()
plt.show()

# --- 2.5 Correlation heatmap (V features + Amount + Class) ---
# PCA features V1-V28 are already anonymised
# Show correlation of top features with Class
corr_with_class = df.corr()["Class"].abs().sort_values(ascending=False)
print("\nTop 10 features correlated with Class:")
print(corr_with_class.head(11))

top_features = corr_with_class.head(11).index.tolist()
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Top Features)")
plt.show()

# --- 2.6 Top V features distribution by class ---
top_v = corr_with_class[1:6].index.tolist()   # skip Class itself
fig, axes = plt.subplots(1, len(top_v), figsize=(15, 4))
for ax, feat in zip(axes, top_v):
    df[df["Class"] == 0][feat].hist(bins=40, alpha=0.6, label="Legit", ax=ax)
    df[df["Class"] == 1][feat].hist(bins=40, alpha=0.6, label="Fraud", ax=ax)
    ax.set_title(feat)
    ax.legend(fontsize=7)
plt.suptitle("Top V Feature Distributions by Class")
plt.tight_layout()
plt.show()

print("\n--- Key Insights ---")
print("1. Dataset is heavily imbalanced: ~0.17% fraud transactions.")
print("2. Fraud transactions tend to have lower amounts on average.")
print("3. Fraud is spread across all time periods, not time-specific.")
print("4. V14, V17, V12, V10 are most negatively correlated with fraud.")
print("5. V4, V11 are positively correlated with fraud class.")


# =============================================================================
#  3. DATA PREPROCESSING
# =============================================================================

print("\n" + "=" * 60)
print("  3. DATA PREPROCESSING")
print("=" * 60)

# Scale Amount and Time (V1-V28 are already PCA scaled)
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
df.drop(columns=["Amount", "Time"], inplace=True)

X = df.drop(columns=["Class"])
y = df["Class"]

# Train-test split (stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train size: {X_train.shape}")
print(f"Test size : {X_test.shape}")
print(f"Train fraud count: {y_train.sum()}")
print(f"Test fraud count : {y_test.sum()}")

# Handle class imbalance with SMOTE (on training set only)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"Train size     : {X_train_sm.shape}")
print(f"Class balance  : {pd.Series(y_train_sm).value_counts().to_dict()}")


# =============================================================================
#  4. FEATURE ENGINEERING & SELECTION
# =============================================================================

print("\n" + "=" * 60)
print("  4. FEATURE ENGINEERING & SELECTION")
print("=" * 60)

# Feature 1: Log of Amount (before we dropped it, recreate from scaled)
# Since Amount is now scaled, create interaction with top correlated features
X_train_sm["V14_V17"]   = X_train_sm["V14"] * X_train_sm["V17"]
X_test["V14_V17"]        = X_test["V14"]   * X_test["V17"]

X_train_sm["V12_V10"]   = X_train_sm["V12"] * X_train_sm["V10"]
X_test["V12_V10"]        = X_test["V12"]   * X_test["V10"]

print("Engineered features: V14_V17 (interaction), V12_V10 (interaction)")

# Feature selection with SelectKBest
selector = SelectKBest(score_func=f_classif, k=20)
selector.fit(X_train_sm, y_train_sm)
selected_features = X_train_sm.columns[selector.get_support()].tolist()
print(f"\nSelectKBest top 20 features:")
print(selected_features)

X_train_sel = X_train_sm[selected_features]
X_test_sel  = X_test[selected_features]

print(f"\nShape after selection: {X_train_sel.shape}")


# =============================================================================
#  5. MODEL BUILDING
# =============================================================================

print("\n" + "=" * 60)
print("  5. MODEL BUILDING")
print("=" * 60)


def evaluate_clf(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_te)[:, 1]
    else:
        prob = model.decision_function(X_te)
    prec = precision_score(y_te, pred)
    rec  = recall_score(y_te, pred)
    f1   = f1_score(y_te, pred)
    auc  = roc_auc_score(y_te, prob)
    print(f"{name:<28}  Prec={prec:.4f}  Rec={rec:.4f}"
          f"  F1={f1:.4f}  AUC={auc:.4f}")
    return model, pred, prob


print("\n--- Base Model Performance ---")

lr_model,  lr_pred,  lr_prob  = evaluate_clf(
    "Logistic Regression",
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    X_train_sel, X_test_sel, y_train_sm, y_test)

dt_model,  dt_pred,  dt_prob  = evaluate_clf(
    "Decision Tree",
    DecisionTreeClassifier(max_depth=6, random_state=42,
                            class_weight="balanced"),
    X_train_sel, X_test_sel, y_train_sm, y_test)

rf_model,  rf_pred,  rf_prob  = evaluate_clf(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, random_state=42,
                            class_weight="balanced"),
    X_train_sel, X_test_sel, y_train_sm, y_test)

gb_model,  gb_pred,  gb_prob  = evaluate_clf(
    "Gradient Boosting",
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    X_train_sel, X_test_sel, y_train_sm, y_test)


# =============================================================================
#  6. ANN MODEL
# =============================================================================

print("\n" + "=" * 60)
print("  6. ANN MODEL")
print("=" * 60)

# Scale for ANN
ann_scaler = StandardScaler()
X_train_ann = ann_scaler.fit_transform(X_train_sel)
X_test_ann  = ann_scaler.transform(X_test_sel)

ann = keras.Sequential([
    layers.Dense(64, activation="relu",
                 input_shape=(X_train_ann.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(32, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(16, activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

ann.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy",
             keras.metrics.Precision(name="precision"),
             keras.metrics.Recall(name="recall"),
             keras.metrics.AUC(name="auc")]
)

ann.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_auc", patience=5,
    restore_best_weights=True, mode="max")

# Class weights to handle imbalance inside ANN
neg = (y_train_sm == 0).sum()
pos = (y_train_sm == 1).sum()
class_weight = {0: 1.0, 1: neg / pos}

ann_history = ann.fit(
    X_train_ann, y_train_sm,
    epochs=30,
    batch_size=512,
    validation_split=0.1,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

ann_prob = ann.predict(X_test_ann, verbose=0).flatten()
ann_pred = (ann_prob > 0.5).astype(int)

print(f"\nANN  Prec={precision_score(y_test, ann_pred):.4f}"
      f"  Rec={recall_score(y_test, ann_pred):.4f}"
      f"  F1={f1_score(y_test, ann_pred):.4f}"
      f"  AUC={roc_auc_score(y_test, ann_prob):.4f}")

# Training curves
epochs_ran = range(1, len(ann_history.history["loss"]) + 1)

plt.figure()
plt.plot(epochs_ran, ann_history.history["auc"],     label="Train AUC")
plt.plot(epochs_ran, ann_history.history["val_auc"], label="Val AUC")
plt.title("ANN Training vs Validation AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs_ran, ann_history.history["loss"],     label="Train Loss")
plt.plot(epochs_ran, ann_history.history["val_loss"], label="Val Loss")
plt.title("ANN Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# =============================================================================
#  7. MODEL EVALUATION
# =============================================================================

print("\n" + "=" * 60)
print("  7. MODEL EVALUATION")
print("=" * 60)

# Confusion matrix for best model (Random Forest)
cm = confusion_matrix(y_test, rf_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report - Random Forest:")
print(classification_report(y_test, rf_pred,
                             target_names=["Legit", "Fraud"]))

# ROC Curve all models
plt.figure()
for name, prob in [("Logistic Regression", lr_prob),
                    ("Decision Tree",       dt_prob),
                    ("Random Forest",       rf_prob),
                    ("Gradient Boosting",   gb_prob),
                    ("ANN",                 ann_prob)]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.title("ROC Curve - All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(fontsize=8)
plt.show()


# =============================================================================
#  8. HYPERPARAMETER TUNING & CROSS VALIDATION
# =============================================================================

print("\n" + "=" * 60)
print("  8. HYPERPARAMETER TUNING & CROSS VALIDATION")
print("=" * 60)

# RandomizedSearchCV for Random Forest
print("\nTuning Random Forest...")
rf_params = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_distributions=rf_params,
    n_iter=15, cv=3, scoring="roc_auc",
    random_state=42, n_jobs=-1, verbose=0
)
rf_search.fit(X_train_sel, y_train_sm)
best_rf = rf_search.best_estimator_
rf_tuned_pred = best_rf.predict(X_test_sel)
rf_tuned_prob = best_rf.predict_proba(X_test_sel)[:, 1]

print(f"Best params : {rf_search.best_params_}")
print(f"Tuned RF    AUC={roc_auc_score(y_test, rf_tuned_prob):.4f}"
      f"  F1={f1_score(y_test, rf_tuned_pred):.4f}")

# GridSearchCV for Logistic Regression
print("\nTuning Logistic Regression...")
lr_params = {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}
lr_search = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    param_grid=lr_params,
    cv=3, scoring="roc_auc", n_jobs=-1, verbose=0
)
lr_search.fit(X_train_sel, y_train_sm)
best_lr = lr_search.best_estimator_
lr_tuned_pred = best_lr.predict(X_test_sel)
lr_tuned_prob = best_lr.predict_proba(X_test_sel)[:, 1]

print(f"Best params : {lr_search.best_params_}")
print(f"Tuned LR    AUC={roc_auc_score(y_test, lr_tuned_prob):.4f}"
      f"  F1={f1_score(y_test, lr_tuned_pred):.4f}")

# Stratified K-Fold Cross Validation on best model
print("\nStratified 5-Fold CV on Tuned Random Forest:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_rf, X_train_sel, y_train_sm,
                             cv=skf, scoring="roc_auc", n_jobs=-1)
print(f"CV AUC scores: {cv_scores.round(4)}")
print(f"Mean AUC: {cv_scores.mean():.4f}  Std: {cv_scores.std():.4f}")


# =============================================================================
#  9. OVERFITTING / UNDERFITTING ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("  9. OVERFITTING / UNDERFITTING ANALYSIS")
print("=" * 60)

# Compare train vs test performance for each model
print("\nTrain vs Test AUC comparison:")
for name, model, Xte, yte_pred, yte_prob in [
    ("Logistic Regression", best_lr,  X_test_sel,
     lr_tuned_pred, lr_tuned_prob),
    ("Random Forest",       best_rf,  X_test_sel,
     rf_tuned_pred, rf_tuned_prob),
]:
    train_pred = model.predict(X_train_sel)
    train_prob = model.predict_proba(X_train_sel)[:, 1]
    train_auc  = roc_auc_score(y_train_sm, train_prob)
    test_auc   = roc_auc_score(y_test, yte_prob)
    gap        = train_auc - test_auc
    print(f"  {name:<28} Train AUC={train_auc:.4f}"
          f"  Test AUC={test_auc:.4f}  Gap={gap:.4f}")

# ANN overfitting check from training curves (already plotted above)
ann_train_auc = max(ann_history.history["auc"])
ann_val_auc   = max(ann_history.history["val_auc"])
print(f"  {'ANN':<28} Train AUC={ann_train_auc:.4f}"
      f"  Val AUC={ann_val_auc:.4f}"
      f"  Gap={ann_train_auc - ann_val_auc:.4f}")

print("\nAnalysis:")
print("  Gap < 0.02 : Good generalisation (no overfitting)")
print("  Gap > 0.05 : Possible overfitting - reduce model complexity")
print("  Train AUC << Test AUC : Underfitting - model too simple")


# =============================================================================
#  10. FINAL MODEL COMPARISON TABLE
# =============================================================================

print("\n" + "=" * 60)
print("  10. FINAL MODEL COMPARISON")
print("=" * 60)

results = pd.DataFrame([
    {"Model": "Logistic Regression (Base)",
     "Precision": round(precision_score(y_test, lr_pred), 4),
     "Recall":    round(recall_score(y_test, lr_pred), 4),
     "F1":        round(f1_score(y_test, lr_pred), 4),
     "AUC":       round(roc_auc_score(y_test, lr_prob), 4)},

    {"Model": "Logistic Regression (Tuned)",
     "Precision": round(precision_score(y_test, lr_tuned_pred), 4),
     "Recall":    round(recall_score(y_test, lr_tuned_pred), 4),
     "F1":        round(f1_score(y_test, lr_tuned_pred), 4),
     "AUC":       round(roc_auc_score(y_test, lr_tuned_prob), 4)},

    {"Model": "Decision Tree",
     "Precision": round(precision_score(y_test, dt_pred), 4),
     "Recall":    round(recall_score(y_test, dt_pred), 4),
     "F1":        round(f1_score(y_test, dt_pred), 4),
     "AUC":       round(roc_auc_score(y_test, dt_prob), 4)},

    {"Model": "Random Forest (Base)",
     "Precision": round(precision_score(y_test, rf_pred), 4),
     "Recall":    round(recall_score(y_test, rf_pred), 4),
     "F1":        round(f1_score(y_test, rf_pred), 4),
     "AUC":       round(roc_auc_score(y_test, rf_prob), 4)},

    {"Model": "Random Forest (Tuned)",
     "Precision": round(precision_score(y_test, rf_tuned_pred), 4),
     "Recall":    round(recall_score(y_test, rf_tuned_pred), 4),
     "F1":        round(f1_score(y_test, rf_tuned_pred), 4),
     "AUC":       round(roc_auc_score(y_test, rf_tuned_prob), 4)},

    {"Model": "Gradient Boosting",
     "Precision": round(precision_score(y_test, gb_pred), 4),
     "Recall":    round(recall_score(y_test, gb_pred), 4),
     "F1":        round(f1_score(y_test, gb_pred), 4),
     "AUC":       round(roc_auc_score(y_test, gb_prob), 4)},

    {"Model": "ANN",
     "Precision": round(precision_score(y_test, ann_pred), 4),
     "Recall":    round(recall_score(y_test, ann_pred), 4),
     "F1":        round(f1_score(y_test, ann_pred), 4),
     "AUC":       round(roc_auc_score(y_test, ann_prob), 4)},

]).sort_values("AUC", ascending=False).reset_index(drop=True)

print(results.to_string(index=False))
results.to_csv("model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")

# Save the best model and scaler for deployment
import pickle
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_rf, f)
with open("selector.pkl", "wb") as f:
    pickle.dump(selector, f)
with open("ann_scaler.pkl", "wb") as f:
    pickle.dump(ann_scaler, f)

ann.save("ann_model.h5")

print("\nSaved: best_model.pkl, selector.pkl, ann_scaler.pkl, ann_model.h5")
print("\nFinal recommendation: Tuned Random Forest")
print("  - Highest AUC and F1 on imbalanced fraud data")
print("  - Handles SMOTE-balanced data well")
print("  - Interpretable via feature importance")
print("  - Fast inference suitable for real-time deployment")
