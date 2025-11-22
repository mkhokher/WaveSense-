import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, f1_score, make_scorer
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ===================== CONFIG =====================
COMBINED_PATH = "/Users/zaid_21/Desktop/Capstone/all_gestures_csv/combined.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV = 3  # cross-val folds

# Small, fast grids (expand later if needed)
SVM_GRID = {
    "clf__estimator__C": [1, 5],
    "clf__estimator__gamma": ["scale", 0.1],
}
RF_GRID = {
    "clf__estimator__n_estimators": [300],
    "clf__estimator__max_depth": [None, 20],
    "clf__estimator__min_samples_leaf": [1, 3],
}
LOGREG_GRID = {
    "clf__estimator__C": [0.5, 1, 2],
    "clf__estimator__penalty": ["l2"],
    "clf__estimator__solver": ["lbfgs"],
}

# ============== 1) LOAD & BASIC CLEANING ==============
df = pd.read_csv(COMBINED_PATH)
if "gesture" not in df.columns:
    raise ValueError("combined.csv must contain a 'gesture' column.")

# Keep small categoricals if present; everything else will be coerced numeric
candidate_cats = [c for c in ["type", "role"] if c in df.columns]
cat_df = df[candidate_cats].copy() if candidate_cats else pd.DataFrame(index=df.index)

# Coerce non-categorical feature columns (except gesture) to numeric
feature_cols = [c for c in df.columns if c not in (["gesture"] + candidate_cats)]
num_block = df[feature_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))

# Drop numeric columns that are entirely NaN after coercion (avoids imputer shape errors)
num_block = num_block.loc[:, num_block.notna().any(axis=0)]

# Build raw X; target y (strings)
X_raw = pd.concat([cat_df, num_block], axis=1)
y = df["gesture"].astype(str).values

# ============== 2) PREPROCESSOR ==============
cat_cols = list(cat_df.columns) if not cat_df.empty else []
num_cols = list(num_block.columns)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols),
    ],
    remainder="drop"
)

# Train/test split (stratified)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ============== 3) SAFE MULTICLASS SCORER (macro-F1) ==============
def macro_f1_multi(y_true, y_pred):
    # Works for string or numeric multiclass labels
    return f1_score(y_true, y_pred, average="macro")

SCORER = make_scorer(macro_f1_multi)

# ============== 4) CANDIDATE OVR MODELS ==============
pipelines = {
    "svm_rbf": Pipeline([
        ("prep", preprocessor),
        ("clf", OneVsRestClassifier(
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            n_jobs=None
        ))
    ]),
    "rf": Pipeline([
        ("prep", preprocessor),
        ("clf", OneVsRestClassifier(
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            n_jobs=None
        ))
    ]),
    "logreg": Pipeline([
        ("prep", preprocessor),
        ("clf", OneVsRestClassifier(
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            n_jobs=None
        ))
    ])
}

param_grids = {
    "svm_rbf": SVM_GRID,
    "rf": RF_GRID,
    "logreg": LOGREG_GRID
}

# ============== 5) GRID SEARCH (macro-F1) ==============
best_name, best_model, best_cv = None, None, -np.inf

for name, pipe in pipelines.items():
    print(f"\n>>> Tuning {name} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence convergence warnings, etc.
        grid = GridSearchCV(pipe, param_grids[name], scoring=SCORER, cv=CV, n_jobs=-1, verbose=0)
        grid.fit(X_train_raw, y_train)
    print(f"  best CV macro-F1: {grid.best_score_:.4f}")
    print(f"  best params: {grid.best_params_}")
    if grid.best_score_ > best_cv:
        best_cv = grid.best_score_
        best_name = name
        best_model = grid.best_estimator_

print(f"\n*** Selected model: {best_name} (CV macro-F1={best_cv:.4f}) ***")

# ============== 6) FINAL FIT & EVALUATION ==============
best_model.fit(X_train_raw, y_train)
y_pred = best_model.predict(X_test_raw)

print("\n=== Test Accuracy ===")
print(f"{accuracy_score(y_test, y_pred):.4f}")

print("\n=== Classification Report (per gesture) ===")
print(classification_report(y_test, y_pred, digits=4))

# Confusion Matrix
cm_labels = np.unique(np.concatenate([y_test, y_pred]))
cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
plt.figure(figsize=(6.5, 5.5))
plt.imshow(cm, interpolation="nearest")
plt.title(f"Confusion Matrix (OvR — {best_name})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(ticks=range(len(cm_labels)), labels=cm_labels, rotation=45, ha="right")
plt.yticks(ticks=range(len(cm_labels)), labels=cm_labels)
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.tight_layout()
plt.show()

# ============== 7) ROC CURVES (One-vs-Rest) ==============
# Transform test features through the selected pipeline's preprocessor
X_test = best_model.named_steps["prep"].transform(X_test_raw)
ovr_est = best_model.named_steps["clf"]

# Scores: decision_function preferred; else predict_proba
if hasattr(ovr_est, "decision_function"):
    scores = ovr_est.decision_function(X_test)  # (n_samples, n_classes)
else:
    scores = ovr_est.predict_proba(X_test)

# Binarize true labels for OvR ROC
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)  # (n_samples, n_classes), 0/1 per class

# If a single column, keep 2D shape
if scores.ndim == 1:
    scores = scores.reshape(-1, 1)

classes = lb.classes_
plt.figure(figsize=(7, 6))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.title(f"ROC Curves — One-vs-Rest ({best_name})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.show()
