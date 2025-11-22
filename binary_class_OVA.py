import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)

# ===== CONFIG =====
COMBINED_PATH = "/Users/zaid_21/Desktop/Capstone/all_gestures_csv/combined.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ===== 1) Load =====
df = pd.read_csv(COMBINED_PATH)
if "gesture" not in df.columns:
    raise ValueError("combined.csv must contain a 'gesture' column.")

# Keep candidate small categoricals if present; everything else becomes numeric if possible
candidate_cats = [c for c in ["type", "role"] if c in df.columns]

# Coerce non-categorical columns (except gesture) to numeric
feature_cols = [c for c in df.columns if c not in (["gesture"] + candidate_cats)]
num_block = df[feature_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))

# Drop numeric columns that are entirely NaN after coercion (bad/empty features)
num_block = num_block.loc[:, num_block.notna().any(axis=0)]

# Rebuild a working features frame (categoricals separate)
cat_df = df[candidate_cats].copy() if candidate_cats else pd.DataFrame(index=df.index)
X_raw = pd.concat([cat_df, num_block], axis=1)

# Target labels
y_gesture = df["gesture"].astype(str).values

# ===== 2) Preprocess (impute → encode/scale) =====
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

# Train/test split (stratified by gesture)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_gesture, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_gesture
)

# Fit transforms
X_train = preprocessor.fit_transform(X_train_raw)
X_test  = preprocessor.transform(X_test_raw)

# ===== 3) One-vs-Rest Logistic Regression =====
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

ovr = OneVsRestClassifier(LogisticRegression(max_iter=2000))
ovr.fit(X_train, y_train_enc)

# ===== 4) Evaluation =====
y_pred_enc = ovr.predict(X_test)
y_pred = le.inverse_transform(y_pred_enc)

print("\n=== Accuracy ===")
print(f"{accuracy_score(y_test, y_pred):.4f}")

print("\n=== Classification Report (per gesture) ===")
print(classification_report(y_test, y_pred, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (OvR Logistic)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(ticks=range(len(le.classes_)), labels=le.classes_, rotation=45, ha="right")
plt.yticks(ticks=range(len(le.classes_)), labels=le.classes_)
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.tight_layout()
plt.show()

# ===== 5) ROC Curves (one-vs-rest, one line per gesture) =====
# Use decision_function if available; fallback to predict_proba
if hasattr(ovr, "decision_function"):
    scores = ovr.decision_function(X_test)  # shape (n_samples, n_classes)
else:
    scores = ovr.predict_proba(X_test)

lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test_enc)   # shape (n_samples, n_classes); one-vs-rest ground truth

plt.figure(figsize=(7,6))
for i, cls_name in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{cls_name} (AUC = {roc_auc:.3f})")

# Chance line
plt.plot([0,1], [0,1], linestyle="--")
plt.title("ROC Curves — One-vs-Rest (per gesture)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.show()

# ===== 6) Quick summary =====
print("\nClasses (gesture order used internally):", list(le.classes_))
