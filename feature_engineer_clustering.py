import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import KernelPCA

# ===== PATH =====
combined_path = "/Users/zaid_21/Desktop/Capstone/all_gestures_csv/combined.csv"

# ===== 1) LOAD =====
df = pd.read_csv(combined_path)
if "gesture" not in df.columns:
    raise ValueError("combined.csv must contain a 'gesture' column.")

# Drop obvious IDs/blobs if present
df = df.drop(columns=[c for c in ["mac","CSI_DATA","real_timestamp","local_timestamp"] if c in df.columns],
             errors="ignore")

# Keep small categoricals aside BEFORE any coercion
candidate_cats = [c for c in ["type","role"] if c in df.columns]
cat_df = df[candidate_cats].copy() if candidate_cats else pd.DataFrame(index=df.index)

# Coerce everything else (except gesture + candidate_cats) to numeric
work_cols = [c for c in df.columns if c not in (["gesture"] + candidate_cats)]
num_block = df[work_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))

# Drop numeric columns that are all-NaN after coercion
num_block = num_block.loc[:, num_block.notna().any(axis=0)]
numeric_cols = list(num_block.columns)

if len(numeric_cols) == 0 and (cat_df is None or cat_df.shape[1] == 0):
    raise ValueError("No usable features left after cleaning. Check your data.")

# Impute numerics FIRST (avoids NaNs for PolynomialFeatures)
num_imputer = SimpleImputer(strategy="median")
num_imputed_arr = num_imputer.fit_transform(num_block)
num_imputed = pd.DataFrame(num_imputed_arr, columns=numeric_cols, index=df.index)

# ===== 2) FEATURE ENGINEERING (on imputed numerics) =====
fe = num_imputed.copy()

# Z-scores (if present)
if "rssi" in fe.columns:
    fe["rssi_z"] = (fe["rssi"] - fe["rssi"].mean()) / (fe["rssi"].std() + 1e-9)
if "rate" in fe.columns:
    fe["rate_z"] = (fe["rate"] - fe["rate"].mean()) / (fe["rate"].std() + 1e-9)
if "noise_floor" in fe.columns:
    fe["noise_z"] = (fe["noise_floor"] - fe["noise_floor"].mean()) / (fe["noise_floor"].std() + 1e-9)

# Ratios / differences
if set(["rssi","rate"]).issubset(fe.columns):
    fe["rssi_rate_ratio"] = fe["rssi"] / (fe["rate"] + 1e-6)
if set(["rssi","noise_floor"]).issubset(fe.columns):
    fe["rssi_noise_diff"] = fe["rssi"] - fe["noise_floor"]
if set(["rate","noise_floor"]).issubset(fe.columns):
    fe["rate_noise_ratio"] = fe["rate"] / (np.abs(fe["noise_floor"]) + 1e-6)

# Row-wise stats on original numeric block
orig = num_imputed
fe["feat_mean"]  = orig.mean(axis=1)
fe["feat_std"]   = orig.std(axis=1)
fe["feat_max"]   = orig.max(axis=1)
fe["feat_min"]   = orig.min(axis=1)
fe["feat_range"] = fe["feat_max"] - fe["feat_min"]

# Small polynomial expansion on stable base cols (only if >=2 exist)
poly_base_cols = [c for c in ["rssi","rate","noise_floor","feat_mean","feat_std","feat_range"] if c in fe.columns]
if len(poly_base_cols) >= 2:
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_vals = poly.fit_transform(fe[poly_base_cols])
    poly_names = poly.get_feature_names_out(poly_base_cols)
    poly_df = pd.DataFrame(poly_vals, columns=[f"poly_{n}" for n in poly_names], index=fe.index)
    fe = pd.concat([fe, poly_df], axis=1)

# ===== 3) PREPROCESS: one-hot cats + scale nums =====
# Fill missing categoricals (if any)
if not cat_df.empty:
    cat_df = cat_df.fillna("unknown")
cat_cols = list(cat_df.columns)

pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), list(fe.columns)),
    ],
    remainder="drop"
)

X_source = pd.concat([cat_df, fe], axis=1) if not cat_df.empty else fe
X = pre.fit_transform(X_source)

y_gesture = df["gesture"].astype(str).values
gestures = sorted(pd.unique(y_gesture))
K = len(gestures)

# ===== 4) KMEANS =====
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Map cluster -> majority gesture
cluster_to_gesture = {}
for k in range(K):
    mask = (y_kmeans == k)
    cluster_to_gesture[k] = pd.Series(y_gesture[mask]).mode()[0] if mask.any() else f"cluster_{k}"
y_kmeans_as_gesture = np.array([cluster_to_gesture[c] for c in y_kmeans])

# ===== 5) OVR Logistic =====
le = LabelEncoder()
y_codes = le.fit_transform(y_gesture)
ovr = OneVsRestClassifier(LogisticRegression(max_iter=2000))
ovr.fit(X, y_codes)
y_pred_codes = ovr.predict(X)
y_pred_gesture = le.inverse_transform(y_pred_codes)

# ===== 6) KernelPCA (nonlinear 2D) =====
X_dense = X.toarray() if hasattr(X, "toarray") else X
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.05, random_state=42)
X2 = kpca.fit_transform(X_dense)

# ===== 7) Plot (two panels, consistent colors + labels) =====
cmap = plt.cm.get_cmap("tab10", K)
palette = {g: cmap(i) for i, g in enumerate(gestures)}
def cols(lbls): return [palette[g] for g in lbls]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("CSI Gesture Clustering — Fixed NaNs, Strong FE, KMeans vs OvR", fontsize=14, weight="bold")

# KMeans
axes[0].scatter(X2[:,0], X2[:,1], c=cols(y_kmeans_as_gesture), s=10)
axes[0].set_title("KMeans (colored by majority gesture)")
axes[0].set_xlabel("F1"); axes[0].set_ylabel("F2")

# OvR
axes[1].scatter(X2[:,0], X2[:,1], c=cols(y_pred_gesture), s=10)
axes[1].set_title("OvR Predictions (colored by gesture)")
axes[1].set_xlabel("F1"); axes[1].set_ylabel("F2")

from matplotlib.lines import Line2D
handles = [Line2D([0],[0], marker='o', linestyle='', color=palette[g], label=g) for g in gestures]
fig.legend(handles=handles, loc="upper center", ncol=min(5, K), frameon=False)

plt.tight_layout(rect=[0,0,1,0.92])
plt.show()

# Quick sanity
print("Gestures:", gestures)
print("Cluster → majority gesture mapping:")
for c, g in cluster_to_gesture.items():
    print(f"  Cluster {c} → {g}")
