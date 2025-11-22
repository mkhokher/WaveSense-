import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA

# ====== PATH TO YOUR COMBINED CSV ======
combined_path = "/Users/zaid_21/Desktop/Capstone/all_gestures_csv/combined.csv"

# ====== LOAD ======
df = pd.read_csv(combined_path)
if "gesture" not in df.columns:
    raise ValueError("combined.csv must have a 'gesture' column (gesture label from folder name).")

# ====== FEATURE PREP ======
DROP_COLS = ("mac", "CSI_DATA")            # common ID/blob fields to drop if present
CANDIDATE_CATS = ("type", "role")          # small categorical fields (optional)

def infer_column_types(df, candidate_cats=CANDIDATE_CATS, drop_defaults=DROP_COLS):
    cat_cols = [c for c in candidate_cats if c in df.columns]
    drop_cols = [c for c in drop_defaults if c in df.columns]
    numeric_cols = []
    for c in df.columns:
        if c in cat_cols or c in drop_cols or c == "gesture":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().sum() >= max(1, int(0.8 * len(df))):
                df[c] = coerced
                numeric_cols.append(c)
    return df, cat_cols, numeric_cols, drop_cols

def build_preprocessor(cat_cols, numeric_cols):
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, numeric_cols),
    ], remainder="drop")

df, cat_cols, numeric_cols, drop_cols = infer_column_types(df)
if len(cat_cols) + len(numeric_cols) == 0:
    raise ValueError("No usable features detected after inference. Check your columns.")

df_model = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
X_source = df_model[cat_cols + numeric_cols].copy()
y_gesture = df_model["gesture"].astype(str).values

preprocessor = build_preprocessor(cat_cols, numeric_cols)
X = preprocessor.fit_transform(X_source)

# ====== KMEANS (K = number of gestures) ======
gestures = sorted(pd.unique(y_gesture))
K = len(gestures)

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Map cluster -> majority gesture (so colors reflect gestures, not arbitrary cluster IDs)
cluster_to_gesture = {}
for k in range(K):
    mask = (y_kmeans == k)
    if mask.sum() == 0:
        cluster_to_gesture[k] = f"cluster_{k}"
    else:
        cluster_to_gesture[k] = pd.Series(y_gesture[mask]).mode()[0]
y_kmeans_as_gesture = np.array([cluster_to_gesture[c] for c in y_kmeans])

# ====== OVR LOGISTIC (predict gestures) ======
le = LabelEncoder()
y_codes = le.fit_transform(y_gesture)
ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
ovr.fit(X, y_codes)
y_pred_codes = ovr.predict(X)
y_pred_gesture = le.inverse_transform(y_pred_codes)

# ====== PCA FOR 2D PLOTTING ======
X_dense = X.toarray() if hasattr(X, "toarray") else X
pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(X_dense)

# ====== CONSISTENT COLORS PER GESTURE ======
# Use tab10 palette (enough for up to 10 gestures). Extend if needed.
cmap = plt.cm.get_cmap("tab10", K)
palette = {g: cmap(i) for i, g in enumerate(gestures)}

def colorize(labels):
    return [palette[g] for g in labels]

# ====== PLOTS (SIDE BY SIDE) ======
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("CSI Gestures — KMeans vs OvR (colored & labeled by gesture)", fontsize=14, weight="bold")

# Left: KMeans clusters colored by majority gesture per cluster
axes[0].scatter(X2[:, 0], X2[:, 1], c=colorize(y_kmeans_as_gesture), s=10)
axes[0].set_title("KMeans (k = #gestures) — colored by majority gesture")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

# Right: OvR predictions colored by predicted gesture
axes[1].scatter(X2[:, 0], X2[:, 1], c=colorize(y_pred_gesture), s=10)
axes[1].set_title("OvR Logistic Regression — colored by predicted gesture")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

# Shared legend with gesture labels & colors
from matplotlib.lines import Line2D
legend_handles = [Line2D([0], [0], marker='o', linestyle='',
                         markerfacecolor=palette[g], markeredgecolor='none', label=g)
                  for g in gestures]
fig.legend(handles=legend_handles, loc="upper center", ncol=min(5, K), frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# ====== (Optional) quick text summary ======
print("Gestures:", gestures)
print("Cluster -> majority gesture mapping:")
for c, g in cluster_to_gesture.items():
    print(f"  Cluster {c} -> {g}")
