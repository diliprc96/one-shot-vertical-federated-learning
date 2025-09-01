# Requirements:
# pip install numpy scikit-learn

import json
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


RNG = np.random.default_rng(42)

# ========================
# Utility functions
# ========================
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def one_hot(y, C):
    m = np.zeros((y.shape[0], C), dtype=float)
    m[np.arange(y.shape[0]), y] = 1.0
    return m

def mask_and_unmask(arr, seed=1234, scale=1e-3):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, scale, size=arr.shape)
    masked = arr + noise
    return masked, noise

def approx_size_bytes(arrays):
    total = 0
    for a in arrays:
        if a is None:
            continue
        total += a.astype(np.float32).nbytes
    return total

def fit_encoder(X_local, n_components):
    scaler = StandardScaler().fit(X_local)
    Xs = scaler.transform(X_local)
    pca = PCA(n_components=n_components, random_state=0).fit(Xs)
    return scaler, pca

def transform_encoder(scaler, pca, X):
    return pca.transform(scaler.transform(X))

def local_phase(E_o, E_u, grad_o, C, selftrain_thr=0.85, random_state=0):
    # KMeans clustering on gradients
    km = KMeans(n_clusters=C, n_init=10, random_state=random_state)
    y_temp = km.fit_predict(grad_o)

    # Train classifier on overlap embeddings with temp labels
    local_clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=200, random_state=random_state
    ).fit(E_o, y_temp)

    # Self-training on unaligned
    if E_u.shape[0] > 0:
        prob_u = local_clf.predict_proba(E_u)
        y_u = np.argmax(prob_u, axis=1)
        conf = prob_u.max(axis=1)
        keep = conf >= selftrain_thr
        if np.any(keep):
            E_sup = np.vstack([E_o, E_u[keep]])
            y_sup = np.concatenate([y_temp, y_u[keep]])
            local_clf = LogisticRegression(
                multi_class="multinomial", solver="lbfgs", max_iter=200, random_state=random_state
            ).fit(E_sup, y_sup)

    # Return updated embeddings (logits)
    W_loc = local_clf.coef_.astype(np.float32)
    b_loc = local_clf.intercept_.astype(np.float32)
    Z_o = E_o @ W_loc.T + b_loc
    return Z_o, dict(W=W_loc, b=b_loc)

def local_logits(E, W, b):
    return E @ W.T + b


# ========================
# Build dataset
# ========================
data = load_wine()
X, y = data.data.astype(np.float32), data.target.astype(int)
C = len(np.unique(y))
n, d = X.shape   # 178 x 13

# Split features to Client A and B
idxA = np.arange(0, 6)
idxB = np.arange(6, d)
XA_full, XB_full = X[:, idxA], X[:, idxB]

# Train/test split
X_train_A, X_test_A, X_train_B, X_test_B, y_train, y_test = train_test_split(
    XA_full, XB_full, y, test_size=0.3, stratify=y, random_state=0
)
n_train = X_train_A.shape[0]

# ========================
# Main experiment loop
# ========================
for overlap_ratio in [0.1, 0.3, 0.5, 0.7]:
    print("\n==============================")
    print(f" Running with overlap_ratio = {overlap_ratio}")
    print("==============================")

    # --- Partition overlap vs unaligned ---
    n_overlap = int(round(overlap_ratio * n_train))
    perm = RNG.permutation(n_train)
    ov_idx = perm[:n_overlap]
    rest_idx = perm[n_overlap:]
    split_point = len(rest_idx) // 2
    ua_idx, ub_idx = rest_idx[:split_point], rest_idx[split_point:]

    XA_o, XB_o, y_o = X_train_A[ov_idx], X_train_B[ov_idx], y_train[ov_idx]
    XA_u, XB_u = X_train_A[ua_idx], X_train_B[ub_idx]

    # --- Client-side encoders ---
    dimA, dimB = 5, 5
    XA_local_all = np.vstack([XA_o, XA_u]) if XA_u.size else XA_o
    XB_local_all = np.vstack([XB_o, XB_u]) if XB_u.size else XB_o

    scA, pcaA = fit_encoder(XA_local_all, dimA)
    scB, pcaB = fit_encoder(XB_local_all, dimB)

    EA_o = transform_encoder(scA, pcaA, XA_o)
    EB_o = transform_encoder(scB, pcaB, XB_o)
    EA_u = transform_encoder(scA, pcaA, XA_u) if XA_u.size else np.zeros((0, dimA))
    EB_u = transform_encoder(scB, pcaB, XB_u) if XB_u.size else np.zeros((0, dimB))

    # --- Server initial training ---
    H_o = np.concatenate([EA_o, EB_o], axis=1)
    clf_server = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=200, random_state=0
    ).fit(H_o, y_o)

    W, b = clf_server.coef_.astype(np.float32), clf_server.intercept_.astype(np.float32)
    Z = H_o @ W.T + b
    P = softmax(Z)
    Delta = P - one_hot(y_o, C)
    grad_H = Delta @ W
    grad_A, grad_B = grad_H[:, :dimA], grad_H[:, dimA:]

    # --- Masking ---
    grad_A_masked, noiseA = mask_and_unmask(grad_A, seed=111)
    grad_B_masked, noiseB = mask_and_unmask(grad_B, seed=222)
    payload_up_1 = approx_size_bytes([EA_o, EB_o])
    payload_down = approx_size_bytes([grad_A_masked, grad_B_masked])

    grad_A_received, grad_B_received = grad_A_masked - noiseA, grad_B_masked - noiseB

    # --- Local phase ---
    ZA_o, debugA = local_phase(EA_o, EA_u, grad_A_received, C, random_state=0)
    ZB_o, debugB = local_phase(EB_o, EB_u, grad_B_received, C, random_state=1)

    # --- Upload updated reps ---
    ZA_o_masked, noiseZA = mask_and_unmask(ZA_o, seed=333)
    ZB_o_masked, noiseZB = mask_and_unmask(ZB_o, seed=444)
    payload_up_2 = approx_size_bytes([ZA_o_masked, ZB_o_masked])

    ZA_o_recv, ZB_o_recv = ZA_o_masked - noiseZA, ZB_o_masked - noiseZB
    H_o_updated = np.concatenate([ZA_o_recv, ZB_o_recv], axis=1)

    clf_server_2 = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=500, random_state=0
    ).fit(H_o_updated, y_o)

    # --- Test evaluation ---
    EA_test = transform_encoder(scA, pcaA, X_test_A)
    EB_test = transform_encoder(scB, pcaB, X_test_B)
    ZA_test = local_logits(EA_test, debugA["W"], debugA["b"])
    ZB_test = local_logits(EB_test, debugB["W"], debugB["b"])
    H_test = np.concatenate([ZA_test, ZB_test], axis=1)

    y_pred = clf_server_2.predict(H_test)
    acc = accuracy_score(y_test, y_pred)

    comm_rounds = 3
    payload_mb = (payload_up_1 + payload_down + payload_up_2) / (1024**2)

    summary = {
        "n_train": int(n_train),
        "n_overlap": int(n_overlap),
        "n_unaligned_A": int(XA_u.shape[0]),
        "n_unaligned_B": int(XB_u.shape[0]),
        "clientA_embed_dim": int(dimA),
        "clientB_embed_dim": int(dimB),
        "classes": int(C),
        "comm_rounds": comm_rounds,
        "approx_payload_MB": round(float(payload_mb), 4),
        "test_accuracy": round(float(acc), 4),
    }

    print("=== One-Shot VFL (simplified) — Results ===")
    print(json.dumps(summary, indent=2))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
