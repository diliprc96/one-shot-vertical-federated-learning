# Requirements

# numpy
# scikit-learn

# Run this command in terminal or bash `pip install numpy scikit-learn``

import os
import math
import json
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

RNG = np.random.default_rng(42)

# utilities


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def one_hot(y, C):
    m = np.zeros((y.shape[0], C), dtype=float)
    m[np.arange(y.shape[0]), y] = 1.0
    return m

def mask_and_unmask(arr, seed=1234, scale=1e-3):
    # Simulate additive masking using a shared RNG seed
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, scale, size=arr.shape)
    masked = arr + noise
    # On the "server" we can regenerate and subtract since seed is shared.
    return masked, noise

def approx_size_bytes(arrays):
    # Rough on-wire size (float64 assumed unless cast); we cast to float32 to be conservative
    total = 0
    for a in arrays:
        if a is None:
            continue
        total += a.astype(np.float32).nbytes
    return total


# Building a 2-view dataset


data = load_wine()
X, y = data.data.astype(np.float32), data.target.astype(int)
C = len(np.unique(y))
n, d = X.shape   # 178 x 13


# Split features to client A (first 6) and B (remaining 7)
idxA = np.arange(0, 6)
idxB = np.arange(6, d)

XA_full = X[:, idxA]
XB_full = X[:, idxB]


# Train/test split (test is fully overlapping for eval)
X_train_A, X_test_A, X_train_B, X_test_B, y_train, y_test = train_test_split(
    XA_full, XB_full, y, test_size=0.3, stratify=y, random_state=0
)

n_train = X_train_A.shape[0]

# Choose overlapping indices (e.g., 35% of train); rest become unaligned split across clients
overlap_ratio = 0.35
n_overlap = int(round(overlap_ratio * n_train))
perm = RNG.permutation(n_train)
ov_idx = perm[:n_overlap]
rest_idx = perm[n_overlap:]

# Split the remaining (non-overlap) rows equally as unaligned A or unaligned B
split_point = len(rest_idx) // 2
ua_idx = rest_idx[:split_point]  # only at A
ub_idx = rest_idx[split_point:]  # only at B


# Overlapping sets (both clients have these feature rows; labels live ONLY on server)
XA_o = X_train_A[ov_idx]
XB_o = X_train_B[ov_idx]
y_o  = y_train[ov_idx]  # lives on "server" only

# Unaligned sets
XA_u = X_train_A[ua_idx]  # only A has these
XB_u = X_train_B[ub_idx]  # only B has these


# Client‑side unsupervised encoders (StandardScaler + PCA)


def fit_encoder(X_local, n_components):
    scaler = StandardScaler().fit(X_local)
    Xs = scaler.transform(X_local)
    pca = PCA(n_components=n_components, random_state=0).fit(Xs)
    return scaler, pca

def transform_encoder(scaler, pca, X):
    return pca.transform(scaler.transform(X))


dimA, dimB = 5, 5

# Fit on each client's OWN local data (overlap + its unaligned portion)
XA_local_all = np.vstack([XA_o, XA_u]) if XA_u.size else XA_o
XB_local_all = np.vstack([XB_o, XB_u]) if XB_u.size else XB_o

scA, pcaA = fit_encoder(XA_local_all, dimA)
scB, pcaB = fit_encoder(XB_local_all, dimB)


EA_o = transform_encoder(scA, pcaA, XA_o)
EB_o = transform_encoder(scB, pcaB, XB_o)


EA_u = transform_encoder(scA, pcaA, XA_u) if XA_u.size else np.zeros((0, dimA), dtype=np.float32)
EB_u = transform_encoder(scB, pcaB, XB_u) if XB_u.size else np.zeros((0, dimB), dtype=np.float32)

# Server trains linear softmax on overlap embeddings


H_o = np.concatenate([EA_o, EB_o], axis=1)
clf_server = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", max_iter=200, random_state=0
).fit(H_o, y_o)


W = clf_server.coef_.astype(np.float32)      # shape [C, D]
b = clf_server.intercept_.astype(np.float32) # shape [C]


# Per‑sample gradient wrt input embeddings: grad_h = (softmax(z) - one_hot(y)) @ W
Z = H_o @ W.T + b
P = softmax(Z)
Y_onehot = one_hot(y_o, C).astype(np.float32)
Delta = (P - Y_onehot)  # [n_overlap, C]
grad_H = Delta @ W      # [n_overlap, D]


grad_A, grad_B = grad_H[:, :dimA], grad_H[:, dimA:]

# Mask (dummy encryption) on wire
grad_A_masked, noiseA = mask_and_unmask(grad_A, seed=111, scale=1e-3)
grad_B_masked, noiseB = mask_and_unmask(grad_B, seed=222, scale=1e-3)


payload_up_1 = approx_size_bytes([EA_o.astype(np.float32), EB_o.astype(np.float32)])
payload_down = approx_size_bytes([grad_A_masked.astype(np.float32), grad_B_masked.astype(np.float32)])

# "Clients" unmask upon receipt
grad_A_received = grad_A_masked - noiseA
grad_B_received = grad_B_masked - noiseB


# Clients: k‑means on gradients → temporary labels on overlap + local self‑training on unaligned


def local_phase(E_o, E_u, grad_o, C, selftrain_thr=0.85, random_state=0):
    # KMeans on gradients to get temp labels
    km = KMeans(n_clusters=C, n_init=10, random_state=random_state)
    y_temp = km.fit_predict(grad_o)

    # Supervised training on overlap embeddings → temp labels
    local_clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=200, random_state=random_state
    ).fit(E_o, y_temp)

    # Self‑training on unaligned (add confident predictions as extra labels)
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

    # Updated "embeddings" = pre‑softmax logits of local classifier
    #   z = E @ W_loc^T + b_loc
    W_loc = local_clf.coef_.astype(np.float32)      # [C, dim]
    b_loc = local_clf.intercept_.astype(np.float32) # [C]
    Z_o = E_o @ W_loc.T + b_loc  # [n_overlap, C]

    return Z_o, dict(local_clf=local_clf, W=W_loc, b=b_loc)


ZA_o, debugA = local_phase(EA_o, EA_u, grad_A_received, C, selftrain_thr=0.85, random_state=0)
ZB_o, debugB = local_phase(EB_o, EB_u, grad_B_received, C, selftrain_thr=0.85, random_state=1)



# Clients upload updated overlap representations (masked)


ZA_o_masked, noiseZA = mask_and_unmask(ZA_o, seed=333, scale=1e-3)
ZB_o_masked, noiseZB = mask_and_unmask(ZB_o, seed=444, scale=1e-3)

payload_up_2 = approx_size_bytes([ZA_o_masked.astype(np.float32), ZB_o_masked.astype(np.float32)])

# Server unmasks
ZA_o_recv = ZA_o_masked - noiseZA
ZB_o_recv = ZB_o_masked - noiseZB

H_o_updated = np.concatenate([ZA_o_recv, ZB_o_recv], axis=1)

# Retrain server classifier on updated embeddings
clf_server_2 = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", max_iter=500, random_state=0
).fit(H_o_updated, y_o)

# Evaluate on test set (fully overlapping)


EA_test = transform_encoder(scA, pcaA, X_test_A)
EB_test = transform_encoder(scB, pcaB, X_test_B)


# Clients apply their learned local transformations to produce updated test reps
def local_logits(E, W, b):
    return E @ W.T + b

ZA_test = local_logits(EA_test, debugA["W"], debugA["b"])
ZB_test = local_logits(EB_test, debugB["W"], debugB["b"])
H_test = np.concatenate([ZA_test, ZB_test], axis=1)

y_pred = clf_server_2.predict(H_test)
acc = accuracy_score(y_test, y_pred)


# summary


comm_rounds = 3  # upload initial reps, download gradients, upload updated reps
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
    "test_accuracy": round(float(acc), 4)
}

print("=== One‑Shot VFL (simplified) — Results ===")
print(json.dumps(summary, indent=2))
print("\nClassification report (server on updated embeddings):")
print(classification_report(y_test, y_pred, digits=4))
