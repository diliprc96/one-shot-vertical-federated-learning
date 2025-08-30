# one-shot-vertical-federated-learning
Simplified One‑Shot VFL (conceptual simulation) — sklearn/numpy version

# What this script does
# • Creates a 2‑view dataset from Wine (13 features → split to two clients).
# • Simulates overlapping samples (with labels on the server) and unaligned samples per client.
# • Clients learn unsupervised feature extractors (StandardScaler+PCA).
# • Server trains a linear softmax classifier on concatenated client embeddings (overlap only).
# • Server computes per‑sample gradients wrt embeddings and sends (masked) gradients to clients.
# • Each client runs KMeans on received gradients to assign temporary labels for overlapping rows.
# • Clients train a simple local classifier on (overlap + high‑confidence pseudo‑labeled unaligned) data
#   and use the classifier’s linear logits as *updated embeddings* to send back.
# • Server retrains on updated embeddings and evaluates on a fully‑overlapping test set.

# Notes
# -----
# • This captures the key ideas of Sun et al.'s "one‑shot VFL": (single download of gradients,
#   local semi‑/self‑supervised update, single re‑upload of representations, server‑side fine‑tune).
# • We also add a tiny additive "mask" to simulate dummy encryption during transmission.

# Outputs
# -------
# • Final test accuracy (server classifier on concatenated updated embeddings).
# • Communication summary (number of rounds + approximate payload sizes).
# • A reusable script is saved to /mnt/data/one_shot_vfl_simplified.py
