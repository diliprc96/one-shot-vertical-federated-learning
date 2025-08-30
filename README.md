# Vertical Federated Learning with Limited Overlap

This repository contains a simplified implementation of **communication-efficient Vertical Federated Learning (VFL)** inspired by the paper *“Communication-Efficient Vertical Federated Learning with Limited Overlapping Samples”* (Sun et al.). The goal is to demonstrate how VFL can be performed in scenarios where:

* Multiple parties (clients) hold **different features** of the same population,
* Only a **small fraction of samples overlap** across clients,
* Communication between clients and server is **costly**, making one-shot or few-shot protocols preferable.

The implementation focuses on a **tabular, scikit-learn-based pipeline** that is lightweight and easy to extend.

---

## ✨ Features

* **Two-view dataset simulation**
  Splits a tabular dataset into two non-overlapping feature subsets to mimic the vertical data partition across organizations.

* **Overlap & non-overlap handling**
  Creates a small overlapping pool of samples shared between clients and additional **unaligned samples** that only one client owns.

* **Client-side encoders**
  Each client learns an unsupervised representation of its local features (via `StandardScaler + PCA`).

* **One-shot training pipeline**

  * **Round 1:** Clients upload overlap representations to the server.
  * **Server:** trains a preliminary classifier and produces per-sample signals (gradients/logits).
  * **Gradient clustering:** server signals are clustered (k-means) to form **temporary labels** for overlap.
  * **Clients:** perform semi-supervised learning with temporary labels and their unaligned data.
  * **Round 2:** Clients send updated overlap representations.
  * **Server:** trains the final classifier on updated embeddings.

* **Privacy hook**
  Demonstrates **pseudo-encryption** with additive noise masking during communication.

* **Evaluation**
  Reports accuracy on a hold-out test set (with full overlap) along with communication summary (rounds + payload size).

---

## 🏗 Project Structure

```
.
├── one_shot_vfl_simplified.py   # Standalone script implementation
├── one_shot_VFL.ipynb           # Jupyter notebook version with experiments
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Install dependencies:

  ```bash
  pip install numpy scikit-learn
  ```

### Run the script

```bash
python one_shot_vfl_simplified.py
```

Expected output includes:

* Test accuracy on the hold-out set
* Number of overlapping vs. unaligned samples
* Communication rounds and approximate payload size

### Run the notebook

Open the notebook in Jupyter or VSCode to reproduce the experiments and modify overlap ratios, augmentation noise, or dataset.

---

## 📊 Example Results

```
=== One-Shot VFL (simplified) — Results ===
{
  "n_train": 124,
  "n_overlap": 43,
  "n_unaligned_A": 40,
  "n_unaligned_B": 41,
  "clientA_embed_dim": 5,
  "clientB_embed_dim": 5,
  "classes": 3,
  "comm_rounds": 3,
  "approx_payload_MB": 0.0043,
  "test_accuracy": 0.7778
}
```

---

## 🔬 Key Insights

* **One-shot VFL drastically reduces communication**: only 2 uploads and 1 download are required.
* **Gradient clustering + local semi-supervised learning** allows clients to make use of unlabeled and unaligned samples.
* **Overlap size strongly affects accuracy**: smaller overlaps benefit more from the one-shot approach compared to traditional iterative VFL.
* This simplified version demonstrates the **core intuition** without heavy deep learning frameworks, making it suitable for rapid experimentation on tabular datasets.

---

## 📈 Extensions

You can extend this work by:

* Trying different datasets (e.g., UCI tabular sets, healthcare or finance data).
* Implementing alternative encoders (MLP autoencoders instead of PCA).
* Comparing with iterative VFL baselines to measure accuracy vs. communication trade-offs.
* Adding stronger privacy techniques (secure aggregation, homomorphic encryption, DP noise).

---

## 📚 References

* Sun, Shaofeng, et al. *Communication-Efficient Vertical Federated Learning with Limited Overlapping Samples.*
* Xu, et al. *FedCVT: Semi-Supervised Vertical Federated Learning with Cross-View Training.*

