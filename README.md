# Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection

This college ML project compares two **unsupervised** anomaly detectors for credit-card fraud:

| Model | Anomaly score (higher = more suspicious) |
|---|---|
| Gaussian Mixture Model (scikit-learn) | negative log-likelihood, `-gmm.score_samples(x)` |
| Undercomplete autoencoder (PyTorch) | per-transaction reconstruction MSE |

- Both models are trained **only on legitimate transactions**.
- Alert thresholds are chosen on a **validation** set, using a 1% false-positive-rate budget.
- Both models are then evaluated on the **same untouched test set**.

## Project files

```
card-fraud-triage/
├── card_fraud_triage.ipynb   # the full project (sections A–H), with saved outputs
├── requirements.txt          # Python packages
└── README.md                 # this file
```

## 1. Setup

You need Python 3.10 or newer.

```bash
cd card-fraud-triage
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the dataset

The CSV is not included because it is about 150 MB.

1. Go to <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>. You need a free Kaggle account.
2. Click **Download** and unzip the archive.
3. Put `creditcard.csv` **in the `card-fraud-triage/` folder**, next to the notebook.

The notebook also looks one folder up (`../creditcard.csv`). If neither path exists, it stops with a clear error message.

**If you cannot get the dataset:**

- Set `USE_SYNTHETIC_DEMO = True` in the loading cell (Section A).
- The notebook then creates a small *fake* dataset with the same columns, so you can check that every cell runs.
- Those numbers are meaningless. Do **not** report them as results.

## 3. Run it

```bash
jupyter notebook card_fraud_triage.ipynb
```

Then choose **Kernel → Restart & Run All**.

- A full run takes about 1–2 minutes on a modern laptop CPU. No GPU is needed.
- Most of that time is spent training the four autoencoder configurations.

## 4. What the notebook does

| Section | Content |
|---|---|
| A | Load the data; check shape, missing values, duplicates and class balance; explain why accuracy misleads |
| B | Drop duplicates, drop `Time`, apply `log1p(Amount)`; stratified 60/20/20 split; fit the scaler on legitimate training rows only |
| C | Fit GMMs with K = 1, 2, 3, 4, 6, 8 and pick K by the lowest BIC; score = −log-likelihood |
| D | Train 29→16→b→16→29 autoencoders on legitimate rows, with early stopping on a legitimate-only holdout; run the baseline plus Experiments A and B; select the configuration by validation AP |
| E | Set each threshold at the 99th percentile of legitimate validation scores (1% FPR budget) for the GMM, the baseline AE and the selected AE |
| F | Test-set average precision, recall at 1% test FPR, recall at the validation threshold, precision@100, confusion matrices, comparison table |
| G | PR curves, recall-vs-FPR and review-capacity plots, a fraud-by-fraud check of missed frauds, honest interpretation |
| H | Conclusions, limitations, future work |

## 5. Key design decisions (for the viva)

- **No leakage.**
  - Duplicates are removed *before* splitting.
  - The scaler and both models are fitted only on legitimate training rows.
  - Thresholds use only validation data.
  - The test set is used once, at the end.
- **`Class` is never a model input.** Fraud rows in the training split are discarded.
- **`Time` is excluded.** It counts seconds since the start of a two-day recording, which says nothing about the transaction itself.
- **Random split, not chronological.** This is stated clearly in the notebook. Results show performance on the same two days, not on future data.
- **GMM `reg_covar = 1e-3`.** With the default value, some components collapsed onto near-singular directions and BIC never reached a minimum.
- **Same score direction everywhere.** Higher score means more suspicious, and a transaction is flagged when `score > threshold`.
- **AE configuration chosen by validation AP.**
  - The networks are trained only on legitimate rows.
  - The *validation* fraud labels are used to choose between configurations, which is hyperparameter selection.
  - The test set is never used for a choice.
- **Two different "recall" numbers.**
  - *Recall @ 1% FPR (test ROC)* uses a cut-off derived from the test labels. It measures ranking quality and is not achievable in practice.
  - *Recall @ val threshold* uses the threshold fixed on validation data. It is what a deployed system would get.

## 6. Autoencoder experiments

The original autoencoder (bottleneck 8, 50 epochs, patience 5) was still improving when it hit its epoch cap. We tested:

- **Experiment A:** 100 epochs and patience 8, with the same bottleneck of 8.
- **Experiment B:** bottlenecks of 4, 8 and 12, each with 100 epochs and patience 8.

Everything else was kept the same: data split, features, hidden size 16, Adam with learning rate 1e-3, batch size 512, and seed 42. The baseline was re-run with the same code and reproduced the recorded results exactly.

| Config | Bottleneck | Max epochs / patience | Best early-stop MSE (epoch) | Stopped by | Validation AP |
|---|---|---|---|---|---|
| AE-base | 8 | 50 / 5 | 0.4353 (49) | epoch cap | 0.405 |
| **AE-b8** (selected) | 8 | 100 / 8 | 0.4256 (99) | epoch cap | **0.413** |
| AE-b4 | 4 | 100 / 8 | 0.5055 (100) | epoch cap | 0.377 |
| AE-b12 | 12 | 100 / 8 | 0.4023 (65) | early stopping (epoch 73) | 0.282 |

**Findings**

- Longer training helped validation AP only slightly (+0.008).
- The largest bottleneck had the *lowest* reconstruction loss but the *worst* fraud ranking: a lower loss does not mean better detection.
- **AE-b8** was selected because it had the highest validation AP.

## 7. Results from our run

These come from `SEED = 42` on the full dataset. Re-run the notebook to reproduce or check them.

Test set: 56,746 transactions, 95 of them fraud.

| Metric (test) | GMM (K = 6) | AE baseline | AE selected (AE-b8) |
|---|---|---|---|
| Average precision (PR-AUC) | 0.473 | 0.357 | 0.358 |
| Recall @ 1% FPR (test ROC, test-derived cut-off) | 0.779 | 0.779 | 0.779 |
| Precision @ top-100 | 0.49 | 0.41 | 0.41 |
| Recall @ validation threshold | 0.779 (74/95) | 0.779 (74/95) | 0.779 (74/95) |
| Precision @ validation threshold | 0.107 | 0.112 | 0.111 |
| Test FPR @ validation threshold | 1.09% | 1.04% | 1.05% |
| Transactions flagged | 694 | 663 | 667 |

**Baseline AE compared with the selected AE**

- The small validation gain did not carry over to the test set: test AP went from 0.3568 to 0.3575, and every other metric is the same or differs by a few flagged transactions.
- The tuning neither improved nor worsened the autoencoder in any meaningful way.
- The GMM remains the best ranker.

**Missed frauds, checked fraud by fraud**

- The GMM and the selected AE each miss 21 test frauds, and 20 of those are the same transactions.
- The baseline and selected AEs miss exactly the same 21.

There are only 95 test frauds and a single split, so treat small differences as noise.

## 8. Deliverables

1. `card_fraud_triage.ipynb`: the complete, executed notebook.
2. `requirements.txt`
3. `README.md`
