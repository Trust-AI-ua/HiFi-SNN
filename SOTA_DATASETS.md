# SOTA-comparison datasets — taxonomy, split, preprocessing, metrics

Reference for the three modern IoT/IoMT datasets added for the SOTA hybrid-SNN-NIDS
head-to-head (TODO items 4/6). Each row was **verified against the source paper** so our
evaluation can be matched to published numbers. Loaders: `src/data/{ciciot2023,
ciciomt2024, edgeiiot}.py`, each with a `label_mode` switch.

## 1. Label granularities (the `label_mode` switch)

Every dataset is benchmarked at three granularities in its paper. Our loaders cache the
**finest** labels and remap on load (`src/data/_label_modes.py`), `label_mode ∈
{binary, category, full}`. Benign is **class index 0** in every mode (so DR/FAR work).

| Dataset | `binary` | `category` (default) | `full` | Source paper |
|---|--:|--:|--:|---|
| CICIoT2023 | 2 | **8** (Benign + DDoS/DoS/Mirai/Recon/Spoofing/Web/BruteForce) | 34 | Neto et al., *Sensors* 2023 |
| CICIoMT2024 | 2 | **6** (Benign + DDoS/DoS/Recon/MQTT/Spoofing) | 19 | Dadkhah et al., *Internet of Things* 2024 |
| Edge-IIoTset | 2 | **6** (Normal + DDoS/Information-Gathering/Injection/MITM/Malware) | 15 | Ferrag et al., *IEEE Access* 2022 |

Category mappings match each paper exactly (verified): CICIoT2023 — VulnerabilityScan→Recon,
the 6 web attacks→Web; CICIoMT2024 — the 18 types→5 families by filename prefix;
Edge-IIoTset — the 14 attacks→5 threat groups per the paper's Table.

**For a head-to-head, set `label_mode` to whatever the specific competitor reports** (e.g.
TASNN). The default `category` is the most-compared multi-class scenario across all three.

## 2. Train/test split

| Dataset | Paper split | Our loader | Note |
|---|---|---|---|
| CICIoT2023 | 80/20, random shuffle of the merged 169 shards | merge → **stratified 80/20** (`test_fraction=0.2`) + 90/10 val | stratified ≈ random at this scale; val is ours (monitoring only, doesn't touch test) |
| CICIoMT2024 | 80/20 by **CIC-provided files** | uses `csv/train` (→ train+val) and `csv/test` **as given** | honours the official split exactly |
| Edge-IIoTset | user-defined (commonly 80/20) | stratified 80/20 (`test_fraction=0.2`) + 90/10 val | follows the official drop+dummy recipe, then 80/20 |

## 3. Preprocessing — **one deliberate, necessary deviation**

The papers normalise with **StandardScaler** (z-score). **We use `log1p` + MinMax→[0,1]
instead, by necessity: the latency/rate/delta spike encoders require inputs bounded in
[0,1]** — StandardScaler's unbounded z-scores cannot be spike-encoded. This is applied
**consistently across all datasets** (so within-study comparisons are fair), and the
SNN-vs-DNN comparison is run with **both** models on the *same* preprocessing, so it stays
internally fair. Exact-number parity with a DNN paper's StandardScaler result is therefore
not expected — the claim is on the accuracy–energy frontier, not headline-number matching.

Other recipe details we follow: CICIoT2023/CICIoMT2024 are all-numeric (no one-hot);
Edge-IIoTset uses the authors' published recipe (drop 15 identifier/leakage columns,
dropna, drop_duplicates, one-hot the 7 categoricals). No SMOTE/undersampling (matching the
papers); class imbalance is handled at train time via optional class weighting.

## 4. Evaluation metrics

The papers report **Accuracy, Precision, Recall, F1** (and confusion matrices). Our
`src/metrics.py` reports a **superset**: those four plus balanced-accuracy, MCC, Cohen's κ,
ROC-AUC, and the IDS-canonical DR/FAR — so we can report the exact same four for a direct
comparison while also giving the imbalance-robust metrics (MCC, balanced-acc) the papers omit.

## 5. Vanilla-SNN run configs (LeakyParallel + latency)

The Phase-1 vanilla-SNN scripts use the **same training hyperparameters across all
datasets** — `hidden_size=128`, `num_steps(T)=25`, `batch_size=128`, `epochs=10`,
`lr=1e-3` (Adam) — which the README documents as aligned with the SNN-NIDS literature
(Zhou et al. 2020, arXiv:2010.07803; TASNN 2025; Attention-CNN-LSTM, *Nature* 2025).
Encoding = latency (TTFS), neuron = `LeakyParallel` (the Phase-1 winning operating point).

What differs per dataset is the **label granularity (`label_mode`) and subsample**, chosen
to match each dataset's recognized benchmark:

| Dataset (config) | `label_mode` | #classes | subsample | Rationale (literature) |
|---|---|--:|--:|---|
| `unswnb15` | native `attack_cat` | 10 | 1.0 (full) | official partitioned train/test; the standard UNSW-NB15 multiclass |
| `ciciomt2024` | `category` | 6 | 0.25 | the paper's 5 attack families + Benign; CIC train/test files honoured |
| `ciciot2023` | `category` | 8 | 0.05 | Neto et al. *Sensors* 2023 category-level scenario |
| **`edgeiiot`** | **`full`** | **15** | 1.0 (full) | **Edge-IIoTset's recognized benchmark is the 15-class `Attack_type`** (Ferrag et al., *IEEE Access* 2022, DOI 10.1109/ACCESS.2022.3165809; the authors' official Colab classifies on `Attack_type`). 80/20 split. |
| **`toniot`** | **`full`** | **10** | 1.0 (full) | **TON-IoT (Network) recognized benchmark is the 10-class `type`** (Moustafa, *Sustainable Cities and Society* 2021; UNSW ToN_IoT). No coarser attack-family grouping exists, so `category` == `full` == the 10-class `type`. Stratified 80/20. |

Edge-IIoTset deliberately uses `full` (15-class) rather than `category` because the 15-class
`Attack_type` — not the 5-threat grouping — is what the source paper and follow-up SNN/DL-NIDS
work report. TON-IoT (Network) likewise uses its native 10-class `type` (the recognized
benchmark; no coarser grouping exists in the literature). The `label_mode` switch
(`binary` | `category` | `full`) lets us also run the binary (or, for the CIC datasets,
the coarser category) variants if a specific competitor uses those.

TON-IoT preprocessing note: we drop the network identifiers / free-text content fields
(`src_ip`, `dst_ip`, `dns_query`, `http_uri`, `http_user_agent`, `ssl_subject`, `ssl_issuer`)
because they memorise the specific testbed (e.g. the OWASP-Shepherd certificate appears only
in web-attack flows) and would leak; the remaining low-cardinality Zeek/Bro categoricals are
one-hot encoded.

## 6. Open item for the head-to-head
Before locking a comparison table, confirm the **exact `label_mode` each competitor uses**
(TASNN et al.) per dataset and match it. The switch makes this a one-line config change
(`label_mode: binary|category|full`), forwarded to every model via `src/training*.py`.

## 7. Literature sources used for the config match

Each dataset's label taxonomy, split, preprocessing and metric set above was set from the
**source paper** (and, where noted, the dataset's official release/recipe). What we took
from each:

| Dataset | Source paper | What we matched to it |
|---|---|---|
| **CICIoT2023** | Neto, Dadkhah, Ferreira, Zohourian, Lu, Ghorbani, *"CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment,"* **Sensors** 23(13):5941, 2023 — <https://www.mdpi.com/1424-8220/23/13/5941> | the binary / **8-category** / 34-fine scenarios; the 34→8 grouping (incl. VulnerabilityScan→Recon, 6 web attacks→Web); 80/20 split; metric set |
| **CICIoMT2024** | Dadkhah, Neto, et al., *"CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoMT,"* **Internet of Things**, 2024 — <https://www.sciencedirect.com/science/article/pii/S2542660524002920> | the binary / **6-category** / 19-fine scenarios; the CIC-provided `train`/`test` file split (honoured as-is); metric set |
| **Edge-IIoTset** | Ferrag, Friha, Hamouda, Maglaras, Janicke, *"Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning,"* **IEEE Access** 10:40281–40306, 2022, DOI [10.1109/ACCESS.2022.3165809](https://ieeexplore.ieee.org/document/9751703) | the **15-class `Attack_type`** benchmark; the authors' drop-15-columns + dropna/dedup + dummy-encode recipe (the official Colab classifies on `Attack_type`) |
| **UNSW-NB15** | Moustafa & Slay, *"UNSW-NB15: a comprehensive data set for network intrusion detection systems,"* **MilCIS** 2015; and Moustafa & Slay, *"The evaluation of Network Anomaly Detection Systems … UNSW-NB15 … vs KDD99,"* **Information Security Journal** 25(1–3), 2016 | the official **175,341/82,332** train/test partition; the **10-class `attack_cat`** target (Normal + 9 attacks) |
| **TON-IoT** (Network) | Moustafa, *"A new distributed architecture for evaluating AI-based security systems … Network TON_IoT datasets,"* **Sustainable Cities and Society** 72:102994, 2021; and Alsaedi, Moustafa, Tari, Mahmood, Anwar, *"TON_IoT Telemetry Dataset …,"* **IEEE Access** 8, 2020 | the **10-class `type`** target (normal + 9 attacks) and binary `label`; Zeek/Bro flow feature schema |

**SNN training hyperparameters** (hidden 128, T=25, batch 128, 10 epochs, Adam lr 1e-3) are
aligned with the SNN-NIDS literature: Zhou et al., *"Spiking neural networks for intrusion
detection"* (arXiv:2010.07803, 2020); TASNN (2025); and the Attention-CNN-LSTM IoT-IDS work
(*Nature Scientific Reports*, 2025).

## 8. UNSW-NB15 accuracy sanity-check vs the literature

Our UNSW-NB15 10-class results — **Vanilla SNN acc ≈ 0.76 / F1-macro ≈ 0.44**, **Tree-to-Spike
acc ≈ 0.745** — sit squarely in the band reported for the **official partition with no
resampling**, and the low macro-F1 reflects the near-zero recall on the tiny classes (Worms,
Shellcode, Analysis), exactly as the literature describes.

The widely-quoted **97–99%** UNSW-NB15 multiclass numbers are **not apples-to-apples**: they
come from (a) **merging** the two files and taking a fresh **IID random split**, and/or (b)
**SMOTE/ADASYN oversampling** of minority classes. On the *official* split this is impossible
to reach without resampling because of a documented **train/test distribution shift** —
Normal traffic is **31.94%** of train but **44.94%** of test ([Kasongo & Sun / "Improving the
Performance of ML-based NIDS on the UNSW-NB15 Dataset", PMC8221886](https://pmc.ncbi.nlm.nih.gov/articles/PMC8221886/),
which itself only reaches 98.43% *after* SMOTE). This is the same effect that caps NSL-KDD at
~76% on the official KDDTest+ (see TODO). We deliberately keep the official split and skip
resampling, so ~0.76 is the honest, comparable number — not an underperformance.
