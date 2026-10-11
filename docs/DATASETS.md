# Datasets

Eleven public network-intrusion datasets: four IoT benchmarks that the manuscript's main text
reports, and seven general-purpose benchmarks reported in its appendix. None is redistributed
here. Download each into `dataset/<folder>/` (gitignored); the loader in `src/data/` reads the
raw files and writes a preprocessed cache next to them on first use.

## Overview

Rows are the splits the reported runs used (seed 42; train excludes the 10% validation
hold-out). Every model trains on exactly these splits, features and labels.

| Dataset | Folder | Loader | Classes | Features | Train | Test | Split | Subsample |
|---|---|---|--:|--:|--:|--:|---|--:|
| **CICIoT2023** | `cic-iot2023` | [`ciciot2023.py`](../src/data/ciciot2023.py) | 8 | 46 | 1,680,715 | 466,866 | stratified 80/20 | 0.05 |
| **CICIoMT2024** | `cic-iomt-2024` | [`ciciomt2024.py`](../src/data/ciciomt2024.py) | 6 | 45 | 1,611,186 | 403,545 | provider train/test files | 0.25 |
| **Edge-IIoTset** | `edge-iiotset` | [`edgeiiot.py`](../src/data/edgeiiot.py) | 14 | 64 | 1,374,705 | 381,863 | stratified 80/20 | 1.0 |
| **TON-IoT** (network) | `ton-iot` | [`toniot.py`](../src/data/toniot.py) | 10 | 115 | 151,950 | 42,209 | stratified 80/20 | 1.0 |
| CSE-CIC-IDS2018 | `cse-cic-ids2018` | [`cseids2018.py`](../src/data/cseids2018.py) | 5 | 78 | 5,113,376 | 2,434,942 | stratified 70/30 | 0.5 |
| CIC-IDS2017 | `cicids2017` | [`cicids2017.py`](../src/data/cicids2017.py) | 5 | 78 | 1,783,368 | 849,223 | stratified 70/30 | 1.0 |
| CTU-13 | `ctu13` | [`ctu13.py`](../src/data/ctu13.py) | 2 | 37 | 504,712 | 240,340 | stratified 70/30 | 1.0 |
| KDDCup99 | `kddcup99` | [`kddcup99.py`](../src/data/kddcup99.py) | 5 | 119 | 444,618 | 311,029 | official train / corrected test | 1.0 |
| NSL-KDD | `nslkdd` | [`nslkdd.py`](../src/data/nslkdd.py) | 5 | 122 | 113,375 | 22,544 | official KDDTrain+ / KDDTest+ | 1.0 |
| UNSW-NB15 | `unsw-nb15` | [`unswnb15.py`](../src/data/unswnb15.py) | 10 | 196 | 157,806 | 82,332 | official training / testing set | 1.0 |
| ISCX-IDS2012 | `iscxids2012` | [`iscxids2012.py`](../src/data/iscxids2012.py) | 2 | 175 | 1,305,143 | 621,498 | stratified 70/30 | 1.0 |

Subsample is the fraction of rows drawn (stratified) before splitting, used where a corpus has
millions of near-identical flood flows. The validation split (10% of the training partition,
stratified) is used for checkpoint and threshold selection only; test is touched once, at the
end.

## Labels

Benign is class 0 in every dataset, so detection rate and false-alarm rate are defined on
benign versus any attack.

| Dataset | Labels used | Classes |
|---|---|---|
| CICIoT2023 | category | Benign, BruteForce, DDoS, DoS, Mirai, Recon, Spoofing, Web |
| CICIoMT2024 | category | Benign, DDoS, DoS, MQTT, Recon, Spoofing |
| Edge-IIoTset | `Attack_type` (full) | Normal and 13 attacks: Backdoor, DDoS_HTTP, DDoS_ICMP, DDoS_TCP, DDoS_UDP, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS |
| TON-IoT | `type` | normal, backdoor, ddos, dos, injection, mitm, password, ransomware, scanning, xss |
| UNSW-NB15 | `attack_cat` | Normal and nine attack categories |
| KDDCup99, NSL-KDD | attack family | normal, dos, probe, r2l, u2r |
| CIC-IDS2017, CSE-CIC-IDS2018 | NSL-KDD families | CIC attack labels mapped to normal, dos, probe, r2l, u2r |
| CTU-13 | binary | normal, botnet |
| ISCX-IDS2012 | binary | normal, attack |

The CIC/Edge/TON loaders accept `label_mode` (`binary`, `category`, `full`; mapping in
[`src/data/_label_modes.py`](../src/data/_label_modes.py)); the table lists the modes the
reported runs use.

## Preprocessing

The same pipeline for every dataset and every model:

- categorical fields one-hot encoded; `log1p` on heavy-tailed numeric fields; MinMax scaling to
  [0, 1] fitted on the training partition (spike encoders need bounded inputs);
- no oversampling or undersampling; class imbalance enters only through the teacher's cost
  weights and, where configured, class-weighted losses;
- TON-IoT drops addresses and free-text fields (`src_ip`, `dst_ip`, `dns_query`, `http_uri`,
  `http_user_agent`, `ssl_subject`, `ssl_issuer`), which identify the testbed rather than the
  attack.

### Repairs

A leakage audit ([`results/diagnostics/leakage_audit`](../results/diagnostics/leakage_audit))
found two corpora whose published files reveal the label; both are repaired in
[`src/data/repair.py`](../src/data/repair.py) before any model sees them.

- **Edge-IIoTset.** The zero placeholder is spelled `0` or `0.0` depending on the source
  capture, and rows whose tshark fields contained commas are shifted by one or two columns. The
  loader canonicalises the values, decodes the three verified row layouts
  (`scripts/verify_edgeiiot_repair.py` checks them against the published pcaps), drops the
  payload and identifier columns `http.referer` and `dns.qry.name`, and drops the MITM class,
  which cannot be repaired (15 classes become 14).
- **CSE-CIC-IDS2018.** `Flow ID`, `Src IP`, `Src Port` and `Dst IP` exist only in
  `02-20-2018.csv`; concatenation zero-pads them for the other nine days, making `Src Port` a
  capture-day indicator. They are dropped.

The runs from before the repair are kept in [`results/archive/pre_repair`](../results/archive/pre_repair).

## Notes on specific benchmarks

- **NSL-KDD and UNSW-NB15** ship official partitions whose test sets differ in distribution
  from training (novel attacks, a different class mix). The official partitions are used as
  given, so accuracies near 76% are the honest numbers for these protocols, not the 97 to 99%
  reported after merging the files into a random split or oversampling.
- **CICIoMT2024**'s provider partition is at the level of capture files: test flows come from
  pcap files never used in training. Whether those files are separate recording sessions is not
  documented (pre-registered item B4).
- **Cross-partition duplicates** (identical feature vectors in train and test) are measured in
  item B5: 20.9% of CICIoT2023 test rows, 16.1% of TON-IoT, 5.2% of CICIoMT2024 and 1.9% of
  Edge-IIoTset; no IoT class is dominated by them.

## Sources

| Dataset | Reference | Download |
|---|---|---|
| CICIoT2023 | E. C. P. Neto et al., "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment," *Sensors* 23(13):5941, 2023 | [UNB CIC](https://www.unb.ca/cic/datasets/iotdataset-2023.html) |
| CICIoMT2024 | S. Dadkhah et al., "CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoMT," *Internet of Things*, 2024 | [UNB CIC](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) |
| Edge-IIoTset | M. A. Ferrag et al., "Edge-IIoTset: A new comprehensive realistic cyber security dataset of IoT and IIoT applications," *IEEE Access* 10, 2022 | [IEEE DataPort](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications) |
| TON-IoT | N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets," *Sustainable Cities and Society* 72, 2021 | [UNSW](https://research.unsw.edu.au/projects/toniot-datasets) |
| CSE-CIC-IDS2018 | I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, *ICISSP* 2018 (CSE-CIC-IDS2018 release) | [UNB CIC](https://www.unb.ca/cic/datasets/ids-2018.html) |
| CIC-IDS2017 | I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization," *ICISSP* 2018 | [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) |
| CTU-13 | S. García et al., "An empirical comparison of botnet detection methods," *Computers & Security* 45, 2014 | [Stratosphere](https://www.stratosphereips.org/datasets-ctu13) |
| KDDCup99 | KDD Cup 1999 data (DARPA 1998, MIT Lincoln Laboratory) | [KDD archive](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) |
| NSL-KDD | M. Tavallaee et al., "A detailed analysis of the KDD CUP 99 data set," *IEEE CISDA* 2009 | [UNB CIC](https://www.unb.ca/cic/datasets/nsl.html) |
| UNSW-NB15 | N. Moustafa, J. Slay, "UNSW-NB15: A comprehensive data set for network intrusion detection systems," *MilCIS* 2015 | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| ISCX-IDS2012 | A. Shiravi et al., "Toward developing a systematic approach to generate benchmark datasets for intrusion detection," *Computers & Security* 31(3), 2012 | [UNB CIC](https://www.unb.ca/cic/datasets/ids.html) |

Each dataset is subject to its provider's terms of use.
