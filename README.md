# HiSNN
This repo is for Hybrid Spiking Neural Networks (HiSNN) for Intrusion Detections


# Project Architecture:
1. Propose 2 hybrid approaches with SNN [one with "KAN" and one with "AutoEncoder and Energy-Based Model Head"
2. SNN Ablation Study [with different rate encoding, with different neuron model]
3. 5 different datasets [with various sizes and using partial data vs full dataset], Binary Classifier Dataset vs Multi-Class Dataset
4. Use the dataset and the model published in previous surveys for benchmarking and comparisons
5. Use: SNN, KAN, Deep Neural Network, RNN as baseline comparison
6. Use Metrics for Comparison: Accuracy, Neurons Firing, Energy Consumption, F1, Precision, Recall, ROC-AUC, etc.
7. Use mathematical formulation to show model hardness and other things
8. Show trade-off [training time, computational overhead, inference time]
9. Show Attack Performance/Robustness [Membership Inference Attack, Model Stealing, White-box attacks: FGSM, PGD, C&W, BIM, MIM}


# Main datasets Repo used in this project: 

Repo Link with all the Cybersecurity Datasets: https://github.com/ctinnil/CY0P5_ML_Datasets

### 1. **KDD Cup 1999**:
 The KDD Cup 1999 dataset is one of the earliest datasets used for intrusion detection research. It contains data from the DARPA Intrusion Detection Evaluation experiment.
 - [KDD Cup 1999 Dataset](https://www.kdd.org/kdd-cup/view/kdd-cup-1999/Data)
 - [Kaggle version](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data)

### 2. **NSL-KDD**:
 The NSL-KDD dataset is a modified version of the well-known KDD Cup 1999 dataset, addressing issues such as redundancy and balance. The new dataset is reduced to the unique values and balanced representation of the different types of the described attacks.
 - [NSL-KDD Dataset](http://www.unb.ca/cic/datasets/nsl.html)
 - [Kaggle version](https://www.kaggle.com/datasets/hassan06/nslkdd)

### 3. **CTU-13**:
 The CTU-13 dataset is particularly notable for its comprehensive representation of botnet traffic, allowing researchers to analyze and develop detection methods for botnet-related activities. CTU-13 features traffic data related to a total of 7 distinct botnets, featuring a mix of both real-world botnet and user-simulated traffic in a university-like environment.
 - [CTU-13 Dataset](https://www.stratosphereips.org/datasets-ctu13)
 - [Kaggle version](https://www.kaggle.com/datasets/dhoogla/ctu13)

### 4. **ISCXIDS2012**:
 The ISCXIDS2012 dataset consists of network traffic data, including both normal network traffic and various types of simulated and real-world cyberattacks.
 - [ISCXIDS2012 Dataset](https://www.unb.ca/cic/datasets/ids.html)
 - [Shortcut to downloads](http://205.174.165.80/CICDataset/ISCX-IDS-2012/Dataset/)

### 5. **CIC-IDS2017**:
 The Canadian Institute for Cybersecurity Intrusion Detection Systems (CICIDS2017) dataset contains network traffic data specific to machine learning for intrusion detection system (IDS) research, describing various attack scenarios, such as DoS, DDoS, and port scanning.
 - [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
 - [Kaggle version](https://www.kaggle.com/datasets/cicdataset/cicids2017)

### 6. **CSE-CIC-IDS2018**:
 The colaboorative project between the Communications Security Establishment (CSE) and the Canadian Institute for Cybersecurity (CIC) resulted in a comprehensive dataset, describing various attack scenarios, such as DoS, DDoS, and port scanning, that can be used for machine learning intrusion detection system (IDS) research.
 - [CSE-CIC-IDS2018 Dataset](https://www.unb.ca/cic/datasets/ids-2018.html)
 - [AWs S3](https://aws.amazon.com/cli/)
 
 ```AWS
 aws s3 sync s3://cse-cic-ids2018/dir/ ./localdir
 ```