<div align="center">

# Privacy-Preserving Blockchain-Enabled Federated Learning for Secure Medical Image Analysis
### A Lightweight PyTorch Implementation with Weighted FedAvg, Blockchain Logging, and Privacy Protection

**Project developed for academic demonstration and reproducible experimentation**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-supported-orange)
![Federated Learning](https://img.shields.io/badge/Federated%20Learning-enabled-yellow)
![Blockchain](https://img.shields.io/badge/Blockchain-logging-purple)
![Privacy](https://img.shields.io/badge/Privacy-preserving-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

## Overview

Federated learning enables multiple clients to collaboratively train a machine learning model without sharing raw data. However, federated systems still face important challenges such as:
- model update tampering,
- privacy leakage from shared gradients or weights,
- poisoned or malicious client updates,
- lack of traceability and auditability.

This project implements a lightweight and reproducible solution that combines:

- **Federated Learning** using **Weighted Federated Averaging (FedAvg)**
- **Blockchain-style logging** for update verification and auditability
- **Privacy protection** using clipping and Gaussian noise
- **Attack simulation** using a malicious client
- **Evaluation and visualization** through accuracy/loss plots and saved checkpoints

The implementation is designed to be easy to understand, run, and extend.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Proposed Architecture](#proposed-architecture)
3. [How It Works](#how-it-works)
4. [Results & Metrics](#results--metrics)
5. [Project Structure](#project-structure)
6. [Core Modules](#core-modules)
7. [Setup & Installation](#setup--installation)
8. [Running the Project](#running-the-project)
9. [Implementation Notes](#implementation-notes)
10. [Limitations](#limitations)
11. [Future Work](#future-work)
12. [Author](#author)

---

## Problem Statement

Centralized AI training is often unsuitable for sensitive domains because data must be shared with a central server. In medical, security, and privacy-sensitive applications, this creates several problems:

- Raw data may contain sensitive information.
- Uploaded gradients or updates may leak private details.
- Malicious clients can send poisoned updates.
- There is no built-in trust or verifiable audit trail for model updates.

This project addresses these issues by building a federated learning system where:
- clients train locally,
- only model updates are shared,
- updates are logged in a blockchain-style ledger,
- privacy is improved using update clipping and noise,
- training progress is evaluated after every round.

---

## Proposed Architecture

<div align="center">
<img src="assets/architecture.png" width="700" alt="Architecture Diagram"/>
</div>

### System Components
| Component | Role |
|---|---|
| Client Nodes | Train local models on private data partitions |
| Global Server | Aggregates client updates using Weighted FedAvg |
| Blockchain Ledger | Records update hashes and round metadata |
| Privacy Module | Applies clipping and Gaussian noise |
| Evaluation Module | Measures loss and accuracy on test data |
| Visualization Module | Generates plots and checkpoints |

---

## How It Works

### Training Workflow
1. The global model is initialized on the server.
2. The training dataset is split among multiple clients.
3. Each client trains locally for a small number of epochs.
4. Local updates are clipped for stability.
5. Optional Gaussian noise is added for privacy.
6. Each update is logged in a blockchain-style ledger.
7. The server aggregates updates using **Weighted FedAvg**.
8. The updated global model is evaluated on the test set.
9. Accuracy/loss values are stored and plotted.

### Blockchain Logging
Each client update is recorded with:
- client ID
- training round
- timestamp
- update hash
- previous block hash
- block hash

This makes the update history traceable and tamper-evident.

### Privacy Layer
The privacy component includes:
- update clipping
- optional Gaussian noise injection

This helps reduce sensitivity of client updates.

### Attack Simulation
A malicious client can optionally be enabled to simulate:
- noisy updates
- poisoned updates

This helps demonstrate robustness and the impact of adversarial behavior.

---

## Results & Metrics

The current implementation has been tested successfully on MNIST.

### Example Evaluation Output
- **Test loss:** `0.3548`
- **Test accuracy:** `90.02%`

### Saved Artifacts
- Model checkpoints in `outputs/checkpoints/`
- Training plot in `outputs/plots/`
- Blockchain ledger in `outputs/blockchain/ledger.json`
- Metrics CSV in `outputs/metrics/training_metrics.csv`

### Generated Visualizations
- training progress plot
- round-wise loss curve
- round-wise accuracy curve

---

## Project Structure

```text
project/
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── data.py
│   ├── client.py
│   ├── server.py
│   ├── blockchain.py
│   ├── privacy.py
│   ├── train_federated.py
│   └── evaluate.py
├── outputs/
│   ├── checkpoints/
│   ├── plots/
│   ├── blockchain/
│   └── metrics/
├── data/
├── requirements.txt
└── README.md
