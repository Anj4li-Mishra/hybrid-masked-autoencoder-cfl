# hybrid-masked-autoencoder-cfl

## Clustered Federated Learning for Malicious Network Detection

This repository implements a **Clustered Federated Learning** framework for detecting malicious activity in network traffic using **Hybrid Masked-Denoising Autoencoders (MDAE)**. The system is designed to support protocol-specific clustering, robust handling of missing data, and federated aggregation for privacy-preserving, distributed anomaly detection.

---

## Features

- **Federated Learning:** Train autoencoders collaboratively across multiple clients without sharing raw data.
- **Clustered Protocols:** Separate models for different network protocols (e.g., GOOSE, TCP).
- **Masked-Denoising Autoencoders:** Robust to missing features and noisy inputs.
- **Automated Preprocessing:** Convert CSV files to ready-to-use tensors and masks.
- **Evaluation Utilities:** Threshold computation, anomaly scoring, and confusion matrix visualization.

---

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Federated Training](#federated-training)
  - [Evaluation](#evaluation)
  - [Results](#results)

---

## Getting Started

### Installation

Clone the repository:

`git clone https://github.com/Anj4li-Mishra/hybrid-masked-autoencoder-cfl`
`cd hybrid-masked-autoencoder-cfl`

---

### Data Preparation

- Place your CSV files in the `preprocessed/` directory.

- Run the preprocessing script to generate `data.npy` and `mask.npy`:

`python process.py`


---

### Federated Training

- **Start the server** (specify protocol):

`python server.py <protocol>` 

Example:
`python server.py goose`

- **Start each client** (in separate terminals):
`python client.py preprocessed/clientX_<protocol>`

Replace `clientX_<protocol>` with the actual client directory name.

---

### Evaluation

Run the evaluation script to generate reports and confusion matrices:

`python evaluate.py`

## Results

- Classification reports and normalized confusion matrices are generated for each protocol.
- Saved global models are stored in the `cluster_models/` directory.

---



