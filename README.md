# Causal-Aware Explainable AI for Low-Rate DDoS Detection in Knowledge-Defined IoT Networks

This repository contains the source code for the paper submitted to the **IEEE Internet of Things Journal (Special Issue: Causal AI)**.

## 📖 Overview
We propose a **Causal-XAI Framework** that integrates a Structural Causal Model (SCM) into a deep learning pipeline for detecting Low-Rate DDoS attacks in IoT networks. Unlike traditional black-box models, our approach uses a **Knowledge Plane** to align feature importance with causal structures, ensuring trustworthy and explainable decisions.

## ️ Architecture
- **Causal Knowledge Plane (CKP):** Enriches input features using a learnable adjacency matrix based on domain knowledge (DAG: Burst -> TCP -> Queue -> Attack).
- **Hybrid Backbone:** Supports both CNN-GRU (for time-series) and MLP (for flow-level benchmarks like CIC-DDoS2019).
- **Explainability:** Uses Gradient-based Saliency Maps and DoWhy for causal effect estimation.

## 📊 Key Results
- **Synthetic Data (Proof of Concept):** F1-Score: **92.97%**, Causal Effect: **0.1410** (Validated via DoWhy).
- **Real-World Data (CIC-DDoS2019 MSSQL):** Accuracy: **93.21%** (Comparable to non-causal baseline, but with causal explanations).

## 🚀 Installation & Usage

### Requirements
```bash
pip install -r requirements.txt