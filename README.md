# 📊 Census Income Classification & Customer Segmentation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete machine learning pipeline for **binary income classification** (≤\$50K vs. >\$50K)
and **unsupervised customer segmentation** using U.S. Census Bureau data (1994–1995 CPS).

---

## 🎯 Project Overview

| Task | Method | Key Result |
|------|--------|------------|
| **Classification** | Gradient Boosting (best) | AUC-ROC = **0.955**, Accuracy = **95.8%** |
| **Segmentation** | K-Means (K=4) + PCA | 4 actionable marketing segments |
| **Baseline Models** | Logistic Regression, Random Forest | AUC-ROC ≥ 0.943 |

### Problem Statement

Given 40 demographic and employment features from ~200K census records, this project:

1. **Predicts** whether an individual earns above or below \$50,000/year
2. **Segments** the population into distinct marketing groups for targeted campaigns

---

## 📁 Repository Structure
