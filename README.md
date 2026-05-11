# Baserra - Rental Price Prediction Pipeline

Machine learning pipeline for predicting residential rental prices in Indian cities (Bangalore & Mumbai). Built with LightGBM, XGBoost, CatBoost, Neural Networks, and advanced feature engineering.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Data Sources](#data-sources)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Notebooks](#notebooks)
- [Getting Started](#getting-started)
- [Pipeline Architecture](#pipeline-architecture)
- [Contributing](#contributing)

---

## Overview

Baserra predicts monthly residential rent using property attributes (size, BHK, locality, amenities, furnishing, floor info) and geospatial features (distance to IT parks, metro stations, airport). The project evolved from a basic LightGBM baseline (~21% MAPE) to production-grade pipelines with advanced feature engineering targeting <15% MAPE.

**Cities covered:** Bangalore (primary), Mumbai (secondary)

**Target metric:** MAPE (Mean Absolute Percentage Error)

---

## Key Results

| Pipeline | Best Model | Test MAPE | Test R2 | Dataset Size |
|----------|-----------|-----------|---------|-------------|
| LGBM Baseline | LightGBM | 21.32% | 0.8979 | 208K rows |
| Final Bangalore | Multi-objective Ensemble | ~2.05% | -- | 60K+ rows |
| ILY Pipeline | Stacking Ensemble (7 models) | Target: 12-15% | -- | 208K rows |
| Mumbai Pipeline | LightGBM + Bias Correction | varies | -- | JSON nested |

**Top features by importance:** propertySize > bathroom > locality > totalFloor > furnishing > floorNo > balcony

---

## Project Structure

```
Baserra Work/
|
|-- README.md                              # This file
|-- Final_Bangalore_Pipeline.ipynb         # Production pipeline with FastAPI
|-- Bangalore_Housing_Pipeline.ipynb       # T4 GPU optimized pipeline
|-- Bangalore_Housing_Pipeline_H100.ipynb  # H100 GPU optimized pipeline
|-- Mumbai_Pipeline_colab.ipynb            # Mumbai rent prediction
|-- LightGBM_Analysis.ipynb                # LightGBM train/test analysis
|-- LGBM.ipynb                             # Executed LightGBM with saved results
|-- ABC.ipynb                              # BERT embeddings + K-Means clustering
|-- ILY.ipynb                              # Advanced 60+ feature pipeline
|-- mumbai_data.json                       # Mumbai rental listings data
|-- Mumbai_Pipeline_colab.json             # Mumbai pipeline notebook export
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **ML Models** | LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, Ridge/Lasso/ElasticNet, Stacking Ensemble |
| **Deep Learning** | TensorFlow/Keras (ANN: 4-5 hidden layers, BatchNorm, Dropout) |
| **NLP** | Hugging Face Transformers (BERT embeddings, 768-dim) |
| **Hyperparameter Tuning** | Optuna (TPE sampler, 50-100 trials) |
| **Clustering** | K-Means, DBSCAN (geographic grouping) |
| **Feature Engineering** | scikit-learn, target encoding, Yeo-Johnson transform |
| **Deployment** | FastAPI REST API |
| **GPU Acceleration** | CUDA, Mixed Precision (FP16), XLA JIT |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Data Processing** | Pandas, NumPy |

---

## Data Sources

### Bangalore (~265,000+ rows combined)

- **CSV** - `bangalore_properties_complete.csv` (~8,740 rows) - scraped property data
- **Excel** - `banglore_data_no_broker.xlsx` (~8,740 rows) - NoBroker platform data
- **JSON** - Multiple JSON files (~248,253 rows) - NoBroker Bangalore rentals within 15km radius, 185+ columns

### Mumbai (JSON, nested structure)

- `mumbai_data.json` - Listings organized by locality with fields: title, price, propertySize, maintenance, deposit, location, address, googleMapsPin, parking, bhk, bath, facing, floor, totalFloor

### Common Features

| Feature | Description |
|---------|-------------|
| `rent` / `price` | Monthly rental price (target) |
| `propertysize` / `sqft` | Property size in sq ft |
| `bedroom` / `bhk` | Number of bedrooms |
| `bathroom` | Number of bathrooms |
| `locality` | Neighborhood name |
| `furnishing` | Furnished / Semi / Unfurnished |
| `floorno` / `totalfloor` | Floor information |
| `latitude` / `longitude` | Geolocation |
| `deposit` | Security deposit |
| `amenities_*` | Binary flags (AC, gym, pool, etc.) |

---

## Feature Engineering

The pipelines progressively add more sophisticated features:

**Basic (LGBM, LightGBM_Analysis):**
- BHK extraction, sqft parsing, price normalization
- Target encoding for locality
- Log transforms for skewed features

**Advanced (Final_Bangalore_Pipeline):**
- Haversine distances to Bangalore landmarks (Manyata Tech Park, Whitefield, Electronic City, Koramangala, MG Road Metro, Airport)
- Location premium flags (is_near_metro, is_near_it_hub, is_prime_location)
- DBSCAN geographic clustering
- Multi-objective ensemble (MAPE, MAE, Huber, RMSE)

**Most Advanced (ILY.ipynb):**
- 60+ engineered features
- Hash-based + fuzzy deduplication
- Bedroom-specific locality encoding (1BHK, 2BHK, 3BHK specific)
- Locality tier classification (Tier 1-3)
- Size-based locality statistics (mean, median, std per locality)
- Size z-score per locality
- Interaction features

---

## Models

### Gradient Boosting (Primary)
- **LightGBM** - Best overall performer, GPU-accelerated, Optuna-tuned
- **XGBoost** - With early stopping
- **CatBoost** - Native categorical support

### Ensemble
- **Stacking Ensemble** - XGBoost + LightGBM + Random Forest + CatBoost with Ridge meta-learner
- **Multi-objective LightGBM** - Ensemble of different loss functions (MAPE, MAE, Huber, RMSE)

### Neural Networks
- **ANN** - TensorFlow/Keras Sequential: 4-5 layers (512 -> 256 -> 128 -> 64 -> 32), BatchNorm, Dropout, ReduceLROnPlateau
- **H100 Optimized** - Mixed Precision FP16, XLA JIT, batch size 1024

### Linear Models
- Ridge, Lasso, ElasticNet (baseline comparisons)

### Text Embeddings
- BERT (Hugging Face) for 768-dimensional property description embeddings

---

## Notebooks

<details>
<summary><b>1. Final_Bangalore_Pipeline.ipynb</b> - Production Pipeline</summary>

- 25+ engineered features including geospatial analysis
- FastAPI REST API deployment
- Multi-objective LightGBM ensemble
- Best reported MAPE: ~2.05%
</details>

<details>
<summary><b>2. ILY.ipynb</b> - Advanced Feature Pipeline</summary>

- 60+ features, most sophisticated engineering
- 7 models compared (Ridge, RF, ExtraTrees, XGBoost, LightGBM, CatBoost, Stacking)
- Bedroom-specific locality encoding
- Saves model artifacts (joblib)
</details>

<details>
<summary><b>3. Bangalore_Housing_Pipeline.ipynb</b> - T4 GPU Pipeline</summary>

- End-to-end pipeline for Google Colab T4
- LightGBM + ANN with Optuna (50 trials)
- Target: MAPE < 20%
</details>

<details>
<summary><b>4. Bangalore_Housing_Pipeline_H100.ipynb</b> - H100 GPU Pipeline</summary>

- Same as T4 pipeline but optimized for NVIDIA H100 (80GB HBM3)
- 100 Optuna trials, batch size 1024
- Mixed Precision + XLA JIT
</details>

<details>
<summary><b>5. Mumbai_Pipeline_colab.ipynb</b> - Mumbai Pipeline</summary>

- Nested JSON data parsing
- LightGBM + ANN
- Bias correction for deployment
</details>

<details>
<summary><b>6. ABC.ipynb</b> - BERT + Clustering Pipeline</summary>

- BERT text embeddings (768-dim)
- K-Means clustering (K=150) on coordinates
- Geographic area clustering (8 areas)
- Locality-specific MAPE analysis
</details>

<details>
<summary><b>7. LGBM.ipynb</b> - Executed Baseline</summary>

- Saved LightGBM results: Test MAPE 21.32%, R2 0.8979
- Overfitting analysis: MAPE gap 1.40%
- Feature importance visualization
</details>

<details>
<summary><b>8. LightGBM_Analysis.ipynb</b> - Train/Test Analysis</summary>

- Focused LightGBM train vs test accuracy analysis
- 265K -> 208K rows after cleaning
- Good generalization confirmed
</details>

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost
pip install tensorflow optuna matplotlib seaborn plotly
pip install transformers torch  # for BERT embeddings
pip install fastapi uvicorn     # for deployment
pip install joblib openpyxl     # for model serialization and Excel reading
```

### Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd "Baserra Work"
```

2. Open any notebook in Jupyter or Google Colab:
```bash
jupyter notebook Final_Bangalore_Pipeline.ipynb
```

3. For GPU acceleration, use Google Colab with T4/H100 runtime.

4. For the production API (Final_Bangalore_Pipeline):
```python
# After running the notebook, the FastAPI server starts:
# POST /predict with property features -> returns predicted rent
```

---

## Pipeline Architecture

```
Raw Data (CSV/Excel/JSON)
        |
        v
   Data Loading & Parsing
   (price extraction, BHK parsing, nested JSON flattening)
        |
        v
   Cleaning & Deduplication
   (outlier removal, hash/fuzzy dedup, missing values)
        |
        v
   Feature Engineering
   (target encoding, geospatial, clustering, interactions)
        |
        v
   Train/Test Split (leakage-free)
   (encoders fit on train only)
        |
        v
   Model Training & Tuning
   (LightGBM + Optuna, ANN, Stacking Ensemble)
        |
        v
   Evaluation
   (MAPE, R2, MAE, overfitting analysis)
        |
        v
   Deployment
   (FastAPI REST API, joblib model serialization)
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## License

This project is for internal use at Baserra.
