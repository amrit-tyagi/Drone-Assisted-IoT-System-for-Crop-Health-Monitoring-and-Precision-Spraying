# 🚁🌾 Drone-Assisted IoT System for Crop Health Monitoring and Precision Spraying

### **Integrated UAV + IoT + AI Framework for Smart Agriculture**

This repository implements a **complete prototype** of the system described in the paper:
**“Drone-Assisted IoT System for Crop Health Monitoring and Precision Spraying.”**
It combines:

* **CropAnalysis** → Aerial multispectral & RGB data processing (NDVI, DEM, patch extraction)
* **DeepFake_Detection** → CNN+GRU deep learning architecture adapted for crop-health classification
* **IoT Sensor Integration** → Soil moisture, temperature, humidity inputs
* **Precision Spraying Logic** → Treatment zone identification for autonomous UAV spraying

This project provides the **code, model pipeline, data-processing workflow, scripts, and tools** for creating an **end-to-end intelligent precision farming system**, aligned with the conceptual framework in the associated IEEE-style research paper. 

---

# 📌 Table of Contents

* [1. Overview](#1-overview)
* [2. Key Features](#2-key-features)
* [3. System Architecture](#3-system-architecture)
* [4. Methodology](#4-methodology)
* [5. Repository Structure](#5-repository-structure)
* [6. Installation (Windows + PowerShell + Python)](#6-installation-windows--powershell--python)
* [7. How the Two Models Are Combined](#7-how-the-two-models-are-combined)
* [8. Usage](#8-usage)
* [9. Expected Outcomes](#9-expected-outcomes)
* [10. Future Improvements](#10-future-improvements)
* [11. References](#11-references)

---

# 1. 🔍 Overview

Modern agriculture faces challenges such as:

* Increasing global food demand
* Climate uncertainty
* Water scarcity
* Excessive pesticide use
* Productivity loss due to pests, diseases, nutrient deficiencies

This project implements a **closed-loop system** that connects:

* **UAVs** for real-time crop imaging
* **IoT sensors** for continuous soil & climate data
* **AI/ML models** for crop health classification
* **Autonomous drone spraying** for targeted intervention

The goal is to create a **sustainable, efficient, and intelligent precision agriculture platform**, minimizing resource waste and maximizing crop yields ― fully aligned with the conceptual framework discussed in your research paper. 

---

# 2. ⭐ Key Features

### ✔ UAV-Based Crop Health Monitoring

* NDVI, SAVI, NDRE vegetation indices
* High-resolution RGB and multispectral images
* Automatic patch extraction for AI training

### ✔ IoT-Enabled Environmental Sensing

* Soil moisture
* Temperature
* Humidity
* Real-time data fusion with aerial imagery

### ✔ AI/ML Crop Health Classification

* CNN + GRU architecture adapted from **DeepFake_Detection**
* Models classify:

  * Healthy
  * Nutrient Deficient
  * Water-Stressed
  * Diseased
* Supports multi-temporal analysis

### ✔ Precision Spraying

* Automatic generation of treatment maps
* Variable-rate, site-specific spraying
* UAV-based autonomous execution

### ✔ Fully Local Python Pipeline

* No conda needed
* Works entirely with **PowerShell + Python venv**

---

# 3. 🏗 System Architecture

*(Adapted from IEEE paper — conceptual + implementation)*

```text
         ┌────────────────┐
         │    UAV Drone   │
         │ RGB + NDVI     │
         │ Thermal (opt.) │
         └──────┬─────────┘
                │ Aerial Imagery
                ▼
     ┌─────────────────────────┐
     │   CropAnalysis Module   │
     │  • NDVI computation     │
     │  • DEM + height maps    │
     │  • Patch extraction     │
     └──────────┬──────────────┘
                │ Patches
                ▼
   ┌──────────────────────────────┐
   │     CNN + GRU ML Model       │
   │ (Adapted from DeepFake repo) │
   │  • Classification            │
   │  • Stress detection          │
   └────────────┬─────────────────┘
                │ Labels
                ▼
      ┌───────────────────────┐
      │  Precision Spraying   │
      │ • Spray maps           │
      │ • UAV route planning   │
      └─────────┬──────────────┘
                │ Commands
                ▼
      ┌────────────────────────────┐
      │ IoT Sensor Integration     │
      │ Soil + climate conditions  │
      │ Real-time adjustments      │
      └────────────────────────────┘
```

---

# 4. 📡 Methodology

This repository follows the same 4-phase methodology presented in the paper :

### **1) Data Acquisition**

* UAV captures RGB + multispectral images
* IoT sensors collect soil & microclimate data

### **2) Data Transmission**

* MQTT/HTTP → central server
* Local gateway for drone data

### **3) Data Processing & Analysis**

* Preprocessing (noise removal, correction)
* NDVI computation
* Patch extraction
* CNN + GRU crop health classification
* Fusion of drone + IoT data

### **4) Autonomous Precision Spraying**

* Treatment maps generated
* UAV executes variable-rate spraying commands

---

# 5. 📁 Repository Structure

```
Drone-Assisted-IoT-System/
│
├── CropAnalysis/                      # external repo (raw)
├── DeepFake_Detection/                # external repo (raw)
│
├── CropHealthModel/                   # main working model
│   ├── data/
│   │   ├── patches/                   # extracted training images
│   │   └── patch_metadata.csv
│   ├── models/
│   │   └── cnn_gru_crop.py            # combined architecture
│   ├── scripts/
│   │   ├── train_model.py             # training
│   │   └── predict_flight.py          # prediction pipeline
│   ├── notebooks/
│   │   └── Crop_Feature_Extraction.ipynb
│   ├── output/
│   │   └── prediction.csv
│   └── models/saved/
│       ├── crop_health.h5             # trained model
│       └── label_map.json
│
└── README.md
```

---

# 6. 🛠 Installation (Windows + PowerShell + Python)

### 1️⃣ Create virtual environment

```powershell
python -m venv drone_env
drone_env\Scripts\activate
```

### 2️⃣ Install all dependencies

```powershell
pip install -r CropAnalysis/requirements.txt
pip install -r DeepFake_Detection/requirements.txt

pip install tensorflow opencv-python rasterio earthpy shapely geopandas
pip install scikit-learn pandas matplotlib
```

---

# 7. 🔗 How the Two Models Are Combined

This repository **logically fuses** the two external repos:

### ✔ From **CropAnalysis**:

* NDVI extraction
* DEM + height maps
* Patch extraction
* GeoTIFF processing

### ✔ From **DeepFake_Detection**:

* CNN feature extractor
* GRU temporal modeling
* Training pipeline structure

### ✔ New combined architecture:

`cnn_gru_crop.py` merges them into:

```
Patch → InceptionV3 → TimeDistributed → GRU → Crop-Health-Class
```

---

# 8. ▶️ Usage

### 🟩 **Run Feature Extraction (Jupyter)**

```powershell
jupyter notebook
```

Run:
`notebooks/Crop_Feature_Extraction.ipynb`

### 🟩 **Train Model**

```powershell
python .\scripts\train_model.py
```

### 🟩 **Predict Crop Health from New Drone Image**

```powershell
python .\scripts\predict_flight.py
```

Outputs:

```
output/prediction.csv
```

Includes tile-wise labels like:

* healthy
* diseased
* stress
* spray-needed

---

# 9. 📈 Expected Outcomes

Based on the IEEE research paper analysis: 

### ✔ Up to **90% reduction** in pesticide use

### ✔ Early detection of pests & nutrient deficiencies

### ✔ 20–30% water savings

### ✔ Improved yields by 15–30%

### ✔ High classification accuracy (CNN-based)

### ✔ Autonomous, site-specific spraying

---

# 10. 🚀 Future Improvements

* Multi-UAV swarming
* Edge computing on drone
* Reinforcement learning for autonomous spraying
* Blockchain-based data security
* Hyperspectral data integration
* Digital twin simulation system

