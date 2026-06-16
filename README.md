# ⬡ RoadScan AI: AI-Powered Road Infrastructure & GPS Health Scanner

[![Streamlit App](https://static.streamlit.io/badge_badge.svg)](https://road-scan-ai-pr2s64pwmzfygdtcwdzumq.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Model-MobileNetV2-orange.svg)](https://keras.io/)

### 🚧 Final Year Project · AI Road Infrastructure Monitoring System
An end-to-end computer vision and geospatial web application engineered to automate road surface safety auditing and predictive lifecycle tracking. Using transfer learning on deep convolutional networks, **RoadScan AI** scans structural images for crack patterns, maps highway health via interactive GPS tracking, and outputs intelligent safety advisories.

🌐 **Live Production Deployment:** [Launch RoadScan AI App](https://road-scan-ai-pr2s64pwmzfygdtcwdzumq.streamlit.app/)

---

## 🔍 Core Features & Modules

* **Crack Detection Pipeline:** High-performance binary image classification checking for structural surface anomalies.
* **Damage Timeline Forecasting:** Calculates and projects the estimated deterioration curves and overall lifespan of the audited pavement asset.
* **Intelligent Speed Advisory:** Generates algorithmic, context-aware driving speed recommendations based on localized structural severity indexes.
* **NH GPS Segment Scanner:** Integrates geospatial tools to chart segment-by-segment National Highway health telemetry onto interactive maps.

---

## 🧠 Model & Technical Architecture

### Deep Learning Pipeline
The underlying vision network leverages a **MobileNetV2** backbone trained via **Transfer Learning** to deliver fast, edge-optimized inference pipelines.

| Attribute | Technical Specification |
| :--- | :--- |
| **Model Architecture** | MobileNetV2 |
| **Input Shape** | 224px × 224px × 3 (RGB) |
| **Target Classes** | 2 (`crack: 0` / `no_crack: 1`) |
| **Peak Validation Accuracy** | **100.00%** |
| **Peak Validation AUC** | **100.00%** |

### Geospatial & Backend Stack
* **UI/Application Layer:** Streamlit (Reactive State Management)
* **Geospatial Mapping:** OpenStreetMap API with Plotly Maps engine
* **Static Database Layer:** 50+ Pre-loaded National Highway (NH) coordinate and data clusters
* **Cloud Infrastructure:** Architected for rapid enterprise deployment (Azure Ready)

---
