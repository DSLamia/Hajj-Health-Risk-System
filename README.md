# 🕋 Mecca Integrated Weather & Health Risk Prediction System

An advanced, AI-driven predictive health-risk framework tailored for the microclimate of Mecca and holy sites. This project bridges deep learning inference with real-time field data to predict emergency and heat-related medical loads, empowering field coordinators and pilgrims through continuous monitoring.

---

## 🚀 System Architecture & Flow

The system operates via a tightly decoupled client-server architecture designed for zero-latency inference and absolute operational integrity.

```text
[Pilgrim / Officer Dashboard]
       │
       │ (Real-time HTTP POST via Fetch API)
       ▼
[Flask Backend API (Render Cloud)] ──────► [Supabase DB] (Profiles & Historical Logs)
       │
       ├─► [Live Weather Sync (Open-Meteo API)]
       │
       ▼
[Feature Alignment & Scaling Pipeline (scaler.pkl)]
       │
       │ (Strict 11-Feature Tensor Vector)
       ▼
[Operational Deep Learning Inference (ONNX Runtime Engine)]
       │
       ▼
[Dynamic Risk Level Output & Tailored Preventive Protocols]
🛠️ Key Architectural Updates (Production Deployment)
To transition this system from a local development environment to a live, production-grade cloud solution, the following enhancements were implemented:

Strict 11-Feature Matrix Feeding: Adjusted the preprocessing logical layout to handle true mathematical representations. Categorical values (such as chronic illness markers) are strictly mapped to float numbers (100.0 or 0.0) to match the matrix weights the ONNX model was trained on.

Robust Backend Verification: Reconfigured app.py to securely run parallel user identification checks (via Phone Number matching followed by unique UUID checking) ensuring database integrity before calculations take place.

Elimination of Mock Placeholders: Removed all legacy fallback statistical approximations. The inference system now natively relies 100% on pure ONNX tensor output; if a framework mismatch or asset loss occurs, the server raises a strict runtime exception rather than falsifying clinical criteria.

Cloud Infrastructure Configuration: Implemented dynamic environment porting and automated dependency management via Gunicorn engines for seamless deployment on cloud providers (Render).
📊 Feature Matrix Layout (ONNX & Scaler Alignment)
The model evaluates a rigid array of 11 specific input dimensions structured sequentially:
IndexFeature ParameterData TypeDescription / Operational Context1Age Group Encodedfloat32Encoded ordinal brackets (0 for 1-15, 1 for 16-60, 2 for 61+)2Crowd Density Indicatorfloat32Standard dynamic baseline footprint density value3Temperaturefloat32Real-time ambient temperature fetched dynamically per hour4Humidityfloat32Atmospheric relative humidity percentage5Wind Speedfloat32Meteorological wind tracking parameter6Hospitals Countfloat32Operational healthcare infrastructure capacity index7Health Centers Countfloat32Available clinical processing stations8Total Bed Capacityfloat32Max physical capacity managed under resource databases9Staff Countfloat32Deployed medical personnel and field emergency forces10Ambulance Fleet Sizefloat32Mobilized emergency vehicular response units11Chronic Disease Inputfloat32Boolean health status amplified to tensor metrics (100.0 / 0.0)
📂 Project Repository Structure
app.py — Core production Flask application hosting predictive endpoints and Supabase initialization wrappers.

model_handler.py — Serialized asset validation module executing ONNX runtime session parsing and feature engineering scaling.

static/ — Production front-end asset bundles (UI stylesheets, iconography, video wrappers).

templates/ — Interactive portal templates (Pilgrim dashboard layouts, tactical monitoring interfaces).

requirements.txt — Frozen environment deployment configuration declaring explicit dependencies (onnxruntime, joblib, pandas, etc.).
