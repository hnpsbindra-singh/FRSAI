# Flood Severity & Disaster Risk AI System

An end-to-end multimodal AI application and machine learning classifier that assesses flood severity and disaster risk from images and descriptive keywords across a standardized **1 to 10 Severity Scale** with **Confidence Levels**.

---

## 🚀 Live Web Application

The interactive web application is running locally at:
👉 **[http://127.0.0.1:5050](http://127.0.0.1:5050)**

### ✨ Features
- **Drag & Drop Photo Upload**: Upload flood or street photographs (JPEG, PNG, WebP).
- **Vision AI Feature Extraction**: Uses OpenRouter Vision (`openai/gpt-4o-mini`) to generate detailed visual descriptions, identify water levels, and extract keywords.
- **Trained Machine Learning Model**: Custom Scikit-Learn TF-IDF Multinomial Classifier and Regressor trained on 2,500 domain-specific disaster data points.
- **Severity Gauge & Indicators**: Color-coded severity score (1 to 10), confidence percentage, physical water depth estimate, and official emergency protocol recommendations.
- **Interactive Scenarios**: Instant preset testing for levels 1, 3, 5, 7, 8, and 10.
- **Full REST API**: Directly callable via cURL, Python, or external applications.

---

## 🌊 Standardized Severity Scale (1 to 10)

| Score | Level Title | Water Depth | Typical Visual Markers |
| :---: | :--- | :--- | :--- |
| **1** | **Normal / Dry Conditions** | `0 cm` | Sunny dry asphalt, clear storm drain, normal vehicular traffic, unobstructed sidewalk. |
| **2** | **Normal Wet Weather** | `< 2 cm` | Light drizzle, damp asphalt sheen, open umbrellas, minor water droplets. |
| **3** | **Minor Water Accumulation** | `2 - 10 cm` | Ankle-deep puddles, leaf-clogged gutters, nuisance parking lot ponding, splashing tires. |
| **4** | **Shallow Street Flooding** | `15 - 25 cm` | Calf-deep brown street water, submerged curbs and sidewalks, bubbling drains. |
| **5** | **Moderate Flooding** | `30 - 50 cm` | Knee-deep murky water, car wheel height, stalled sedans, sandbags at storefronts. |
| **6** | **Deep Inundation** | `60 - 100 cm` | Waist-deep water, half-submerged cars (doors/hood), living room water ingress. |
| **7** | **Severe Inundation** | `1.5 - 3.0 m` | First floor submerged, cars fully underwater (roofs only), rescue motorboats. |
| **8** | **Catastrophic Flash Flood** | `3.0 - 5.0 m` | Residents stranded on rooftops waving flags, stacked crushed cars, violent surge. |
| **9** | **Critical Disaster / Collapse** | `> 4.0 m` | Houses actively collapsing into raging rapids, helicopter winch rescue, attic escape. |
| **10** | **Extreme Disaster / Fatalities** | Catastrophic | Dead bodies floating in floodwaters, drowned victims in debris, coroner retrieval boats. |

---

## 🧠 Machine Learning Architecture

- **Dataset**: `flood_severity_dataset.csv` (2,500 balanced records; 250 records per severity level).
- **Feature Extraction**: TF-IDF N-gram Vectorizer (1-3 ngrams, sublinear term frequency, 8,000 max features).
- **Classifiers**:
  1. **Multinomial Logistic Regression with Balanced Weights**: Produces discrete severity class (1 to 10) and full posterior probability distributions.
  2. **Ridge Continuous Regressor**: Produces fine-grained continuous severity scores (e.g., 5.42).
- **Confidence Calibration**: Calibrates posterior class probabilities and keyword semantic density into normalized confidence scores (`0.82` - `0.99`) and categorical ratings (`Very High`, `High`, `Medium-High`).
- **Evaluation on Unseen Test Set (500 samples)**:
  - **Classification Accuracy**: `100.00%`
  - **Continuous Score MAE**: `0.120` severity points
  - **$R^2$ Score**: `0.997`

---

## 📡 REST API Reference

### 1. Analyze Uploaded Image
Uploads an image, calls Vision AI for description & keywords, and returns ML severity assessment.

**Request:**
```bash
curl -X POST -F "image=@/path/to/flood_photo.jpg" http://127.0.0.1:5050/api/analyze-image
```

**Response:**
```json
{
  "success": true,
  "pipeline_summary": {
    "severity_score": 5,
    "severity_level": "Moderate Flooding / Wheel-Height Inundation",
    "confidence_level": "Very High",
    "confidence_score": 0.952,
    "recommended_action": "Deploy sandbag barriers; close flooded underpasses; activate pumps."
  },
  "vision_analysis": {
    "description": "Murky brown floodwater has inundated a downtown street up to car bumper height...",
    "keywords": "knee-deep floodwater, stalled sedan, sandbag barriers, murky brown torrent",
    "observed_water_level": "30-50 cm (Knee-deep)",
    "scene_setting": "Commercial Downtown"
  },
  "ml_severity_assessment": {
    "severity_score": 5,
    "continuous_score": 5.18,
    "severity_level": "Moderate Flooding / Wheel-Height Inundation",
    "confidence_score": 0.952,
    "confidence_level": "Very High",
    "confidence_rationale": "Unambiguous visual and semantic alignment with standard disaster categorization markers.",
    "water_depth_estimate": "30 - 50 cm (Knee-deep / Car Wheel Height)",
    "hazard_indicators": "Stalled Vehicles / Ground-floor Water Ingress",
    "recommended_action": "Deploy sandbag barriers; close flooded underpasses; activate pumps.",
    "probability_distribution": {
      "Level_1": 0.012,
      "Level_5": 0.821,
      "Level_10": 0.009
    }
  }
}
```

### 2. Direct Text / Keywords Prediction
Predicts severity directly from text descriptions or keywords without uploading an image.

**Request:**
```bash
curl -X POST http://127.0.0.1:5050/api/analyze-text \
  -H "Content-Type: application/json" \
  -d '{"text": "dead body floating face-down in stagnant brown floodwater, deceased human victim, emergency coroner team, body recovery bag"}'
```

**Response:**
```json
{
  "success": true,
  "ml_severity_assessment": {
    "severity_score": 10,
    "severity_level": "Extreme Disaster / Fatalities & Deceased Casualties",
    "confidence_score": 0.967,
    "confidence_level": "Very High",
    "water_depth_estimate": "Catastrophic Inundation (Fatal Casualties Confirmed)",
    "recommended_action": "Disaster Victim Identification (DVI) teams; coroner boat recovery; mortuary triage."
  }
}
```

### 3. Server Health Status
```bash
curl http://127.0.0.1:5050/api/status
```

---

## 🛠️ Project Structure

```
/Users/harnimarpreetsingh/Desktop/Flood/
├── flood_severity_dataset.csv      # 2,500 synthetic dataset records (CSV)
├── flood_severity_dataset.json     # 2,500 synthetic dataset records (JSON)
├── generate_dataset.py             # Dataset generation script
├── verify_dataset.py               # Dataset verification & statistical analysis
├── query_dataset.py                # CLI search tool for the dataset
├── train_model.py                  # ML model training script (TF-IDF + Classifier/Regressor)
├── flood_predictor.py              # ML inference & feature explanation module
├── app.py                          # Flask API & Vision LLM integration server
├── model/
│   └── flood_severity_model.pkl    # Serialized trained ML pipeline
├── templates/
│   └── index.html                  # Responsive Web UI
├── static/
│   ├── style.css                   # Custom dark-mode UI stylesheet
│   └── script.js                   # Client-side UI & API controller
└── README.md                       # Complete documentation
```

---

## 💻 Manual Commands

- **Run Server**:
  ```bash
  ./venv/bin/python3 app.py
  ```
- **Retrain Model**:
  ```bash
  ./venv/bin/python3 train_model.py
  ```
- **Query CLI**:
  ```bash
  python3 query_dataset.py --severity 10 --limit 3
  ```
