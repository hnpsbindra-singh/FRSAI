#!/usr/bin/env python3
"""
Government-Grade Flood Severity & Disaster Risk ML Training Pipeline
Retrains the ML model with domain-specific stop words, balanced feature weighting,
and calibrated classification for high-precision severity prediction (1 to 10).
"""

import os
import csv
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score

CSV_PATH = "/Users/harnimarpreetsingh/Desktop/Flood/flood_severity_dataset.csv"
MODEL_DIR = "/Users/harnimarpreetsingh/Desktop/Flood/model"
MODEL_PATH = os.path.join(MODEL_DIR, "flood_severity_model.pkl")

# Domain stop words to eliminate boilerplate template tokens and force focus on physical indicators
DOMAIN_STOP_WORDS = list(ENGLISH_STOP_WORDS) + [
    "image", "photo", "photograph", "showing", "visible", "capture", "capturing", 
    "scene", "flood", "flooding", "water", "waterwater", "view", "observed", 
    "accompanied", "develops", "amidst", "documentary", "style", "overhead", 
    "surveillance", "ground", "level", "depicting", "illustrating", "alongside"
]

SEVERITY_METADATA = {
    1: {
        "level_name": "Normal / Dry Conditions",
        "category": "FEMA Level IV: Routine Monitoring",
        "depth_estimate": "0 cm (Completely Dry)",
        "hazard_summary": "Zero flood hazard; normal traffic and pedestrian activities.",
        "action": "Routine municipal monitoring; standard operational status.",
        "badge_color": "#10b981", # Green
        "threat_level": "None"
    },
    2: {
        "level_name": "Normal Wet Weather / Light Surface Moisture",
        "category": "Advisory: Wet Road Sheen",
        "depth_estimate": "< 2 cm (Thin Water Sheen)",
        "hazard_summary": "Wet pavement; light tire spray; minor traction reduction.",
        "action": "Standard wet weather driving caution; routine stormwater maintenance.",
        "badge_color": "#22c55e", # Light green
        "threat_level": "Very Low"
    },
    3: {
        "level_name": "Minor Water Accumulation / Nuisance Standing Water",
        "category": "Nuisance Advisory: Localized Ponding",
        "depth_estimate": "2 - 10 cm (Ankle-deep Puddles)",
        "hazard_summary": "Curbside ponding; leaf-clogged catch basins; shallow depressions.",
        "action": "Dispatch public works crews to clear sewer grates; warn low-lying parking lots.",
        "badge_color": "#eab308", # Yellow
        "threat_level": "Low"
    },
    4: {
        "level_name": "Shallow Street Flooding / Calf-Deep Inundation",
        "category": "Flood Warning: Urban Street Sheet Flow",
        "depth_estimate": "15 - 25 cm (Shin to Calf-deep)",
        "hazard_summary": "Submerged road curbs and sidewalks; bubbling manholes; slowed traffic.",
        "action": "Issue urban travel warnings; restrict low-clearance vehicles; inspect underpasses.",
        "badge_color": "#f59e0b", # Amber
        "threat_level": "Moderate"
    },
    5: {
        "level_name": "Moderate Flooding / Wheel-Height Inundation",
        "category": "Major Flood Warning: Vehicle Submersion Risk",
        "depth_estimate": "30 - 50 cm (Knee-deep / Car Wheel Height)",
        "hazard_summary": "Water reaching car wheel rims and bumpers; stalled sedans; doorway water ingress.",
        "action": "Deploy sandbag flood barriers; barricade flooded underpasses; activate high-capacity pumps.",
        "badge_color": "#f97316", # Orange
        "threat_level": "Substantial"
    },
    6: {
        "level_name": "Deep Inundation / Half-Submerged Vehicles",
        "category": "Emergency Phase I: Ground Floor Inundation",
        "depth_estimate": "60 - 100 cm (Waist-deep / Half-car Height)",
        "hazard_summary": "Vehicles submerged to windows/hoods; living room inundation; floating debris.",
        "action": "Mandatory evacuation of single-story residences; deploy search and rescue zodiac watercraft.",
        "badge_color": "#ef4444", # Red
        "threat_level": "High"
    },
    7: {
        "level_name": "Severe Flood / First-Floor Submerged",
        "category": "Emergency Phase II: Infrastructure Compromise",
        "depth_estimate": "1.5 - 3.0 m (First-floor Ceiling Level)",
        "hazard_summary": "Vehicles completely submerged (only roofs visible); roads washed out; substation blackout.",
        "action": "Declare State of Emergency; mobilize National Guard; launch motorized boat evacuations.",
        "badge_color": "#dc2626", # Deep Red
        "threat_level": "Severe"
    },
    8: {
        "level_name": "Catastrophic Flash Flood / Rooftop Stranded Victims",
        "category": "Disaster Declaration: Life-Saving Airborne Extraction",
        "depth_estimate": "3.0 - 5.0 m (Rooftop Inundation Level)",
        "hazard_summary": "Residents trapped on rooftops waving distress signals; stacked vehicles; raging surge.",
        "action": "Immediate tactical airborne helicopter winching; swiftwater rescue teams; mass shelters.",
        "badge_color": "#b91c1c", # Dark Red
        "threat_level": "Extreme / Critical"
    },
    9: {
        "level_name": "Critical Disaster / Structural Collapse & Life Peril",
        "category": "Catastrophic Crisis: Active Structural Collapse",
        "depth_estimate": "> 4.0 m (Raging Rapids / Structural Shearing)",
        "hazard_summary": "Inhabited buildings collapsing into torrents; violent whirlpools; dam breach waves.",
        "action": "Execute urgent emergency survival extractions; mandatory downriver sirens; emergency evacuations.",
        "badge_color": "#991b1b", # Crimson
        "threat_level": "Life-Threatening Imminent Peril"
    },
    10: {
        "level_name": "Extreme Disaster / Fatalities & Deceased Casualties",
        "category": "Mass Casualty Incident: DVI & Coroner Recovery",
        "depth_estimate": "Catastrophic Inundation (Fatal Casualties Confirmed)",
        "hazard_summary": "Dead bodies floating in floodwaters; drowned casualties in debris; biohazard risks.",
        "action": "Activate Disaster Victim Identification (DVI) units, forensic mortuary watercraft, and biohazard protocol.",
        "badge_color": "#581c87", # Purple/Black
        "threat_level": "Mass Fatality / Biohazard"
    }
}

def load_data():
    texts = []
    severities = []
    conf_scores = []
    
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Emphasize keywords and specific physical depth cues
            combined = f"{row['keywords']} {row['keywords']} {row['hazard_indicators']} {row['image_caption']}"
            texts.append(combined)
            severities.append(int(row['severity_score']))
            conf_scores.append(float(row['confidence_score']))
            
    return texts, severities, conf_scores

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Loading flood severity dataset (2,500 samples)...")
    texts, severities, conf_scores = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, severities, test_size=0.2, random_state=42, stratify=severities
    )
    
    # Advanced TF-IDF with domain stop words to emphasize physical markers
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        sublinear_tf=True,
        stop_words=DOMAIN_STOP_WORDS,
        min_df=2
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 1. Base Multinomial Logistic Regression
    base_clf = LogisticRegression(
        C=3.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    
    # Calibrate probability outputs for trustworthy government confidence scores
    calibrated_clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
    calibrated_clf.fit(X_train_vec, y_train)
    
    # 2. Continuous Regressor with L2 regularization
    regressor = Ridge(alpha=2.0, solver="auto", random_state=42)
    regressor.fit(X_train_vec, y_train)
    
    # Evaluation
    y_pred = calibrated_clf.predict(X_test_vec)
    y_pred_cont = regressor.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred_cont)
    r2 = r2_score(y_test, y_pred_cont)
    
    print("\n" + "="*65)
    print("GOVERNMENT-GRADE ML MODEL EVALUATION RESULTS (TEST SET: 500)")
    print("="*65)
    print(f"Stratified Classification Accuracy : {acc * 100:.2f}%")
    print(f"Continuous Score MAE              : {mae:.3f} severity points")
    print(f"Continuous Score R2               : {r2:.3f}")
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_test, y_pred, digits=3))
    
    # Save model bundle
    bundle = {
        "vectorizer": vectorizer,
        "classifier": calibrated_clf,
        "regressor": regressor,
        "metadata": SEVERITY_METADATA,
        "classes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
        
    print(f"High-precision model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train()
