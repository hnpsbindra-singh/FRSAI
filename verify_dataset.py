#!/usr/bin/env python3
"""
Verification and Statistical Analysis Script for Flood Severity Synthetic Dataset
"""

import csv
import json
from collections import Counter, defaultdict

CSV_PATH = "/Users/harnimarpreetsingh/Desktop/Flood/flood_severity_dataset.csv"
JSON_PATH = "/Users/harnimarpreetsingh/Desktop/Flood/flood_severity_dataset.json"

def verify():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        
    print(f"============================================================")
    print(f"DATASET VERIFICATION & QUALITY REPORT")
    print(f"============================================================")
    print(f"Total records in CSV: {len(reader)}")
    
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    print(f"Total records in JSON: {len(json_data)}")
    assert len(reader) == len(json_data), "Mismatch between CSV and JSON record counts!"
    assert len(reader) >= 2000, "Dataset contains fewer than 2000 entries!"
    
    # Severity distribution
    severity_counts = Counter(int(row["severity_score"]) for row in reader)
    print(f"\n--- Severity Score Distribution (1-10) ---")
    for score in range(1, 11):
        print(f"  Severity {score:2d}: {severity_counts[score]:4d} records ({reader[0]['severity_level'] if score==1 else ''})")
        assert severity_counts[score] > 0, f"Severity score {score} has no records!"
        
    # Confidence level distribution
    confidence_counts = Counter(row["confidence_level"] for row in reader)
    print(f"\n--- Confidence Level Distribution ---")
    for level, count in confidence_counts.most_common():
        print(f"  {level:12s}: {count:4d} records ({count/len(reader)*100:.1f}%)")
        
    scores = [float(row["confidence_score"]) for row in reader]
    print(f"  Confidence score range: {min(scores):.3f} to {max(scores):.3f} (Mean: {sum(scores)/len(scores):.3f})")
    
    # Check field completeness
    fields = reader[0].keys()
    print(f"\n--- Columns / Schema ---")
    for field in fields:
        empty_count = sum(1 for row in reader if not row[field] or row[field].strip() == "")
        print(f"  {field:25s}: {len(reader) - empty_count}/{len(reader)} populated (Empty: {empty_count})")
        assert empty_count == 0, f"Field '{field}' has empty values!"

    # Sample verification for edge cases
    print(f"\n============================================================")
    print(f"SAMPLE VERIFICATION: SEVERITY 1 (Completely Normal Situation)")
    print(f"============================================================")
    sev1_samples = [row for row in reader if int(row["severity_score"]) == 1][:3]
    for idx, s in enumerate(sev1_samples, 1):
        print(f"[{idx}] ID: {s['id']} | Confidence: {s['confidence_level']} ({s['confidence_score']})")
        print(f"    Keywords: {s['keywords']}")
        print(f"    Caption : {s['image_caption']}")
        print(f"    Depth   : {s['water_depth_estimate']}")
        print(f"    Action  : {s['recommended_action']}\n")

    print(f"============================================================")
    print(f"SAMPLE VERIFICATION: SEVERITY 10 (Dead Body / Extreme Casualties)")
    print(f"============================================================")
    sev10_samples = [row for row in reader if int(row["severity_score"]) == 10][:3]
    for idx, s in enumerate(sev10_samples, 1):
        print(f"[{idx}] ID: {s['id']} | Confidence: {s['confidence_level']} ({s['confidence_score']})")
        print(f"    Keywords: {s['keywords']}")
        print(f"    Caption : {s['image_caption']}")
        print(f"    Depth   : {s['water_depth_estimate']}")
        print(f"    Action  : {s['recommended_action']}\n")

    # Sample intermediate severities
    for test_sev in [3, 5, 7, 9]:
        sample = next(row for row in reader if int(row["severity_score"]) == test_sev)
        print(f"--- Sample Severity {test_sev}: {sample['severity_level']} ---")
        print(f"    Keywords: {sample['keywords']}")
        print(f"    Caption : {sample['image_caption']}")
        print(f"    Confidence: {sample['confidence_level']} ({sample['confidence_score']})\n")

    # Keyword uniqueness and vocabulary size
    all_keywords = set()
    for row in reader:
        for kw in row["keywords"].split(","):
            all_keywords.add(kw.strip().lower())
    print(f"============================================================")
    print(f"VOCABULARY & UNIQUENESS SUMMARY")
    print(f"============================================================")
    print(f"Total Unique Keyword Phrases in Dataset: {len(all_keywords)}")
    unique_captions = len(set(row["image_caption"] for row in reader))
    print(f"Total Unique Image Captions: {unique_captions} / {len(reader)}")
    print(f"Verification completed successfully!")

if __name__ == "__main__":
    verify()
