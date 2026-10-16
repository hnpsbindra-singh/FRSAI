#!/usr/bin/env python3
"""
Interactive / CLI Search and Filter Utility for Flood Severity Synthetic Dataset
Usage:
    python3 query_dataset.py --severity 10
    python3 query_dataset.py --keyword "dead body"
    python3 query_dataset.py --keyword "sunny"
    python3 query_dataset.py --min-confidence 0.95 --limit 5
"""

import csv
import argparse

CSV_PATH = "/Users/harnimarpreetsingh/Desktop/Flood/flood_severity_dataset.csv"

def query(severity=None, keyword=None, min_confidence=None, confidence_level=None, limit=5):
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        
    results = reader
    
    if severity is not None:
        results = [r for r in results if int(r["severity_score"]) == severity]
        
    if keyword:
        kw_clean = keyword.lower().strip()
        results = [r for r in results if kw_clean in r["keywords"].lower() or kw_clean in r["image_caption"].lower()]
        
    if min_confidence is not None:
        results = [r for r in results if float(r["confidence_score"]) >= min_confidence]
        
    if confidence_level:
        results = [r for r in results if r["confidence_level"].lower() == confidence_level.lower()]
        
    print(f"Matched {len(results)} records (showing up to {limit}):\n")
    for idx, r in enumerate(results[:limit], 1):
        print(f"[{idx}] {r['id']} | Severity: {r['severity_score']}/10 ({r['severity_level']})")
        print(f"    Confidence : {r['confidence_level']} (Score: {r['confidence_score']})")
        print(f"    Rationale  : {r['confidence_rationale']}")
        print(f"    Depth Est. : {r['water_depth_estimate']}")
        print(f"    Setting    : {r['visual_context_setting']}")
        print(f"    Keywords   : {r['keywords']}")
        print(f"    Caption    : {r['image_caption']}")
        print(f"    Hazards    : {r['hazard_indicators']}")
        print(f"    Action     : {r['recommended_action']}")
        print("-" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Flood Severity Dataset")
    parser.add_argument("--severity", type=int, choices=range(1, 11), help="Severity score (1-10)")
    parser.add_argument("--keyword", type=str, help="Search term in keywords or caption")
    parser.add_argument("--min-confidence", type=float, help="Minimum confidence score (0.0 - 1.0)")
    parser.add_argument("--confidence-level", type=str, choices=["Very High", "High", "Medium-High"], help="Confidence level filter")
    parser.add_argument("--limit", type=int, default=5, help="Number of records to display")
    
    args = parser.parse_args()
    query(
        severity=args.severity,
        keyword=args.keyword,
        min_confidence=args.min_confidence,
        confidence_level=args.confidence_level,
        limit=args.limit
    )
