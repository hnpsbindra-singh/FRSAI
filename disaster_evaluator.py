from flood_predictor import FloodSeverityPredictor

ICS_PHASES = {
    1: {"phase": "Level IV: Routine Municipal Monitoring", "lead_agency": "Municipal Works & Meteorological Dept"},
    2: {"phase": "Level IV: Wet Surface Traffic Advisory", "lead_agency": "Traffic Police / Highway Authority"},
    3: {"phase": "Level III: Localized Nuisance Drainage Advisory", "lead_agency": "Municipal Drainage & Sanitation Dept"},
    4: {"phase": "Level III: Urban Road Sheet Flow Warning", "lead_agency": "District Disaster Management Authority (DDMA)"},
    5: {"phase": "Level II: Major Flood Warning — Vehicle Inundation", "lead_agency": "State Disaster Response Force (SDRF) / Fire Dept"},
    6: {"phase": "Level II: Deep Inundation — Ground Floor Evacuation", "lead_agency": "SDRF / Civil Defense Quick Response Teams"},
    7: {"phase": "Level I: State of Emergency — Severe Inundation", "lead_agency": "NDRF / National Guard (Motorized Boats & Watercraft)"},
    8: {"phase": "Level I: Catastrophic Flash Flood — Airborne Extraction", "lead_agency": "Armed Forces Helicopter & Swiftwater Rescue Units"},
    9: {"phase": "Level I: Active Structural Collapse — Life Extraction", "lead_agency": "National Emergency Crisis Management Command"},
    10: {"phase": "Special Phase: Mass Casualty Incident — DVI & Mortuary", "lead_agency": "Forensic Mortuary Corps / Coroner Recovery Teams"}
}

class GovernmentDisasterEvaluator:
    def __init__(self):
        self.predictor = FloodSeverityPredictor()

    def evaluate(self, vision_data):
        desc = (vision_data.get("scene_description", "") or 
                vision_data.get("reasoning", "") or 
                vision_data.get("objective_description", "")).strip()
        keywords = vision_data.get("keywords", [])
        keywords_str = " ".join(keywords).lower()
        depth_text = vision_data.get("water_depth_estimate_text", "").strip()
        submerged = vision_data.get("submerged_landmarks", []) or vision_data.get("submerged_objects", [])
        submerged_clean = [s.lower().replace("_", " ") for s in submerged if s and s != "none"]
        human_impact = vision_data.get("human_impact", "").replace("_", " ")
        damage = vision_data.get("damage_level", "").replace("_", " ")
        setting = vision_data.get("scene_setting", "")
        water_present = vision_data.get("water_present", True)
        fatalities = vision_data.get("fatalities_present", False)
        desc_lower = desc.lower()

        ml_input_parts = [
            " ".join(keywords),
            desc,
            depth_text,
            " ".join(submerged_clean),
            human_impact,
            damage,
            setting
        ]
        ml_input = " ".join([p for p in ml_input_parts if p]).strip()

        ml_res = self.predictor.predict(ml_input)
        ml_score = ml_res["severity_score"]
        ml_continuous = ml_res["continuous_score"]
        ml_expected = ml_res["expected_score"]
        conf_score = ml_res["confidence_score"]
        conf_level = ml_res["confidence_level"]
        conf_rationale = ml_res["confidence_rationale"]

        final_severity = ml_score
        continuous_score = ml_continuous
        audit_trail = [
            f"ML Model Pipeline evaluated visual description → Predicted Severity Level {ml_score} (Expected Value: {ml_expected:.2f}, Continuous: {ml_continuous}).",
            f"Top active model features: {', '.join(ml_res.get('top_detected_keywords', [])[:6])}."
        ]
        override_applied = False

        has_fatalities_text = (
            ("dead body" in desc_lower or "corpse" in desc_lower or "deceased victim" in desc_lower or
             "drowned victim" in desc_lower or "dead bodies" in desc_lower or "floating body" in desc_lower) and
            not any(neg in desc_lower for neg in ["no dead", "no corpse", "no body", "no bodies", "no drowned", "without fatalities", "no fatalities"])
        )
        if fatalities is True or has_fatalities_text:
            final_severity = 10
            continuous_score = 10.0
            conf_rationale = "CRITICAL PROTOCOL: Confirmed visual evidence of deceased victims / dead bodies. Level 10 Mass Casualty Protocol locked."
            audit_trail.append("SAFETY GUARD: Deceased casualty verified → Level 10 locked.")
            override_applied = True

        has_catastrophic_damage = any(k in desc_lower or k in keywords_str for k in [
            "building collapse", "collapsed", "collapse", "debris flow", "mudflow", "landslide", "washout", "destruction", "sheared"
        ])
        if not override_applied and final_severity <= 2 and has_catastrophic_damage:
            probs = ml_res.get("probability_distribution", {})
            probs_disaster = {int(k.replace("Level_", "")): v for k, v in probs.items() if int(k.replace("Level_", "")) >= 7}
            if probs_disaster:
                final_severity = max(probs_disaster, key=probs_disaster.get)
            else:
                final_severity = 9
            continuous_score = float(final_severity)
            conf_rationale = f"DISASTER RE-ANCHOR: Severe structural collapse & debris flow detected. Re-anchored to ML's top disaster class: Level {final_severity}."
            audit_trail.append(f"DOMAIN GUARD: Structural collapse / debris flow present. Re-anchored to ML's top disaster class: Level {final_severity}.")
            override_applied = True

        is_raft_or_boat = any(w in desc_lower or w in keywords_str or w in human_impact for w in [
            "raft", "boat", "bhela", "canoe", "dinghy", "punting pole", "raft navigation", "makeshift raft", "watercraft"
        ])
        if not override_applied and is_raft_or_boat and final_severity < 6:
            probs = ml_res.get("probability_distribution", {})
            probs_deep = {int(k.replace("Level_", "")): v for k, v in probs.items() if int(k.replace("Level_", "")) in [6, 7]}
            if probs_deep:
                final_severity = max(probs_deep, key=probs_deep.get)
            else:
                final_severity = 6
            continuous_score = float(final_severity)
            conf_rationale = "WATERCRAFT NAVIGATION GROUNDING: Evacuation via raft/boat confirms deep unpassable floodwaters (1.5-3 m). Re-anchored to Deep Inundation protocol."
            audit_trail.append(f"PHYSICAL GROUNDING: Watercraft/raft navigation physically impossible in shallow water → Re-anchored to Level {final_severity}.")
            override_applied = True

        elif (not override_applied and
              water_present is False and
              damage in ["none", "", "minor standing water"] and
              not any(k in desc_lower for k in ["collapse", "collapsed", "destruction", "washout", "debris", "ruins", "landslide", "mudflow", "flood", "torrent"]) and
              final_severity <= 2):
            final_severity = 1
            continuous_score = 1.0
            conf_rationale = "VERIFIED: Zero floodwater confirmed. Intact infrastructure and dry ground. Level 1 baseline locked."
            audit_trail.append("SAFETY GUARD: Completely dry benign scene confirmed → Level 1 locked.")
            override_applied = True

        meta = self.predictor.metadata.get(final_severity, {})
        ics_info = ICS_PHASES.get(final_severity, {})

        return {
            "severity_score": final_severity,
            "continuous_score": continuous_score,
            "severity_level": meta.get("level_name", f"Level {final_severity}"),
            "fema_category": meta.get("category", "Incident Evaluation"),
            "threat_level": meta.get("threat_level", "Standard"),
            "badge_color": meta.get("badge_color", "#3b82f6"),
            "confidence_score": conf_score,
            "confidence_level": conf_level,
            "confidence_rationale": conf_rationale,
            "water_depth_estimate": meta.get("depth_estimate", depth_text or "Variable"),
            "submerged_landmarks_detected": submerged_clean or ["None (dry infrastructure)"],
            "ics_operational_phase": ics_info.get("phase", "Standard Response"),
            "ics_lead_agency": ics_info.get("lead_agency", "Local Emergency Management"),
            "recommended_action": meta.get("action", "Assess local situation."),
            "ml_consensus_details": {
                "ml_model_name": "Trained NLP CalibratedClassifierCV + Ridge",
                "ml_predicted_score": ml_score,
                "ml_expected_score": round(ml_expected, 2),
                "final_score": final_severity,
                "override_applied": override_applied,
                "audit_notes": audit_trail
            },
            "probability_distribution": ml_res.get("probability_distribution", {})
        }
