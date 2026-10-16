import os
import base64
import json
import io
import requests
from PIL import Image
from flask import Flask, request, jsonify, render_template
from disaster_evaluator import GovernmentDisasterEvaluator

app = Flask(__name__, static_folder="static", template_folder="templates")

def _load_env():
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")

evaluator = GovernmentDisasterEvaluator()

def normalize_and_optimize_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            mask = img.split()[-1] if "A" in img.mode else None
            bg.paste(img, mask=mask)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
            
        max_dim = 1600
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception as e:
        return image_bytes, "image/jpeg"

def analyze_image_with_vision(image_bytes):
    normalized_bytes, safe_mime = normalize_and_optimize_image(image_bytes)
    b64_image = base64.b64encode(normalized_bytes).decode("utf-8")
    data_url = f"data:{safe_mime};base64,{b64_image}"

    system_prompt = (
        "You are an Emergency Scene Grounding Officer for a national disaster management agency.\n"
        "Your task is to provide an objective, fact-based visual description of the flood or disaster scene.\n"
        "DO NOT ASSIGN ANY NUMERICAL SCORE OR SEVERITY RATING. A separate trained machine learning model will calculate the score.\n\n"
        "CRITICAL PHYSICAL ANCHOR RULES (BE ACCURATE, GROUND IN PHYSICAL EVIDENCE):\n"
        "- FLOATING RAFTS & BOATS: If people are navigating on a makeshift raft (e.g. banana tree trunks, bamboo, wooden planks) or boat using a punting pole or oars, the flood is DEEP and UNPASSABLE on foot (typically 1.5 to 3+ meters deep, submerging village houses, trees, and roads). Rafts and boats require deep water to float. NEVER describe raft or boat navigation as ankle-deep or minor standing water!\n"
        "- If someone is actively RIDING A BICYCLE or walking easily on road pavement, the water is shallow (< 20 cm ankle-deep or shin-deep sheet flow).\n"
        "- If passenger cars are driving through water and not stalled, with water only touching bottom of tire treads, this is shallow street water.\n"
        "- Only describe water as knee-deep (30-50 cm) if vehicles are stalled with water covering wheel rims and bumpers, or water is entering doorways.\n"
        "- Only describe water as waist-deep (60-100 cm) if water reaches halfway up car doors or window sills.\n"
        "- Describe water as deep/severe (1.5-3.0 m) if houses/trees are submerged past lower walls, cars are submerged to roofs, or boat/raft is required for transport.\n"
        "- If multi-story buildings are collapsed, sheared, or buried in deep river silt/mudflow from a flash flood, explicitly describe structural collapse, debris flow, and building destruction. In mudslide and debris flow scenes, focus on the severe structural failure and sediment devastation; do NOT describe post-flood mudfields as 'dry' or 'normal'.\n"
        "- If deceased human bodies or corpses are confirmed in water or debris, explicitly describe dead bodies.\n\n"
        "Return STRICTLY a JSON object with this schema:\n"
        "{\n"
        "  \"water_present\": true/false,\n"
        "  \"scene_description\": \"Detailed 2-4 sentence factual description focusing on water depth, submerged objects, human activity, and infrastructure condition.\",\n"
        "  \"water_depth_estimate_text\": \"e.g. 0 cm / Completely Dry | Ankle-deep 5-10 cm | Shin-deep 15-20 cm | Knee-deep 40 cm | Waist-deep 80 cm | Deep multi-meter flood 1.5-3 m (raft/boat navigation) | Debris flow / Torrent > 4 m\",\n"
        "  \"submerged_landmarks\": [\"village_houses\", \"trees\", \"farmland\", ...],\n"
        "  \"human_impact\": \"normal_walking | cycling_passable | cautious_wading | waist_deep_wading | raft_boat_evacuation | trapped_in_vehicle | stranded_on_roof | confirmed_fatalities\",\n"
        "  \"damage_level\": \"none | minor_standing_water | ground_floor_ingress | severe_inundation_boat_only | structural_washout | building_collapse\",\n"
        "  \"fatalities_present\": true/false,\n"
        "  \"fatalities_details\": \"none\" or text description of bodies,\n"
        "  \"scene_setting\": \"Urban Commercial | Residential | Rural Village / Farmland | Mountain River | Highway Corridor | etc.\",\n"
        "  \"keywords\": [\"keyword1\", \"keyword2\", \"keyword3\", \"keyword4\", \"keyword5\", \"keyword6\", \"keyword7\", \"keyword8\"]\n"
        "}"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5050",
        "X-Title": "Gov Flood Assessment AI"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Describe this scene and extract its physical hydrological anchors for machine learning evaluation."
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": data_url}
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1000
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
    
    if response.status_code != 200:
        fallback_payload = dict(payload)
        fallback_payload["model"] = "nvidia/nemotron-3.5-super-vl:free"
        fallback_resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=fallback_payload, timeout=30)
        if fallback_resp.status_code == 200:
            response = fallback_resp
        else:
            raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")
        
    res_data = response.json()
    raw_content = res_data["choices"][0]["message"]["content"]
    
    try:
        return json.loads(raw_content)
    except Exception:
        clean_content = raw_content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_content)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status": "healthy",
        "compliance": "Government Disaster Incident Command System Standard (ICS)",
        "scale": "1 (Normal/Dry) to 10 (Fatalities/Mass Casualty)",
        "vision_model": OPENROUTER_MODEL,
        "has_api_key": bool(OPENROUTER_API_KEY)
    })

RESCUE_PRIORITY_MAP = {
    1: "Routine",
    2: "Routine",
    3: "Low",
    4: "Low",
    5: "Medium",
    6: "High",
    7: "High",
    8: "Critical",
    9: "Critical",
    10: "Critical"
}

ROAD_ACCESS_MAP = {
    1: "Fully accessible. Normal road and traffic conditions.",
    2: "Open and passable. Wet road surface; reduce speed on highway curves.",
    3: "Passable with caution. Shallow curbside ponding; road lanes open.",
    4: "Restricted access. Water spreading across lanes; low-clearance vehicles restricted.",
    5: "Impassable for sedans and small vehicles. High-clearance vehicles only.",
    6: "Impassable for motor vehicles. High-clearance or boat transport required.",
    7: "Completely impassable on land. Boat and watercraft access only.",
    8: "Cut off and isolated. Airborne helicopter or swiftwater rescue boat access only.",
    9: "Hazardous disaster zone. Washed-out bridges and sheared roads; tactical air rescue only.",
    10: "Restricted mass casualty recovery zone. Authorized emergency personnel only."
}

RECOMMENDATIONS_MAP = {
    1: [
        "Normal operations; standard municipal drainage monitoring.",
        "Ensure storm catch basins remain free of leaves and trash."
    ],
    2: [
        "Issue wet road traffic advisory for commuters.",
        "Advise motorists to reduce speed to prevent hydroplaning."
    ],
    3: [
        "Dispatch public works teams to clear blocked street sewer grates.",
        "Warn pedestrians of localized pooling in low-lying parking lots and sidewalks."
    ],
    4: [
        "Issue urban street sheet flow advisories across local traffic channels.",
        "Restrict low-clearance vehicles from flooded lanes and dip intersections.",
        "Monitor drainage channels for rapid water level changes."
    ],
    5: [
        "Barricade flooded avenues and underpasses; tow stalled passenger cars.",
        "Deploy sandbag defenses at commercial storefronts and ground-floor entrances.",
        "Activate district high-capacity stormwater dewatering pumps."
    ],
    6: [
        "Initiate mandatory ground-floor evacuation for affected residential homes.",
        "Deploy inflatable rescue rafts and swiftwater drysuit personnel.",
        "Disconnect ground-level electrical supplies and gas mains to avoid secondary fires."
    ],
    7: [
        "Declare State of Emergency; mobilize State Disaster Response Force (SDRF / NDRF).",
        "Deploy motorized rescue boats and watercraft for first-floor ceiling extractions.",
        "Establish regional shelter camps with clean drinking water and medical triage."
    ],
    8: [
        "Launch immediate airborne search and rescue; deploy helicopters with winch cables for rooftop strandings.",
        "Issue mandatory immediate evacuation sirens for all downriver communities.",
        "Mobilize military swiftwater teams and prepare casualty receiving centers."
    ],
    9: [
        "Execute immediate life-or-death emergency tactical extractions.",
        "Evacuate all personnel from collapsing riverbanks and unstable masonry buildings.",
        "Sound continuous disaster alert sirens along downstream river corridor."
    ],
    10: [
        "Deploy forensic mortuary teams and Disaster Victim Identification (DVI) units.",
        "Deploy coroner recovery watercraft with secure body recovery bags.",
        "Establish secure morgue facilities and victim identification family assistance centers."
    ]
}

def extract_image_bytes_from_request():
    if request.is_json:
        data = request.get_json(silent=True) or {}
        b64 = data.get("image_base64") or data.get("imageBase64") or ""
        if b64:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            return base64.b64decode(b64)
            
    if "image" in request.files:
        file = request.files["image"]
        if file.filename != "":
            return file.read()

    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            return file.read()

    if "image_base64" in request.form:
        b64 = request.form["image_base64"]
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    return None

@app.route("/api/v1/score-base64", methods=["POST"])
@app.route("/api/flood-score", methods=["POST"])
@app.route("/api/analyze-image", methods=["POST"])
def api_analyze_image():
    image_bytes = extract_image_bytes_from_request()
    if not image_bytes:
        return jsonify({"error": "No image data provided. Send 'image_base64' in JSON body or 'image' file in multipart/form-data."}), 400
        
    try:
        vision_result = analyze_image_with_vision(image_bytes)
        eval_result = evaluator.evaluate(vision_result)
        
        score = int(eval_result["severity_score"])
        severity_level = str(eval_result["severity_level"])
        rescue_priority = RESCUE_PRIORITY_MAP.get(score, "Medium")
        road_access = ROAD_ACCESS_MAP.get(score, "Exercise standard caution.")
        recommendations = RECOMMENDATIONS_MAP.get(score, [eval_result.get("recommended_action", "Stay in a secure elevated location.")])
        title = f"Level {score}: {severity_level}"
        description = f"{eval_result['fema_category']} — {eval_result['ics_operational_phase']}. {eval_result['confidence_rationale']}"
        what_is_in_image = str(vision_result.get("scene_description", "") or vision_result.get("reasoning", "") or vision_result.get("objective_description", ""))
        estimated_depth = str(eval_result.get("water_depth_estimate", "") or vision_result.get("water_depth_estimate_text", "Depth under evaluation"))
        confidence = round(float(eval_result.get("confidence_score", 0.95)), 4)

        response_payload = {
            "severity_score": score,
            "severity_level": severity_level,
            "rescue_priority": rescue_priority,
            "confidence": confidence,
            "title": title,
            "description": description,
            "what_is_in_image": what_is_in_image,
            "estimated_depth": estimated_depth,
            "road_access": road_access,
            "recommendations": recommendations,
            "success": True,
            "incident_report": {
                "severity_score": score,
                "continuous_score": eval_result["continuous_score"],
                "severity_level": severity_level,
                "fema_incident_category": eval_result["fema_category"],
                "threat_level": eval_result["threat_level"],
                "badge_color": eval_result["badge_color"],
                "confidence_score": confidence,
                "confidence_level": eval_result["confidence_level"],
                "confidence_rationale": eval_result["confidence_rationale"],
                "water_depth_estimate": estimated_depth,
                "submerged_landmarks_detected": eval_result["submerged_landmarks_detected"],
                "ics_operational_phase": eval_result["ics_operational_phase"],
                "ics_lead_agency": eval_result["ics_lead_agency"],
                "recommended_action": eval_result["recommended_action"],
                "consensus_details": eval_result["ml_consensus_details"]
            },
            "vision_grounding": {
                "objective_description": what_is_in_image,
                "exact_keywords": vision_result.get("keywords", []),
                "water_presence_verified": vision_result.get("water_present", True),
                "depth_measurement_category": eval_result.get("fema_category", "Hydrological Evaluation"),
                "physical_water_depth_text": vision_result.get("water_depth_estimate_text", ""),
                "scene_setting": vision_result.get("scene_setting", "Urban"),
                "human_impact_observed": vision_result.get("human_impact", "none"),
                "structural_damage_observed": vision_result.get("damage_level", "none"),
                "fatalities_detected": vision_result.get("fatalities_present", False)
            },
            "probability_distribution": eval_result["probability_distribution"]
        }
        return jsonify(response_payload)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze-text", methods=["POST"])
def api_analyze_text():
    data = request.get_json(silent=True) or {}
    text = data.get("text") or data.get("keywords") or ""
    
    if not text.strip():
        return jsonify({"error": "Please provide 'text' or 'keywords'"}), 400
        
    try:
        text_lower = text.lower()
        fatalities = any(w in text_lower for w in ["dead body", "corpse", "deceased", "drowned victim", "fatalities", "dead bodies"])
        water_present = not any(w in text_lower for w in ["zero water", "completely dry", "no water accumulation", "bone dry", "0 cm"])
        
        synthetic_vision = {
            "water_present": water_present,
            "scene_description": text,
            "water_depth_estimate_text": "",
            "submerged_landmarks": [],
            "fatalities_present": fatalities,
            "human_impact": "fatalities in water" if fatalities else "",
            "damage_level": "",
            "scene_setting": "Field Observation",
            "keywords": [k.strip() for k in text.replace(".", ",").split(",") if k.strip()]
        }
        
        eval_result = evaluator.evaluate(synthetic_vision)
        return jsonify({
            "success": True,
            "incident_report": {
                "severity_score": eval_result["severity_score"],
                "continuous_score": eval_result["continuous_score"],
                "severity_level": eval_result["severity_level"],
                "fema_incident_category": eval_result["fema_category"],
                "threat_level": eval_result["threat_level"],
                "badge_color": eval_result["badge_color"],
                "confidence_score": eval_result["confidence_score"],
                "confidence_level": eval_result["confidence_level"],
                "confidence_rationale": eval_result["confidence_rationale"],
                "water_depth_estimate": eval_result["water_depth_estimate"],
                "ics_operational_phase": eval_result["ics_operational_phase"],
                "ics_lead_agency": eval_result["ics_lead_agency"],
                "recommended_action": eval_result["recommended_action"],
                "consensus_details": eval_result["ml_consensus_details"]
            },
            "probability_distribution": eval_result["probability_distribution"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
