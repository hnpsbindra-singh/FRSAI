#!/usr/bin/env python3
"""
Flood Severity Synthetic Dataset Generator
Generates a realistic, highly varied synthetic dataset of image keywords, captions,
and metadata mapped to flood severity scores (1 to 10) with confidence levels.
Total target entries: 2,500 (250 entries per severity level from 1 to 10).
"""

import csv
import json
import random
import os

# Set seed for reproducibility while maintaining broad diversity
random.seed(42)

SEVERITY_CONFIG = {
    1: {
        "level_name": "Normal / Dry Conditions",
        "depth_range": "0 cm (Completely Dry)",
        "hazard_default": ["none", "normal_environment"],
        "action_default": "Routine monitoring; no flood hazard detected.",
        "settings": [
            "Urban Downtown", "Suburban Neighborhood", "Highway Interchange", 
            "Rural Countryside", "City Park", "Commercial Strip", 
            "Riverbank Promenade", "Industrial Park", "Residential Cul-de-sac", 
            "Mountain Road", "Coastal Boulevard", "Underpass Highway"
        ],
        "confidence_distributions": [
            (0.95, 0.99, "Very High", "Unambiguous visual absence of standing water; clear dry pavement and standard conditions."),
            (0.90, 0.94, "High", "Clear dry ground with normal environmental factors and unobstructed drainage."),
            (0.85, 0.89, "Medium-High", "Predominantly dry scene with minor shadows or dry foliage; verified non-hazardous.")
        ],
        "subjects": [
            ("sunny asphalt street", "pedestrians walking in light clothing", "operational traffic lights", "dry concrete sidewalk"),
            ("dry suburban driveway", "parked family sedan", "mowed green lawn", "clean street gutters"),
            ("clear city intersection", "unobstructed storm drain grates", "normal vehicular traffic", "dry crosswalk"),
            ("arid rural dirt road", "dust rising behind tractor", "dry roadside ditches", "clear culverts under roadway"),
            ("municipal park walkway", "joggers on dry paved path", "dry park benches", "functional drainage channel"),
            ("highway overpass", "fast-moving vehicles", "dry asphalt lanes", "clean concrete road barriers"),
            ("commercial shopping plaza", "dry outdoor seating", "customers entering retail shops", "clean dry parking lot"),
            ("riverbank observation deck", "river flowing peacefully within banks", "dry walking trail", "exposed stone embankment"),
            ("downtown underpass", "bone-dry road surface", "bright tunnel illumination", "dry curbs and curbside lane"),
            ("industrial warehouse apron", "dry loading docks", "parked delivery trucks", "clean drainage trenches"),
            ("suburban school zone", "dry pedestrian crosswalk", "school buses parked safely", "dry paved playground"),
            ("mountain pass roadway", "dry gravel shoulders", "clear runoff channels", "dry rocky slope beside road")
        ],
        "modifiers": [
            "bright midday sunlight", "clear blue sky with light clouds", "crisp morning light", 
            "dry mild afternoon weather", "sunbaked pavement texture", "clean street sweeping marks", 
            "dusty dry curb lines", "unobstructed drainage inlets", "dry fallen autumn leaves", 
            "calm pleasant atmospheric conditions", "dry asphalt reflection in bright sun", "completely dry storm sewers"
        ],
        "extra_keywords": [
            "no water accumulation", "dry surface", "zero standing water", "clear drainage", 
            "unhindered traffic", "safe pedestrian zone", "standard road conditions", "optimal weather", 
            "dry road corridor", "zero inundation", "stable river level", "bone dry ground"
        ]
    },
    2: {
        "level_name": "Normal Wet Weather / Light Surface Moisture",
        "depth_range": "< 2 cm (Thin Water Sheen / Dampness)",
        "hazard_default": ["wet_pavement", "minor_slip_hazard"],
        "action_default": "Standard wet weather driving caution; maintain routine drainage maintenance.",
        "settings": [
            "Urban Street", "Suburban Road", "Highway Corridor", 
            "Commercial Plaza", "Downtown Sidewalk", "Park Pathway", 
            "Residential Street", "Rural Highway", "Parking Facility", "Transit Terminal"
        ],
        "confidence_distributions": [
            (0.95, 0.99, "Very High", "Distinct surface reflection and fine droplets consistent with light rain/drizzle without ponding."),
            (0.89, 0.94, "High", "Observable moisture sheen and functional runoff along curb lines."),
            (0.84, 0.88, "Medium-High", "Damp asphalt and wet curbs; water volume well below puddle accumulation threshold.")
        ],
        "subjects": [
            ("light drizzle on asphalt", "glistening wet pavement", "pedestrians with open umbrellas", "damp sidewalk"),
            ("recent rain shower", "wet asphalt road sheen", "slow windshield wiper motion", "damp curb stones"),
            ("overcast morning street", "thin water film on road surface", "reflecting streetlights", "damp autumn leaves"),
            ("commercial parking area", "scattered shallow moisture spots", "damp parking stripes", "tires leaving damp tracks"),
            ("suburban neighborhood lane", "gentle trickle in concrete gutter", "wet garden path", "damp parked cars"),
            ("highway during light mist", "faint spray from passenger car tires", "wet reflective asphalt", "damp road shoulder"),
            ("downtown retail avenue", "wet cobblestone walkway", "raindrops on store window displays", "damp awning edges"),
            ("rural paved lane", "damp bitumen surface", "wet roadside grass", "moist soil along shoulder"),
            ("city transit bus stop", "commuters sheltered under wet canopy", "damp boarding area", "shiny wet street"),
            ("residential driveway", "damp interlocking pavers", "beaded water droplets on car paint", "clean gutter flow")
        ],
        "modifiers": [
            "overcast grey clouds", "gentle passing drizzle", "cool misty ambiance", 
            "reflective street surface sheen", "fine airborne rain droplets", "light pavement dampness", 
            "glistening street asphalt", "intermittent light showers", "moist pavement texture", 
            "free-flowing curb runoff", "clear sewer grates accepting trickle", "damp road markings"
        ],
        "extra_keywords": [
            "surface sheen", "light drizzle", "damp asphalt", "fine droplets", 
            "shallow dampness", "slick pavement", "minimal runoff", "no standing water", 
            "passing shower", "glistening road", "moist ground", "standard wet conditions"
        ]
    },
    3: {
        "level_name": "Minor Water Accumulation / Nuisance Standing Water",
        "depth_range": "2 - 10 cm (Ankle-deep Puddles)",
        "hazard_default": ["standing_water", "clogged_drain", "hydroplaning_risk"],
        "action_default": "Clear leaf debris from grates; monitor low-lying parking lots and pedestrian pathways.",
        "settings": [
            "Shopping Center Parking Lot", "Residential Intersection", "Downtown Crosswalk", 
            "Suburban Curb Lane", "School Drop-off Zone", "Alleyway", 
            "Park Perimeter", "Commercial Driveway", "Older Neighborhood Street", "Industrial Yard"
        ],
        "confidence_distributions": [
            (0.94, 0.98, "Very High", "Clear visual markers of shallow puddling and localized gutter overflow not exceeding curb height."),
            (0.89, 0.93, "High", "Visible ankle-depth water pooling around storm drains and depressed pavement sections."),
            (0.84, 0.88, "Medium-High", "Localized ponding; visual cues confirm water is contained within low-lying depressions.")
        ],
        "subjects": [
            ("ankle-deep puddle", "clogged sewer grate with wet leaves", "car tire throwing water plume", "standing water near curb"),
            ("low-lying parking lot depression", "standing rainwater pool", "pedestrians stepping carefully around puddle", "wet curbside"),
            ("residential street gutter overflowing", "shallow water ponding", "water pooling around front tire base", "soggy turf lawn"),
            ("downtown pedestrian crosswalk", "water accumulating in asphalt dip", "pedestrians jumping over puddle", "damp shop entrance"),
            ("narrow city alleyway", "standing puddle covering cobblestones", "slow drainage near brick wall", "shallow muddy film"),
            ("suburban cul-de-sac", "rainwater gathering at lowest point", "clogged catch basin", "children splashing in rubber boots"),
            ("commercial strip mall entrance", "water backing up around driveway ramp", "caution cone near puddle", "slow moving sedan"),
            ("older neighborhood intersection", "shallow standing water across gutter lane", "cyclist steering around ponding", "wet sidewalk curb"),
            ("industrial warehouse yard", "puddles collecting in tire ruts", "water pooling near drainage ditch", "damp gravel apron"),
            ("park edge walking path", "overflowing bioswale", "standing water across gravel track", "muddy puddle fringe")
        ],
        "modifiers": [
            "debris partially blocking drain inlet", "leaf litter trapped in storm grate", "persistent standing puddle", 
            "cloudy rainwater accumulation", "slow percolation into soil", "rippling puddle surface under breeze", 
            "shallow splash from vehicle tires", "curb-level water buildup", "minor localized ponding", 
            "spongy saturated grass edge", "intermittent raindrop rings", "contained gutter flooding"
        ],
        "extra_keywords": [
            "ankle deep water", "standing puddle", "blocked sewer grate", "gutter overflow", 
            "nuisance ponding", "curb water pooling", "shallow standing pool", "splashing tires", 
            "waterlogged curb", "minor drainage backup", "shallow pool", "passable street water"
        ]
    },
    4: {
        "level_name": "Shallow Street Flooding / Calf-Deep Inundation",
        "depth_range": "15 - 25 cm (Shin to Calf-deep)",
        "hazard_default": ["submerged_curbs", "bubbling_drain", "slow_traffic", "localized_ingress"],
        "action_default": "Issue low-lying road advisories; deploy maintenance crews to clear trunk drains and inspect underpasses.",
        "settings": [
            "Urban Residential Boulevard", "Low-lying Commercial Street", "Suburban Intersection", 
            "Flooded Underpass Entrance", "Coastal Town Road", "Industrial Access Way", 
            "Apartment Complex Road", "Highway Off-ramp", "Market District Lane", "River Road"
        ],
        "confidence_distributions": [
            (0.94, 0.98, "Very High", "Unambiguous submersion of street curbs and sidewalks with water reaching lower car hubcaps."),
            (0.88, 0.93, "High", "Consistent calf-deep water level with bubbling storm drains and slow wading pedestrians."),
            (0.83, 0.87, "Medium-High", "Widespread shallow street sheet flow; clear depth cues from vehicle tires and curbs.")
        ],
        "subjects": [
            ("calf-deep brown street water", "completely submerged sidewalk curbs", "storm drain bubbling upward", "slow crawling traffic"),
            ("flooded residential intersection", "water covering road from lawn to lawn", "pedestrian wading in high boots", "water lapping garden wall"),
            ("commercial street with shin-deep water", "murky brown sheet flow", "water reaching bottom of parked car rims", "sandbags ready on porch"),
            ("underpass approach road", "water pooling across both lanes", "caution barriers placed by police", "slow-moving transit bus"),
            ("coastal neighborhood avenue", "high tide stormwater backflow", "submerged bike lane", "water lapping against residential driveways"),
            ("downtown market road", "shallow brown runoff rushing past shops", "shopkeepers sweeping water from doorways", "floating plastic cups"),
            ("suburban avenue culvert overflow", "murky water spreading across asphalt", "car navigating slowly through wake", "flooded lawn borders"),
            ("industrial access road", "calf-deep brown water", "submerged curb stones", "delivery van pushing small bow wave"),
            ("residential boulevard", "stormwater bubbling out of manhole cover", "brown standing water", "water approaching garage door threshold"),
            ("rural highway edge", "overflowing irrigation ditch spilling onto road", "shallow muddy floodwater", "submerged road shoulder lines")
        ],
        "modifiers": [
            "brown sediment-laden water", "manhole cover vibrating under hydraulic pressure", "water submerging road markings", 
            "small bow wave created by crawling vehicles", "sidewalks invisible beneath water", "water level reaching car wheel rims", 
            "turbid stormwater runoff", "debris floating on water surface", "drainage system completely overwhelmed", 
            "residents observing water level anxiously", "curb lines obliterated by water", "steady shallow current"
        ],
        "extra_keywords": [
            "calf deep water", "shin deep flood", "submerged curbs", "bubbling manhole", 
            "hidden sidewalks", "wheel rim water", "street sheet flow", "wading pedestrians", 
            "slow moving traffic", "murky brown street", "overwhelmed storm drain", "shallow road inundation"
        ]
    },
    5: {
        "level_name": "Moderate Flooding / Knee-Deep / Wheel-Height Inundation",
        "depth_range": "30 - 50 cm (Knee-deep / Car Wheel Height)",
        "hazard_default": ["stalled_vehicles", "sandbagged_storefronts", "ground_floor_ingress", "canal_overtopping"],
        "action_default": "Deploy emergency sandbagging; close flooded underpasses; activate flood pumping stations; issue travel bans.",
        "settings": [
            "Flooded Commercial Downtown", "Residential Neighborhood", "Urban Underpass", 
            "Canal Embankment Road", "Suburban Shopping Strip", "Hospital District Access", 
            "Industrial Park Street", "Riverside Community", "Low-lying Highway Section", "Transit Hub Entrance"
        ],
        "confidence_distributions": [
            (0.95, 0.99, "Very High", "Distinct visual confirmation of water reaching car wheel tops/exhaust pipes with sandbagged thresholds."),
            (0.89, 0.94, "High", "Clear knee-deep water depth indicators on wading adults, stalled compact cars, and overflowing canals."),
            (0.84, 0.88, "Medium-High", "Substantial flood depth across entire street corridor with active ground-floor water ingress.")
        ],
        "subjects": [
            ("knee-deep murky floodwater", "stalled compact sedan with hazard lights flashing", "sandbag barriers stacked at shop doors", "water entering ground-floor foyer"),
            ("flooded commercial avenue", "water reaching car bumper level", "floating trash dumpsters", "residents wading with trousers rolled above knees"),
            ("overflowing urban canal", "muddy water pouring over stone retaining wall", "flooded dual carriageway", "abandoned passenger cars in middle of street"),
            ("residential street inundation", "knee-high brown floodwater", "water flooding garage interiors", "residents pushing stalled car to higher ground"),
            ("suburban underpass filled with knee-deep water", "submerged sedan wheels", "yellow police tape cordoning road", "brown muddy swirl"),
            ("busy downtown intersection flooded", "water touching door bottoms of parked vehicles", "storefront sandbags holding back water", "floating plastic crates"),
            ("riverside village street", "knee-deep brown river water", "sandbags around home entrances", "floating wooden pallets and debris"),
            ("commercial strip parking lot completely inundated", "water at mid-wheel height", "stalled delivery vans", "swirling murky current"),
            ("apartment complex courtyard flooded", "knee-deep water surrounding ground floor balconies", "inflatable dinghy tied to railing", "murky brown water"),
            ("industrial park main road", "submerged fire hydrants", "knee-deep floodwater covering loading bays", "truck navigating through muddy wake")
        ],
        "modifiers": [
            "water halfway up passenger car wheels", "sandbag walls leaking muddy trickles", "floating garbage bins bobbing in current", 
            "exhaust pipes submerged", "brown churned sediment water", "stalled engine smoke and hazard lights", 
            "water lapping against residential thresholds", "residents wading carefully with walking sticks", "murky urban canal overflow", 
            "oil slick sheen floating on flood surface", "overwhelmed municipal pump station", "steady knee-high current"
        ],
        "extra_keywords": [
            "knee deep flood", "wheel height water", "stalled cars", "sandbag barriers", 
            "flooded storefronts", "floating dumpsters", "canal overtopping", "bumper level flood", 
            "submerged fire hydrants", "murky brown torrent", "abandoned sedan", "inundated ground floor"
        ]
    },
    6: {
        "level_name": "Deep Inundation / Waist-Deep Water / Half-Submerged Vehicles",
        "depth_range": "60 - 100 cm (Waist-deep / Half-car Height)",
        "hazard_default": ["half_submerged_cars", "fast_current", "residential_inundation", "watercraft_rescue"],
        "action_default": "Mandatory evacuation of ground floors; dispatch search and rescue boats; shut down regional electrical grid.",
        "settings": [
            "Submerged Residential Suburb", "Downtown Commercial Core", "Flooded Riverside Settlement", 
            "Inundated Highway Underpass", "Low-lying Valley Town", "Industrial Logistics Zone", 
            "Riverside Market Square", "Apartment Block Neighborhood", "Coastal Estuary Community", "Rural Town Main Street"
        ],
        "confidence_distributions": [
            (0.95, 0.99, "Very High", "Unmistakable markers of waist-deep water submerging vehicle windows/hoods and residential living quarters."),
            (0.89, 0.94, "High", "Clear evidence of boat deployments, residents wading chest-high, and floating heavy furniture."),
            (0.83, 0.88, "Medium-High", "Deep standing and flowing floodwaters reaching window sills of residential structures.")
        ],
        "subjects": [
            ("waist-deep muddy floodwater", "cars half-submerged up to door handles", "water inundating home living rooms", "floating sofas and domestic items"),
            ("flooded neighborhood avenue", "residents wading chest-deep holding children and bags", "inflatable rescue rafts deployed", "half-sunken SUVs"),
            ("downtown business district", "water level at shop display windows", "submerged motorcycles and parking meters", "swift brown water current"),
            ("submerged residential street", "cars floating and bumping into fences", "waist-deep swirling water", "emergency rescue workers in orange vests"),
            ("low-lying valley neighborhood", "brown floodwaters reaching residential windowsills", "inflatable boat ferrying elderly residents", "floating refrigerators"),
            ("highway underpass converted into deep lake", "half-submerged delivery truck", "water level touching truck windshield base", "debris trapped in current"),
            ("riverside market square completely engulfed", "waist-deep muddy water", "floating wooden fruit crates and awnings", "rescue boat with outboard motor"),
            ("submerged suburban cul-de-sac", "residents on porch roofs waiting for boats", "waist-deep brown torrent", "floating propane tanks and debris"),
            ("apartment complex ground level", "water flooding first-floor apartments to waist height", "tenants evacuating to upper floors", "swirling brown flood"),
            ("rural town main street", "waist-deep brown river current", "partially submerged tractors and pickup trucks", "emergency rescue rope line across street")
        ],
        "modifiers": [
            "water covering car hoods and door handles", "brown turbid current carrying household furniture", "rescue workers wading in drysuits", 
            "inflatable zodiac boats patrolling streets", "residents holding pets above water level", "electrical meters submerged", 
            "floating debris collision risk", "living room furniture floating against walls", "water surging through shattered glass doors", 
            "deep brown swirling eddies", "roaring floodwater background noise", "submerged street furniture"
        ],
        "extra_keywords": [
            "waist deep flood", "half submerged cars", "chest high water", "submerged vehicle hood", 
            "floating furniture", "rescue raft deployment", "inundated living rooms", "windowsill water level", 
            "floating cars", "submerged motorcycles", "swift brown current", "residential evacuation"
        ]
    },
    7: {
        "level_name": "Severe Flood / First-Floor Submerged / Major Infrastructure Failure",
        "depth_range": "1.5 - 3.0 m (Chest-deep to First-floor Ceiling)",
        "hazard_default": ["fully_submerged_cars", "infrastructure_collapse", "power_blackout", "boat_only_access"],
        "action_default": "Declare State of Emergency; deploy National Guard; mobilize motorized rescue boats; initiate large-scale helicopter airlift.",
        "settings": [
            "Submerged Suburban District", "Industrial Waterfront Zone", "River Basin Town", 
            "Major Highway Corridor", "Islanded Urban Center", "Flooded Hospital Complex", 
            "Mountain Valley Settlement", "Coastal Storm Surge Zone", "Submerged Railway Terminal", "Bridge Approach Highway"
        ],
        "confidence_distributions": [
            (0.95, 0.99, "Very High", "Completely submerged vehicles with only roofs visible; first stories of houses fully underwater; motorboat transit."),
            (0.90, 0.94, "High", "Critical infrastructure inundation, collapsed perimeter walls, and severe deep-water navigation."),
            (0.85, 0.89, "Medium-High", "Extensive multi-meter inundation; landmarks submerged past first floor structural line.")
        ],
        "subjects": [
            ("completely submerged passenger cars with only roofs poking out", "first story of single-family homes completely underwater", "motorized rescue boat navigating street", "residents on second-floor balconies waving for help"),
            ("severely flooded highway interchange", "overturned floating delivery vans", "washed-out asphalt roadway sections", "collapsed concrete highway barrier"),
            ("inundated residential district", "water reaching eaves of single-story houses", "fire department rescue motorboats", "downed high-voltage power lines hanging in water"),
            ("submerged electrical substation", "sparking electrical transformers", "thick oil sheen on deep water", "deep floodwaters submerging control buildings"),
            ("major river crossing bridge", "turbulent brown torrent overtopping bridge deck", "partially collapsed bridge approach", "trees crashing into concrete bridge piers"),
            ("flooded hospital emergency campus", "ground floor completely submerged", "motorboats evacuating patients on stretchers", "submerged emergency backup generators"),
            ("downtown commercial center", "water depth over 2 meters", "street signs submerged to top bracket", "rescue watercraft tying up to second-floor window frames"),
            ("rural valley farming settlement", "submerged barns with only roof peaks visible", "floating livestock carcasses and farm equipment", "military rescue zodiacs"),
            ("submerged railway station", "train cars partially submerged in murky floodwater", "rail tracks two meters underwater", "collapsed retaining wall"),
            ("coastal town during hurricane surge", "two-meter storm surge engulfing homes", "collapsed brick perimeter walls", "rescue boats navigating between submerged streetlights")
        ],
        "modifiers": [
            "vehicles entirely submerged beneath water surface", "first floor completely drowned in brown mud water", "rescue boats passing over submerged fence lines", 
            "severe hydrostatic pressure cracking walls", "fuel slick rainbow sheen covering floodwaters", "snapped utility poles floating horizontally", 
            "violent hydraulic suction around submerged obstacles", "entire ground floor interiors destroyed", "second floor emergency egress via ladders", 
            "motorized boat wakes slapping upper walls", "submerged traffic lights barely breaching surface", "widespread grid power failure"
        ],
        "extra_keywords": [
            "fully submerged cars", "first floor underwater", "roof level flood", "rescue motorboat", 
            "collapsed road", "washed out bridge", "sparking substation", "downed power lines", 
            "severe inundation", "second floor evacuation", "floating vehicles", "catastrophic infrastructure failure"
        ]
    },
    8: {
        "level_name": "Catastrophic Flash Flood / Rooftop Stranded Victims / Destructive Surge",
        "depth_range": "3.0 - 5.0 m (Rooftop Inundation Level)",
        "hazard_default": ["rooftop_stranded_victims", "catastrophic_structural_damage", "vehicles_stacked", "deadly_torrent"],
        "action_default": "Immediate airborne search and rescue; deploy military helicopters for roof winching; mass casualty triage readiness.",
        "settings": [
            "Rooftop Submerged Neighborhood", "Mountain Torrent Basin", "Breached Levee District", 
            "Destroyed River Township", "Flash Flooded Valley Village", "Washed-away Highway Corridor", 
            "Islanded Residential Sector", "Catastrophic Surge Basin", "Demolished Waterfront Community", "Submerged Multi-story Core"
        ],
        "confidence_distributions": [
            (0.96, 0.99, "Very High", "Unambiguous visual evidence of victims trapped on roof ridges with single-story homes fully submerged and raging torrents."),
            (0.90, 0.95, "High", "Massive structural destruction, stacked crushed vehicles, and desperate distress signals from isolated rooftops."),
            (0.85, 0.89, "Medium-High", "Catastrophic multi-meter water depth engulfing entire neighborhoods up to roof rafters.")
        ],
        "subjects": [
            ("residents stranded on rooftop peak", "waving orange distress flag and bedsheet", "violent brown torrent swirling below", "single-story houses submerged to roof gutter lines"),
            ("catastrophic flash flood aftermath", "crushed cars stacked three high against concrete bridge pier", "massive uprooted oak trees battering homes", "trapped family clinging to chimney"),
            ("raging mountain river surge", "houses ripped off concrete foundations", "mudflow carrying corrugated iron roofs and splintered timbers", "stranded victims on second-floor roof deck"),
            ("breached levee district", "vast sea of brown water covering hundreds of houses to the roofline", "aerial drone view of isolated rooftops", "families huddled around television antenna waiting for rescue"),
            ("washed-out mountain village", "two-story homes half-collapsed in violent rapids", "survivors waving flashlights from attic vents", "massive boulder debris blocking valley"),
            ("demolished riverside avenue", "vehicles swept into crumpled heap", "houses shifted dozens of meters downriver", "terrified residents stranded on top of commercial warehouse roof"),
            ("catastrophic surge basin", "water reaching roof eaves across entire town", "military helicopter hovering low over rooftop survivors", "powerful hydraulic currents sweeping shipping containers"),
            ("inundated valley community", "entire residential blocks submerged under 4 meters of muddy water", "people huddled on galvanized tin roofs", "floating wreckage and propane tanks"),
            ("highway toll plaza destroyed", "raging brown deluge sweeping tractor trailers", "survivors clinging to steel overhead sign gantry", "deep destructive flood surge"),
            ("submerged coastal settlement", "surge waters engulfing bungalow roofs", "stranded residents signaling with flares", "violent breaking waves crashing over residential roofs")
        ],
        "modifiers": [
            "stranded families clinging to roof apex", "bright orange rescue marker painted on roof shingles", "raging hydraulic boil around houses", 
            "vehicles crumpled and tossed like toys", "houses detached from foundations floating downstream", "military helicopter winch cable descending", 
            "desperate hand gestures toward sky", "splintered wooden wreckage circulating in whirlpools", "violent roaring muddy cataract", 
            "roofs surrounded by endless muddy ocean", "exhausted survivors in drenched clothing", "extreme hydraulic force ripping walls"
        ],
        "extra_keywords": [
            "stranded on rooftop", "roof peak refuge", "waving distress flag", "houses submerged to roof", 
            "catastrophic flash flood", "cars stacked against bridge", "foundation ripped away", "helicopter hoist rescue", 
            "raging brown torrent", "trapped on chimney", "aerial roof evacuation", "destructive flood surge"
        ]
    },
    9: {
        "level_name": "Critical Disaster / Structural Collapse / Imminent Peril to Life",
        "depth_range": "> 4.0 m (Raging Torrent / Active Structural Collapse)",
        "hazard_default": ["active_structural_collapse", "violent_whirlpool", "dam_breach_wave", "life_threatening_peril"],
        "action_default": "Tactical emergency extraction; emergency air-sea rescue; urgent evacuation of downriver zones; critical survival operations.",
        "settings": [
            "Actively Collapsing Riverfront", "Breached Dam Spillway Zone", "Violent Mountain Flash Flood Gorge", 
            "Collapsing Multi-story Structure", "Raging Flood Whirlpool Basin", "Mudslide-Submerged Settlement", 
            "Industrial Toxic Flood Fire Complex", "Washed-away Evacuation Route", "Catastrophic Canyon Floodway", "Severed Bridge Collapse Site"
        ],
        "confidence_distributions": [
            (0.96, 0.99, "Very High", "Unmistakable active structural collapse into raging torrent; emergency winch rescuing victims seconds before building collapses."),
            (0.91, 0.95, "High", "Violent hydrodynamic forces visibly destroying inhabited buildings; extreme imminent peril to human life."),
            (0.86, 0.90, "Medium-High", "Catastrophic collapse dynamics in raging floodwaters; confirmed high-risk life extraction under extreme conditions.")
        ],
        "subjects": [
            ("two-story brick house actively collapsing into roaring river torrent", "corner of building shearing off into raging brown foam", "coast guard helicopter winching survivor from crumbling roof edge", "violent hydraulic vortex below"),
            ("massive dam breach surge wave", "colossal 5-meter wall of muddy water obliterating masonry buildings", "overturned swiftwater rescue boat caught in violent hydraulic roller", "trapped rescue personnel clinging to safety line"),
            ("catastrophic mudslide surging into floodwaters", "residential homes crushed like matchboxes under mud and raging water", "survivors trapped in splintered roof timbers", "screams from attic crawlspace as water rises to ceiling"),
            ("frantic attic emergency rescue", "firefighter using axe to hack hole through roof shingles", "rising muddy water lapping inside attic insulation", "trapped mother and infant being pulled through jagged roof opening"),
            ("industrial chemical storage facility flooded", "raging torrent carrying burning chemical drums", "black oily smoke rising over submerged structures", "survivors trapped on collapsing high gantry crane"),
            ("multi-span railway bridge buckling under flood torrent", "freight train plunging into raging muddy abyss", "rail workers clinging to dangling twisted steel track", "violent brown rapids roaring underneath"),
            ("flash flood surge sweeping through narrow gorge town", "entire row of houses collapsing sequentially into torrent", "people swept into raging waters struggling against debris", "emergency helicopter battling severe downdrafts"),
            ("submerged apartment complex undergoing structural failure", "external walls bowing outward under water pressure", "residents clinging to balcony railing as floor cracks", "violent swirling debris jam"),
            ("raging torrent flipping military amphibious rescue vehicle", "soldiers and evacuees thrown into fast-moving brown torrent", "emergency safety lines snapping under tension", "extreme turbulence and standing waves"),
            ("landslide and flash flood convergence", "massive surge of liquid mud and uprooted pines smashing into school building", "terrified survivors on collapsing roof ridge", "raging brown deluge tearing structure apart")
        ],
        "modifiers": [
            "masonry walls disintegrating into brown foam", "building frame groaning and leaning into torrent", "helicopter rotor downwash spraying violent mist", 
            "axe marks through roof shingles in race against rising water", "violent hydraulic standing waves and whirlpools", "screaming victims in immediate life peril", 
            "splintering load-bearing timber columns", "black smoke mixing with brown torrential spray", "twisted steel rebar snapping under force", 
            "extreme hydrostatic implosion of walls", "overturning rescue vessel in hydraulic boil", "imminent total collapse of refuge structure"
        ],
        "extra_keywords": [
            "house actively collapsing", "building shearing into river", "helicopter roof winch", "violent hydraulic vortex", 
            "dam breach wall of water", "axe breaking attic roof", "trapped in rising attic", "overturned rescue boat", 
            "imminent structural failure", "raging flood rapids", "crushed by debris flow", "life threatening flood peril"
        ]
    },
    10: {
        "level_name": "Extreme Disaster / Fatalities / Dead Bodies Lying in Water",
        "depth_range": "Variable Catastrophic Inundation (Fatal Casualties Confirmed)",
        "hazard_default": ["human_fatalities", "dead_bodies_in_water", "mass_casualty", "corpse_recovery", "biohazard"],
        "action_default": "Mobilize Disaster Victim Identification (DVI) teams, coroner recovery watercraft, forensic mortuary units, and biohazard containment.",
        "settings": [
            "Receding Flood Mud Basin", "Submerged Urban Street Corner", "Debris Jam at Bridge Pier", 
            "Inundated Residential Culvert", "Flooded Neighborhood Alley", "Submerged Vehicle Graveyard", 
            "Tangled Riverbank Thicket", "Post-Flash Flood Mudflat", "Submerged Commercial District", "Canal Outfall Debris Barrier"
        ],
        "confidence_distributions": [
            (0.96, 0.99, "Very High", "Unambiguous visual confirmation of deceased human victims / dead body floating face-down or resting in floodwaters; active coroner recovery."),
            (0.91, 0.95, "High", "Clear documentation of fatal casualties, body retrieval by disaster teams in watercraft, and draped corpses in flooded terrain."),
            (0.86, 0.90, "Medium-High", "Documented flood fatalities trapped in wreckage or submerged vegetation; verified recovery scene.")
        ],
        "subjects": [
            ("dead body floating face-down in stagnant brown floodwater", "deceased human victim in waterlogged clothes", "emergency coroner team approaching in inflatable dinghy", "yellow body recovery bag on rescue deck"),
            ("deceased flood victim trapped in submerged tree branches", "lifeless body suspended in murky water current", "search and rescue divers preparing recovery harness", "catastrophic flood casualty recovery"),
            ("lifeless human corpse lying face-up on submerged debris pile", "drowned flood victim resting in receding muddy water", "forensic recovery personnel in white protective suits", "tragic flood fatality aftermath"),
            ("bloated dead body floating near submerged car roof", "deceased victim entangled in floating debris and power cables", "police boat marking casualty location with buoy", "mass casualty flood disaster scene"),
            ("dead body lying partially submerged in thick flood mud", "drowned victim washed ashore on debris-strewn residential road", "coroner team placing white shroud over deceased", "devastating flood death"),
            ("multiple dead bodies visible in flood debris basin", "deceased casualties floating among collapsed building wreckage", "military search and recovery watercraft deploying body bags", "extreme disaster mortality"),
            ("drowned corpse pinned against bridge concrete pier", "lifeless body tangled in floating driftwood and twisted sheet metal", "recovery boat retrieving deceased victim from river current", "fatal flash flood outcome"),
            ("unresponsive deceased person floating in flooded street intersection", "dead body lying near submerged storefront entrance", "emergency responders securing drowned victim with recovery rope", "deadly flood inundation"),
            ("deceased child corpse retrieved from murky floodwater", "lifeless victim gently placed in emergency rescue boat", "rescue personnel bowing heads in tragedy", "extreme fatal flood emergency"),
            ("dead bodies washed up against tangled fence line in receding flood", "lifeless victims partially submerged in muddy runoff", "disaster victim identification unit documenting scene", "mass casualty drowning site")
        ],
        "modifiers": [
            "lifeless human body floating motionless", "waterlogged clothing clinging to deceased victim", "forensic recovery team securing corpse", 
            "yellow zippered disaster body bag deployed", "white sheet draped over retrieved corpse in boat", "tragic loss of life in catastrophic flood", 
            "bloated deceased body in stagnant brown water", "corpse entangled in submerged barbed wire and branches", "grim recovery mission by forensic divers", 
            "biohazard protocols in contaminated floodwaters", "floating corpse marked with emergency floating beacon", "somber body retrieval operation"
        ],
        "extra_keywords": [
            "dead body floating in water", "deceased flood victim", "drowned corpse", "lifeless body lying on debris", 
            "floating dead body", "body bag recovery", "coroner boat retrieving corpse", "mass casualty flood", 
            "fatal drowning disaster", "bloated corpse in floodwater", "dead bodies in receding mud", "disaster victim recovery"
        ]
    }
}

def generate_entry(entry_id, severity_score):
    config = SEVERITY_CONFIG[severity_score]
    
    # Select subject and setting
    subject_tuple = random.choice(config["subjects"])
    setting = random.choice(config["settings"])
    modifier = random.choice(config["modifiers"])
    
    # Confidence selection
    conf_choice = random.choice(config["confidence_distributions"])
    min_c, max_c, conf_level, conf_rationale = conf_choice
    conf_score = round(random.uniform(min_c, max_c), 3)
    
    # Pick keywords: 3 from subject, 1-2 from modifiers, 2-3 from extra_keywords
    primary_kws = list(subject_tuple)
    random.shuffle(primary_kws)
    selected_primary = primary_kws[:3]
    
    selected_extras = random.sample(config["extra_keywords"], k=random.randint(2, 4))
    selected_modifier_kw = [modifier.split(";")[0].strip().lower()]
    
    all_kws = list(dict.fromkeys(selected_primary + selected_extras + selected_modifier_kw))
    random.shuffle(all_kws)
    keywords_str = ", ".join(all_kws[:8])
    
    # Build vivid natural caption
    core_subj = subject_tuple[0]
    sec_subj = subject_tuple[1]
    tert_subj = subject_tuple[2]
    
    caption_templates = [
        f"A clear photograph showing {core_subj} in a {setting.lower()}, where {sec_subj} and {tert_subj} are observed under {modifier}.",
        f"Overhead view of a {setting.lower()} depicting {core_subj}, accompanied by {sec_subj} alongside {tert_subj} with {modifier}.",
        f"Ground-level capture in a {setting.lower()} revealing {core_subj}; {sec_subj} is visible near {tert_subj} amidst {modifier}.",
        f"Documentary-style image from a {setting.lower()} capturing {core_subj} as {sec_subj} develops around {tert_subj} in {modifier}.",
        f"Emergency surveillance view of a {setting.lower()} illustrating {core_subj}, with {sec_subj} and {tert_subj} documented under {modifier}."
    ]
    caption = random.choice(caption_templates)
    
    # Hazard indicators
    hazard_list = list(config["hazard_default"])
    if severity_score >= 8:
        hazard_list.append("extreme_danger")
    if severity_score == 10:
        hazard_list.extend(["fatalities", "biohazard_recovery"])
    hazard_str = "; ".join(hazard_list)
    
    return {
        "id": f"FLD_{entry_id:04d}",
        "severity_score": severity_score,
        "severity_level": config["level_name"],
        "confidence_level": conf_level,
        "confidence_score": conf_score,
        "confidence_rationale": conf_rationale,
        "keywords": keywords_str,
        "water_depth_estimate": config["depth_range"],
        "visual_context_setting": setting,
        "hazard_indicators": hazard_str,
        "recommended_action": config["action_default"],
        "image_caption": caption
    }

def main():
    entries_per_severity = 250
    total_entries = entries_per_severity * 10  # 2500 entries
    
    dataset = []
    current_id = 1
    
    # Generate balanced entries for severity 1 through 10
    for severity in range(1, 11):
        for _ in range(entries_per_severity):
            entry = generate_entry(current_id, severity)
            dataset.append(entry)
            current_id += 1
            
    # Randomly shuffle dataset order to simulate real-world natural distribution in final files
    # while preserving deterministic results
    shuffled_dataset = list(dataset)
    random.shuffle(shuffled_dataset)
    
    # Re-assign sequential IDs after shuffle or keep ordered?
    # Keeping clean IDs FLD_0001 to FLD_2500 after shuffle gives a completely realistic, mixed dataset!
    for idx, item in enumerate(shuffled_dataset, start=1):
        item["id"] = f"FLD_{idx:04d}"
        
    csv_file_path = "/Users/harnimarpreetsingh/Desktop/Flood/flood_severity_dataset.csv"
    json_file_path = "/Users/harnimarpreetsingh/Desktop/Flood/flood_severity_dataset.json"
    
    fieldnames = [
        "id", "severity_score", "severity_level", "confidence_level", 
        "confidence_score", "confidence_rationale", "keywords", 
        "water_depth_estimate", "visual_context_setting", 
        "hazard_indicators", "recommended_action", "image_caption"
    ]
    
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shuffled_dataset)
        
    with open(json_file_path, mode="w", encoding="utf-8") as f:
        json.dump(shuffled_dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated {len(shuffled_dataset)} entries.")
    print(f"CSV saved to: {csv_file_path}")
    print(f"JSON saved to: {json_file_path}")

if __name__ == "__main__":
    main()
