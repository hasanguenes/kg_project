"""
Refined Logical Reasoning Pipeline for the Vienna Real Estate Knowledge Graph.
This script loads a base Knowledge Graph, builds a directed transit network using NetworkX, 
and infers new knowledge (travel times, classifications, and proximity edges) based on 
a set of predefined logical rules.
"""

import math
import os
import time
import networkx as nx
from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace

# ==============================================================================
# Initialization & Dynamic Path Resolution
# ==============================================================================
# Resolving absolute paths dynamically ensures the script runs correctly 
# regardless of the current working directory it is executed from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

INPUT_TTL_PATH = os.path.join(
    PROJECT_ROOT, "src", "2_construction", "vienna_kg.ttl"
)
OUTPUT_TTL_PATH = os.path.join(
    PROJECT_ROOT, "src", "3_logic", "inferred_vienna_kg.ttl"
)


def haversine_distance_meters(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two coordinate pairs in meters.
    This is used for high-precision spatial proximity checks (e.g., finding alternative flats).

    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        float: Distance in meters.
    """
    r = 6371000.0  # Earth radius in meters
    
    # Convert degrees to radians for trigonometric functions
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Apply Haversine formula
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


script_start_time = time.time()
print(
    f"[{time.strftime('%H:%M:%S')}] Starting Refined Logical Reasoning"
    " Pipeline..."
)

EX = Namespace("http://vienna-realestate.org/kg/")

# ==============================================================================
# Load Base Knowledge Graph
# ==============================================================================
print(
    f"[{time.strftime('%H:%M:%S')}] Loading Knowledge Graph from"
    f" {INPUT_TTL_PATH}..."
)
g = Graph()
g.parse(INPUT_TTL_PATH, format="turtle")
g.bind("ex", EX)
g.bind("rdfs", RDFS)
g.bind("xsd", XSD)

print(
    f"[{time.strftime('%H:%M:%S')}] Base Graph loaded with {len(g)} triples."
)

# ==============================================================================
# Build NetworkX Graph for Efficient Transit Pathfinding
# ==============================================================================
# We convert RDF transit connections into a NetworkX Directed Graph.
# NetworkX provides highly optimized algorithms (like Dijkstra) for finding 
# the shortest travel times, which is faster than native SPARQL property paths.
print(
    f"[{time.strftime('%H:%M:%S')}] Building NetworkX directed transit"
    " graph..."
)
G = nx.DiGraph()

# Extract all Reified Transit Connections to build the directed edges
for conn_node in g.subjects(RDF.type, EX.TransitConnection):
    from_stops = list(g.subjects(EX.hasConnection, conn_node))
    to_stops = list(g.objects(conn_node, EX.targetStop))
    durations = list(g.objects(conn_node, EX.avgTravelTimeMinutes))

    # If the connection is valid and has a duration, add it as a weighted edge
    if from_stops and to_stops and durations:
        G.add_edge(str(from_stops[0]), str(to_stops[0]), weight=float(durations[0]))

# Define key semantic hubs in Vienna to calculate travel times towards
HUBS = {
    "center": {
        "label": "Stephansplatz",
        "prop": EX.timeToCenterMinutes,
        "stops": [],
    },
    "main_station": {
        "label": "Hauptbahnhof",
        "prop": EX.timeToMainStationMinutes,
        "stops": [],
    },
    "education_hub": {
        "label": "Karlsplatz",
        "prop": EX.timeToEducationHubMinutes,
        "stops": [],
    },
}

# Scan the graph to find the exact URIs for the defined hub stations based on their string labels
for stop_uri, label in g.subject_objects(RDFS.label):
    label_str = str(label)
    for hub_key, hub_info in HUBS.items():
        if hub_info["label"].lower() in label_str.lower():
            hub_info["stops"].append(str(stop_uri))

# Retrieve all flat entities from the graph to apply the reasoning rules
flats = list(g.subjects(RDF.type, EX.Flat))
print(
    f"[{time.strftime('%H:%M:%S')}] Executing rules across {len(flats)}"
    " flats..."
)

# Tracking counters for the rule executions
rule1_count = 0
rule2_count = 0
rule3_count = 0
rule4_count = 0
rule5a_count = 0
rule5b_count = 0

flat_coords = {}

# ==============================================================================
# Rule Execution Engine (Rules 1 - 4 & Rule 5a)
# ==============================================================================
for flat_uri in flats:
    # Get all transit stops geographically near the current flat
    near_stops = list(g.objects(flat_uri, EX.isNearStop))
    travel_times = {}

    # Store flat coordinates in a dictionary for Rule 5b (Proximity checks later)
    lats = list(g.objects(flat_uri, EX.latitude))
    lons = list(g.objects(flat_uri, EX.longitude))
    if lats and lons:
        flat_coords[flat_uri] = (float(lats[0]), float(lons[0]))

    # --- RULE 1: Travel Time Inference ---
    # Calculates the shortest public transport travel time from the flat to predefined hubs
    if near_stops:
        near_stops_str = [str(s) for s in near_stops]
        for hub_key, hub_info in HUBS.items():
            min_travel_time = float("inf")
            target_stops = hub_info["stops"]

            # Evaluate every nearby stop as a potential starting point for the journey
            for start_stop in near_stops_str:
                if start_stop not in G:
                    continue
                
                # Check route against all platform variations of the target hub
                for target_stop in target_stops:
                    if target_stop not in G:
                        continue

                    # If a path exists, calculate the weighted shortest path (Dijkstra)
                    if nx.has_path(G, start_stop, target_stop):
                        path_time = nx.shortest_path_length(
                            G, start_stop, target_stop, weight="weight"
                        )
                        # Keep track of the absolute fastest route found
                        if path_time < min_travel_time:
                            min_travel_time = path_time

            # If a valid route was found, add the inferred travel time literal to the KG
            if min_travel_time < float("inf"):
                rounded_time = round(min_travel_time, 2)
                g.add((
                    flat_uri,
                    hub_info["prop"],
                    Literal(rounded_time, datatype=XSD.float),
                ))
                travel_times[hub_key] = rounded_time
                rule1_count += 1

    # --- RULE 2: CentralFlat Classification (Strict <= 8.0 min) ---
    # Classifies a flat semantically as "CentralFlat" if it is very close to Stephansplatz
    time_to_center = travel_times.get("center")
    if time_to_center is not None and time_to_center <= 8.0:
        g.add((flat_uri, RDF.type, EX.CentralFlat))
        rule2_count += 1

    # --- RULE 3: CommuterFriendlyFlat Classification (<= 12.0 min to Main Station) ---
    # Classifies a flat as "CommuterFriendlyFlat" if Hauptbahnhof is easily accessible
    time_to_main = travel_times.get("main_station")
    if time_to_main is not None and time_to_main <= 12.0:
        g.add((flat_uri, RDF.type, EX.CommuterFriendlyFlat))
        rule3_count += 1

    # --- RULE 4: WellConnectedFlat (>= 4 DISTINCT station names) ---
    # A flat is considered well-connected if it is near many *different* transit stations.
    # We strip platform/direction markers (e.g., "Steig") to ensure we count unique station names.
    distinct_stop_names = set()
    for stop_uri in near_stops:
        labels = list(g.objects(stop_uri, RDFS.label))
        if labels:
            # Clean string to unify platforms belonging to the same physical station
            clean_name = (
                str(labels[0]).split("(")[0].split(" Steig")[0].strip().lower()
            )
            distinct_stop_names.add(clean_name)

    if len(distinct_stop_names) >= 4:
        g.add((flat_uri, RDF.type, EX.WellConnectedFlat))
        rule4_count += 1

    # --- RULE 5a: Direct Station Access Edge (hasDirectStationAccess <= 150m) ---
    # Upgrades a standard "isNearStop" relation (up to 500m) to a stronger semantic edge 
    # if the stop is practically right outside the door (<= 150m).
    if lats and lons:
        flat_lat, flat_lon = float(lats[0]), float(lons[0])
        for stop_uri in near_stops:
            s_lats = list(g.objects(stop_uri, EX.latitude))
            s_lons = list(g.objects(stop_uri, EX.longitude))
            if s_lats and s_lons:
                stop_lat, stop_lon = float(s_lats[0]), float(s_lons[0])
                if haversine_distance_meters(flat_lat, flat_lon, stop_lat, stop_lon) <= 150.0:
                    g.add((flat_uri, EX.hasDirectStationAccess, stop_uri))
                    rule5a_count += 1


# ==============================================================================
# Rule 5b: Proximity Alternative Edge (hasAlternativeFlatInProximity <= 300m)
# ==============================================================================
# This creates semantic links between flats that are in immediate walking distance 
# to each other. Useful for recommender systems ("If you like this flat, here is a nearby alternative").
flat_uris = list(flat_coords.keys())
num_flats = len(flat_uris)

for i in range(num_flats):
    f1 = flat_uris[i]
    lat1, lon1 = flat_coords[f1]

    for j in range(i + 1, num_flats):
        f2 = flat_uris[j]
        lat2, lon2 = flat_coords[f2]

        if haversine_distance_meters(lat1, lon1, lat2, lon2) <= 300.0:
            # Add bidirectional edges
            g.add((f1, EX.hasAlternativeFlatInProximity, f2))
            g.add((f2, EX.hasAlternativeFlatInProximity, f1))
            rule5b_count += 2

# ==============================================================================
# Summary and Serialization
# ==============================================================================
print(f"[{time.strftime('%H:%M:%S')}] Reasoning Execution Summary:")
print(f"  - Rule 1 (Travel Times Inferred): {rule1_count} literals")
print(f"  - Rule 2 (CentralFlat <= 8 min): {rule2_count} flats")
print(f"  - Rule 3 (CommuterFriendlyFlat <= 12 min): {rule3_count} flats")
print(
    "  - Rule 4 (WellConnectedFlat >= 4 distinct stations):"
    f" {rule4_count} flats"
)
print(
    "  - Rule 5a (hasDirectStationAccess <= 150m):"
    f" {rule5a_count} edges created"
)
print(
    "  - Rule 5b (hasAlternativeFlatInProximity <= 300m):"
    f" {rule5b_count} edges created"
)

# Serialize Enriched Knowledge Graph
# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_TTL_PATH), exist_ok=True)

# Export the expanded RDF graph to Turtle format
g.serialize(destination=OUTPUT_TTL_PATH, format="turtle")

total_duration = time.time() - script_start_time
print(
    f"[{time.strftime('%H:%M:%S')}] Success! Knowledge Graph enriched and"
    f" saved to '{OUTPUT_TTL_PATH}' ({len(g)} total triples)."
)