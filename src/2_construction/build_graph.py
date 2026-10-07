"""
Knowledge Graph Construction Pipeline for Vienna Real Estate and Transit Data.
This script reads transit stop data (GTFS) and flat information (JSON), 
computes spatial and temporal relationships, and builds an RDF Knowledge Graph.
"""

import csv
import json
import math
import time
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD

# ==============================================================================
# Initialization & Setup
# ==============================================================================

# Start global time tracking to measure the execution time of the entire pipeline
script_start_time = time.time()
print(
    f"[{time.strftime('%H:%M:%S')}] Starting Knowledge Graph construction"
    " pipeline...\n"
)

# 1. Define namespaces
# Create a custom namespace for the Vienna Real Estate Knowledge Graph entities
EX = Namespace("http://vienna-realestate.org/kg/")


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two GPS points in meters using the Haversine formula.
    This formula accounts for the spherical shape of the Earth.

    Args:
        lat1 (float): Latitude of the first point in decimal degrees.
        lon1 (float): Longitude of the first point in decimal degrees.
        lat2 (float): Latitude of the second point in decimal degrees.
        lon2 (float): Longitude of the second point in decimal degrees.

    Returns:
        float: The distance between the two geographical points in meters.
    """
    R = 6371000  # Earth's mean radius in meters
    
    # Convert latitude and longitude from degrees to radians for math functions
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula application
    # 'a' represents the square of half the chord length between the points
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    # 'c' represents the angular distance in radians
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def clean_uri_string(val):
    """
    Sanitizes strings to create valid RDF URIs. 
    It replaces or removes characters that are illegal or problematic in URIs.

    Args:
        val (any): The input value (usually ID or string) to be sanitized.

    Returns:
        str: A sanitized string safe for appending to a Namespace.
    """
    return (
        str(val)
        .replace(":", "_")
        .replace(" ", "_")
        .replace('"', "")
        .replace("'", "")
    )


# 2. Initialize RDF Graph
# Create the main graph object and bind common namespaces for readable prefixes in Turtle output
g = Graph()
g.bind("ex", EX)
g.bind("rdfs", RDFS)
g.bind("xsd", XSD)


# ==============================================================================
# Step 3: Load Transit Stops
# ==============================================================================
step_start = time.time()
stops_path = "assets/data/wienerlinien/stops.txt"
print(f"[{time.strftime('%H:%M:%S')}] Loading transit stops from {stops_path}...")

# List to keep track of loaded stops for later spatial joins (e.g., connecting flats)
stops = []

with open(stops_path, mode="r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    
    # Clean up column headers in case they contain unwanted quotation marks or whitespaces
    if reader.fieldnames:
        reader.fieldnames = [
            name.replace('"', "").strip() for name in reader.fieldnames
        ]

    for row in reader:
        # Extract and sanitize stop ID
        raw_id = row["stop_id"].replace('"', "").strip()
        stop_id_clean = clean_uri_string(raw_id)
        
        # Extract stop name
        stop_name = row["stop_name"].replace('"', "").strip()

        # Extract coordinates and convert them to floats
        lat = float(row["stop_lat"].replace('"', "").strip())
        lon = float(row["stop_lon"].replace('"', "").strip())

        # Construct the unique RDF URI for the transit stop
        stop_uri = EX[f"stop_{stop_id_clean}"]

        # Add base stop triples to the RDF graph (type, label, and coordinates)
        g.add((stop_uri, RDF.type, EX.TransitStop))
        g.add((stop_uri, RDFS.label, Literal(stop_name, datatype=XSD.string)))
        g.add((stop_uri, EX.latitude, Literal(lat, datatype=XSD.float)))
        g.add((stop_uri, EX.longitude, Literal(lon, datatype=XSD.float)))

        # Store dictionary representation for fast spatial iteration later
        stops.append({"uri": stop_uri, "lat": lat, "lon": lon})

step_duration = time.time() - step_start
print(
    f"[{time.strftime('%H:%M:%S')}] Loaded {len(stops)} transit stops into"
    f" the KG in {step_duration:.2f}s.\n"
)


# ==============================================================================
# Step 4: Connect transit stops using Reification
# ==============================================================================
# Reification is used here to treat a "Connection" between two stops as its own 
# node. This allows us to attach properties to the edge itself, such as the 
# average travel time between the two stops.
step_start = time.time()
stop_times_path = "assets/data/wienerlinien/stop_times.txt"
print(
    f"[{time.strftime('%H:%M:%S')}] Connecting stops with Reification using"
    f" {stop_times_path}..."
)
connections_count = 0


def parse_time_to_minutes(time_str):
    """
    Converts a time string formatted as HH:MM:SS to total elapsed minutes from midnight.
    This simplifies mathematical duration calculations.
    """
    parts = time_str.replace('"', "").strip().split(":")
    return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60.0


try:
    # Dictionary to collect all travel times between two connected stops.
    # Structure: {(from_stop_uri, to_stop_uri): [duration_1, duration_2, ...]}
    edge_travel_times = {}

    with open(stop_times_path, mode="r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames:
            reader.fieldnames = [
                name.replace('"', "").strip() for name in reader.fieldnames
            ]

        # Track previous row values to calculate duration for consecutive stops on the same trip
        previous_trip_id = None
        previous_stop_uri = None
        previous_dep_time = None

        for row in reader:
            trip_id = row["trip_id"].replace('"', "").strip()
            raw_stop_id = row["stop_id"].replace('"', "").strip()
            arr_time_str = row["arrival_time"].replace('"', "").strip()
            dep_time_str = row["departure_time"].replace('"', "").strip()

            current_stop_uri = EX[f"stop_{clean_uri_string(raw_stop_id)}"]

            try:
                arr_minutes = parse_time_to_minutes(arr_time_str)
                dep_minutes = parse_time_to_minutes(dep_time_str)
            except (ValueError, IndexError):
                # Skip malformed time entries
                continue

            # Filter for relevant daytime operational hours: 07:00 (420 min) to 22:00 (1320 min)
            if 420 <= dep_minutes <= 1320:
                # Ensure the current stop and previous stop belong to the exact same transit trip
                if (
                    trip_id == previous_trip_id
                    and previous_stop_uri
                    and previous_dep_time is not None
                ):
                    # Travel duration = arrival time at current stop - departure time from previous stop
                    duration = arr_minutes - previous_dep_time
                    
                    # Sanity check: Travel time between consecutive stops should be between 0 and 30 minutes
                    if 0 < duration < 30:  
                        edge_key = (previous_stop_uri, current_stop_uri)
                        
                        # Initialize list if this specific edge connection is encountered for the first time
                        if edge_key not in edge_travel_times:
                            edge_travel_times[edge_key] = []
                            
                        edge_travel_times[edge_key].append(duration)

            # Update tracking variables for the next iteration
            previous_trip_id = trip_id
            previous_stop_uri = current_stop_uri
            previous_dep_time = dep_minutes

    # Iterate over the aggregated travel times and add Reified Connections to the RDF Graph
    for (from_stop, to_stop), durations in edge_travel_times.items():
        # Calculate average duration across all recorded trips for this edge
        avg_duration = round(sum(durations) / len(durations), 2)

        # Extract base IDs to create a clean, unique URI for the reified connection node
        from_id = str(from_stop).replace(str(EX), "")
        to_id = str(to_stop).replace(str(EX), "")
        conn_uri = EX[f"conn_{from_id}_to_{to_id}"]

        # 1. Attach connection node to the source stop
        g.add((from_stop, EX.hasConnection, conn_uri))
        g.add((conn_uri, RDF.type, EX.TransitConnection))

        # 2. Attach the target stop and the computed average travel time to the connection node
        g.add((conn_uri, EX.targetStop, to_stop))
        g.add(
            (
                conn_uri,
                EX.avgTravelTimeMinutes,
                Literal(avg_duration, datatype=XSD.float),
            )
        )

        # 3. Add a direct (non-reified) connectsTo edge
        # This is kept for simpler pathfinder algorithms or graph embeddings (like PyKEEN) 
        # that struggle with or don't need reified edges.
        g.add((from_stop, EX.connectsTo, to_stop))

        connections_count += 1

    step_duration = time.time() - step_start
    print(
        f"[{time.strftime('%H:%M:%S')}] Created {connections_count} reified"
        f" transit connections in {step_duration:.2f}s.\n"
    )

except FileNotFoundError:
    print(
        f"[{time.strftime('%H:%M:%S')}] Warning: {stop_times_path} not found."
        " Skipping transit stop connections.\n"
    )


# ==============================================================================
# Step 4b: Generate Synthetic Transfer / Footpath Edges (Stops <= 100m apart)
# ==============================================================================
# This step creates "walking transfers" between transit stops that are physically 
# very close to each other, allowing reasoning engines to switch between lines.
step_start = time.time()
print(
    f"[{time.strftime('%H:%M:%S')}] Generating synthetic footpaths/transfers for"
    " nearby stops (<= 100m)..."
)

# 1. Extract all transit stops and their coordinates directly from the current RDF Graph
stop_coords = {}
for stop_uri in g.subjects(RDF.type, EX.TransitStop):
    lats = list(g.objects(stop_uri, EX.latitude))
    lons = list(g.objects(stop_uri, EX.longitude))

    # Ensure the stop has both latitude and longitude before adding to dictionary
    if lats and lons:
        try:
            stop_coords[stop_uri] = (float(lats[0]), float(lons[0]))
        except ValueError:
            continue

stop_uris = list(stop_coords.keys())
num_stops = len(stop_uris)
transfer_count = 0

# 2. Compare stop pairs and create bidirectional transfer connections
WALK_TIME_MINUTES = 2.0   # Assumed constant walking time for nearby transfers
MAX_DISTANCE_METERS = 100.0

# O(N^2) comparison loop to check distance between every stop pair
for i in range(num_stops):
    stop1 = stop_uris[i]
    lat1, lon1 = stop_coords[stop1]

    # Only check pairs ahead in the list to avoid duplicate checks (j > i)
    for j in range(i + 1, num_stops):
        stop2 = stop_uris[j]
        lat2, lon2 = stop_coords[stop2]

        # Optimization: Quick bounding box pre-filter.
        # This skips the computationally expensive Haversine distance calculation 
        # if the coordinate differences are obviously too large (> ~200-300m).
        if abs(lat1 - lat2) > 0.002 or abs(lon1 - lon2) > 0.003:
            continue

        # Calculate accurate real-world distance using Haversine
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        
        # If stops are within 100 meters, generate the transfer edges
        if 0.0 < dist <= MAX_DISTANCE_METERS:
            stop1_id = str(stop1).replace(str(EX), "")
            stop2_id = str(stop2).replace(str(EX), "")

            # Create connection for Direction: Stop A -> Stop B
            conn_ab = EX[f"conn_transfer_{stop1_id}_to_{stop2_id}"]
            g.add((stop1, EX.hasConnection, conn_ab))
            g.add((conn_ab, RDF.type, EX.TransitConnection))
            g.add((conn_ab, EX.targetStop, stop2))
            g.add((
                conn_ab,
                EX.avgTravelTimeMinutes,
                Literal(WALK_TIME_MINUTES, datatype=XSD.float),
            ))

            # Create connection for Direction: Stop B -> Stop A
            conn_ba = EX[f"conn_transfer_{stop2_id}_to_{stop1_id}"]
            g.add((stop2, EX.hasConnection, conn_ba))
            g.add((conn_ba, RDF.type, EX.TransitConnection))
            g.add((conn_ba, EX.targetStop, stop1))
            g.add((
                conn_ba,
                EX.avgTravelTimeMinutes,
                Literal(WALK_TIME_MINUTES, datatype=XSD.float),
            ))

            # Increment by 2 since we added bidirectional edges
            transfer_count += 2

step_duration = time.time() - step_start
print(
    f"[{time.strftime('%H:%M:%S')}] Added {transfer_count} synthetic transfer"
    f" connections in {step_duration:.2f}s.\n"
)


# ==============================================================================
# Step 5: Load Flats and spatially link them to nearby transit stops
# ==============================================================================
step_start = time.time()
flats_path = "assets/data/flat_info.json"
print(f"[{time.strftime('%H:%M:%S')}] Loading flats from {flats_path}...")

with open(flats_path, mode="r", encoding="utf-8") as file:
    flats = json.load(file)

MAX_RADIUS_METERS = 500.0  # Threshold to define "nearby" transit stops
count_flats = 0

for idx, flat in enumerate(flats):
    # Extract ID. Fallback to generating a sequential ID if none is found.
    raw_id = flat.get("id") or flat.get("ID") or f"gen_{idx}"
    flat_id_clean = clean_uri_string(raw_id)
    flat_uri = EX[f"flat_{flat_id_clean}"]

    lat = flat.get("LATITUDE")
    lon = flat.get("LONGITUDE")
    
    # Skip flats that don't have geospatial coordinates
    if lat is None or lon is None:
        continue

    # Add base flat triples (Type and Coordinates)
    g.add((flat_uri, RDF.type, EX.Flat))
    g.add((flat_uri, EX.latitude, Literal(float(lat), datatype=XSD.float)))
    g.add((flat_uri, EX.longitude, Literal(float(lon), datatype=XSD.float)))

    # Extract and add a human-readable title (heading)
    heading = flat.get("HEADING") or flat.get("heading")
    if heading:
        g.add((flat_uri, RDFS.label, Literal(str(heading), datatype=XSD.string)))

    # Store the original Willhaben ID for external referencing
    g.add((
        flat_uri,
        EX.willhabenId,
        Literal(str(raw_id), datatype=XSD.string),
    ))

    # Add 'PRICE' property if available and valid
    if "PRICE" in flat and flat["PRICE"]:
        try:
            price = float(flat["PRICE"])
            g.add((flat_uri, EX.hasPrice, Literal(price, datatype=XSD.float)))
        except ValueError:
            pass

    # Add 'SIZE' property (Checking multiple possible JSON keys)
    size_val = flat.get("ESTATE_SIZE/LIVING_AREA") or flat.get("ESTATE_SIZE")
    if size_val:
        try:
            size = float(size_val)
            g.add((flat_uri, EX.hasSize, Literal(size, datatype=XSD.float)))
        except ValueError:
            pass

    # Add 'ROOMS' property
    rooms_val = flat.get("NUMBER_OF_ROOMS") or flat.get("ROOMS")
    if rooms_val:
        try:
            rooms = int(float(rooms_val))
            g.add((flat_uri, EX.hasRooms, Literal(rooms, datatype=XSD.integer)))
        except ValueError:
            pass

    # Add 'POSTCODE' property
    if "POSTCODE" in flat and flat["POSTCODE"]:
        g.add((
            flat_uri,
            EX.postcode,
            Literal(str(flat["POSTCODE"]), datatype=XSD.string),
        )) 

    # Link the flat directly to ALL transit stops within the defined radius (<= 500m)
    connected_stops_count = 0

    for stop in stops:
        # Calculate distance between the flat and the current stop
        dist = haversine_distance(float(lat), float(lon), stop["lat"], stop["lon"])

        if dist <= MAX_RADIUS_METERS:
            g.add((flat_uri, EX.isNearStop, stop["uri"]))
            connected_stops_count += 1

    # Fallback Mechanism: If no stops were found within the 500m radius, 
    # force a connection to the single absolute closest stop to ensure graph connectivity.
    if connected_stops_count == 0 and stops:
        closest_stop = min(
            stops,
            key=lambda s: haversine_distance(
                float(lat), float(lon), s["lat"], s["lon"]
            ),
        )
        g.add((flat_uri, EX.isNearStop, closest_stop["uri"]))

    count_flats += 1

step_duration = time.time() - step_start
print(
    f"[{time.strftime('%H:%M:%S')}] Processed {count_flats} flats and linked"
    f" them spatially in {step_duration:.2f}s.\n"
)


# ==============================================================================
# Step 6: Serialization
# ==============================================================================
step_start = time.time()
output_ttl = "vienna_kg.ttl"
print(
    f"[{time.strftime('%H:%M:%S')}] Serializing Knowledge Graph to"
    f" {output_ttl}..."
)

# Export the constructed RDF Graph into Turtle format (.ttl)
# Turtle is a standard, human-readable syntax for RDF
g.serialize(destination=output_ttl, format="turtle")

step_duration = time.time() - step_start
total_duration = time.time() - script_start_time

print(
    f"[{time.strftime('%H:%M:%S')}] Knowledge Graph successfully exported to"
    f" '{output_ttl}' with {len(g)} total triples (Serialization time:"
    f" {step_duration:.2f}s)."
)
print(
    f"[{time.strftime('%H:%M:%S')}] Entire pipeline finished in"
    f" {total_duration:.2f}s ({total_duration/60:.2f} minutes)!"
)