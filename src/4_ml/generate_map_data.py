import os
import glob
import gzip
import json
import joblib
import torch
import numpy as np
from rdflib import Graph, Namespace, RDF

# --- Path Definitions ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
KG_PATH = os.path.join(PROJECT_ROOT, "src", "3_logic", "inferred_vienna_kg.ttl")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "src", "map_data.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

EX = Namespace("http://vienna-realestate.org/kg/")

# Locate the latest run directory
valid_dirs = [d for d in glob.glob(os.path.join(RESULTS_DIR, "run_*")) 
              if os.path.exists(os.path.join(d, "price_regressor.pkl"))]

if not valid_dirs:
    raise FileNotFoundError("No valid run with price_regressor.pkl found in results!")

LATEST_RUN_DIR = max(valid_dirs, key=os.path.getctime)


MANUAL_RUN_NAME = "run_20261007_070839_RotatE_dim128"
LATEST_RUN_DIR = os.path.join(RESULTS_DIR, MANUAL_RUN_NAME)
print(f"Loading saved regressor model from: {os.path.basename(LATEST_RUN_DIR)}")

regressor = joblib.load(os.path.join(LATEST_RUN_DIR, "price_regressor.pkl"))

# Load Knowledge Graph & PyKEEN Model
print("Loading Knowledge Graph...")
g = Graph()
g.parse(KG_PATH, format="turtle")

model = torch.load(os.path.join(LATEST_RUN_DIR, "trained_model.pkl"), weights_only=False)

entity_to_id = {}
mapping_path = os.path.join(LATEST_RUN_DIR, "training_triples", "entity_to_id.tsv.gz")
with gzip.open(mapping_path, "rt", encoding="utf-8") as f:
    next(f)
    for line in f:
        e_id, e_uri = line.strip().split("\t")
        entity_to_id[e_uri] = int(e_id)

# Process Flats and Query KG Statements Directly
map_export = []

for flat_uri in g.subjects(RDF.type, EX.Flat):
    flat_str = str(flat_uri)
    
    price_lit = g.value(flat_uri, EX.hasPrice)
    lat_lit = g.value(flat_uri, EX.latitude)
    lon_lit = g.value(flat_uri, EX.longitude)

    if price_lit and flat_str in entity_to_id:
        actual_price = float(price_lit)
        e_id = entity_to_id[flat_str]

        # Get KGE representation and predict price
        vector = model.entity_representations[0](torch.tensor([e_id], device=model.device)).detach().cpu().numpy()[0]
        vector_real = np.concatenate([vector.real, vector.imag]) if np.iscomplexobj(vector) else vector

        pred_price = float(regressor.predict([vector_real])[0])
        is_good_deal = bool(actual_price < pred_price)

        # Directly read inferred assertions from KG
        is_well_connected = (flat_uri, RDF.type, EX.WellConnectedFlat) in g
        has_direct_access = (flat_uri, EX.hasDirectStationAccess, None) in g

        size_lit = g.value(flat_uri, EX.hasSize) 
        rooms_lit = g.value(flat_uri, EX.hasRooms)

        map_export.append({
            "LATITUDE": str(round(float(lat_lit), 6)) if lat_lit else "48.2082",
            "LONGITUDE": str(round(float(lon_lit), 6)) if lon_lit else "16.3738",
            "PRICE": str(round(actual_price, 2)),
            "actual_price": round(actual_price, 2),
            "predicted_price": round(pred_price, 2),
            "is_good_deal": is_good_deal,
            "is_well_connected": is_well_connected,
            "has_direct_access": has_direct_access,
            "estate_size": float(size_lit) if size_lit else None,
            "number_of_rooms": int(rooms_lit) if rooms_lit else None,
            "uri": flat_str
        })

# Save JSON file
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(map_export, f, indent=4, ensure_ascii=False)

print(f"\nSuccessfully generated {len(map_export)} flat records.")
print(f"Output saved to: {OUTPUT_JSON}")