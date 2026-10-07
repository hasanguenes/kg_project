import os
import glob
import json
import gzip
import torch
import numpy as np
from rdflib import Graph, Namespace, RDF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, 
    r2_score, 
    mean_squared_error, 
    mean_absolute_percentage_error
)

# --- Path Definitions ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

KG_PATH = os.path.join(PROJECT_ROOT, "src", "3_logic", "inferred_vienna_kg.ttl")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "src", "map_data.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

EX = Namespace("http://vienna-realestate.org/kg/")

# 1. Robustly find the latest PyKEEN run that contains the trained model
valid_dirs = []
run_folders = glob.glob(os.path.join(RESULTS_DIR, "run_*"))

for d in run_folders:
    # Check for .pkl and the compressed .tsv.gz file
    if os.path.exists(os.path.join(d, "trained_model.pkl")) and os.path.exists(os.path.join(d, "training_triples", "entity_to_id.tsv.gz")):
        valid_dirs.append(d)

if not valid_dirs:
    raise FileNotFoundError("No valid PyKEEN model found in any results directory!")


# # 1. Manually select the PyKEEN run directory
# MANUAL_RUN_NAME = "run_20261006_212847_RotatE_dim256" # Hier deinen Wunschnamen eintragen
# LATEST_RUN_DIR = os.path.join(RESULTS_DIR, MANUAL_RUN_NAME)
# if not os.path.exists(LATEST_RUN_DIR):
#     raise FileNotFoundError(f"Der Ordner {LATEST_RUN_DIR} existiert nicht!")

# Pick the most recently modified directory that is valid
LATEST_RUN_DIR = max(valid_dirs, key=os.path.getctime)

print(f"Loading KGE model from: {os.path.basename(LATEST_RUN_DIR)}")

# 2. Load Knowledge Graph
print("Loading Knowledge Graph for price extraction...")
g = Graph()
g.parse(KG_PATH, format="turtle")

# 3. Load PyKEEN Model & Entity Mappings (using gzip)
model = torch.load(os.path.join(LATEST_RUN_DIR, "trained_model.pkl"), weights_only=False)
entity_to_id = {}
mapping_path = os.path.join(LATEST_RUN_DIR, "training_triples", "entity_to_id.tsv.gz")

with gzip.open(mapping_path, "rt", encoding="utf-8") as f:
    next(f)  # Skip header
    for line in f:
        e_id, e_uri = line.strip().split("\t")
        entity_to_id[e_uri] = int(e_id)

# 4. Build Features (X) and Target (y)
X_vectors = []
y_prices = []
flat_data = []

flats = list(g.subjects(RDF.type, EX.Flat))
for flat_uri in flats:
    flat_str = str(flat_uri)

    price_literal = g.value(flat_uri, EX.hasPrice)
    lat_literal = g.value(flat_uri, EX.latitude)
    lon_literal = g.value(flat_uri, EX.longitude)

    # flat_size_literal = g.value(flat_uri, EX.hasSize)
    # flat_rooms_literal = g.value(flat_uri, EX.hasRooms)
    
    if price_literal and flat_str in entity_to_id:
        actual_price = float(price_literal)
        e_id = entity_to_id[flat_str]
        
        # Extract vector
        vector = model.entity_representations[0](torch.tensor([e_id], device=model.device)).detach().cpu().numpy()[0]
        
        # RotatE Fix: Split complex numbers for Scikit-Learn (concatenate real and imaginary parts)
        if np.iscomplexobj(vector):
            vector_real = np.concatenate([vector.real, vector.imag])
        else:
            vector_real = vector

        # size_val = float(flat_size_literal) if flat_size_literal else 0.0
        # rooms_val = float(flat_rooms_literal) if flat_rooms_literal else 0.0

        # Late Fusion: Größe und Zimmer an den 512-dimensionalen Vektor anhängen
        # combined_features = np.append(vector_real, [size_val, rooms_val])
        # X_vectors.append(combined_features)
            
        X_vectors.append(vector_real)
        y_prices.append(actual_price)
        
        flat_data.append({
            "uri": flat_str,
            "lat": float(lat_literal) if lat_literal else None,
            "lon": float(lon_literal) if lon_literal else None,
            "actual_price": actual_price
        })

X = np.array(X_vectors)
y = np.array(y_prices)

# --- OUTPUT DATASET STATISTICS ---
print("\n--- DATASET STATISTICS ---")
print(f"Total Flats analyzed: {len(y)}")
print(f"Mean Price: {np.mean(y):.2f} EUR")
print(f"Median Price: {np.median(y):.2f} EUR")
print(f"Min Price: {np.min(y):.2f} EUR")
print(f"Max Price: {np.max(y):.2f} EUR")
print(f"Standard Deviation: {np.std(y):.2f} EUR")
print("--------------------------\n")

# 5. Train/Test Split (Simulate Unseen Data - 80% Train, 20% Test)
X_train, X_test, y_train, y_test, data_train, data_test = train_test_split(
    X, y, flat_data, test_size=0.2, random_state=42
)

# 6. Train Random Forest Regressor
print("Training Random Forest Regressor...")
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# 7. Evaluate on Unseen Data
predictions_test = regressor.predict(X_test)
predictions_train = regressor.predict(X_train)

mae = mean_absolute_error(y_test, predictions_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions_test))
mape = mean_absolute_percentage_error(y_test, predictions_test) * 100
r2 = r2_score(y_test, predictions_test)

print("\n--- PRICE REGRESSION RESULTS ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} EUR")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} EUR")
print(f"Mean Abs. Percentage Error (MAPE): {mape:.2f} %")
print(f"R2 Score (Explained Variance): {r2:.4f}")

# 8. Generate Map Export (Deal Detector Service)
map_export = []

def process_flats(data_subset, predictions, status_label):
    for i, flat in enumerate(data_subset):
        pred_price = round(predictions[i], 2)
        # A "Good Deal" means the flat is cheaper than its transit location suggests
        is_good_deal = bool(flat["actual_price"] < pred_price)
        
        flat["predicted_price"] = pred_price
        flat["is_good_deal"] = is_good_deal
        flat["status"] = status_label
        map_export.append(flat)

process_flats(data_test, predictions_test, "unseen_test_data")
process_flats(data_train, predictions_train, "seen_training_data")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(map_export, f, indent=4)

print(f"\nMap data for {len(map_export)} flats successfully exported to {OUTPUT_JSON}!")