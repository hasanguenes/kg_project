import os
import glob
import gzip
import joblib
import torch
import numpy as np
from rdflib import Graph, Namespace, RDF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- Path Definitions ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
KG_PATH = os.path.join(PROJECT_ROOT, "src", "3_logic", "inferred_vienna_kg.ttl")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

EX = Namespace("http://vienna-realestate.org/kg/")

# Find the latest PyKEEN run directory
valid_dirs = [d for d in glob.glob(os.path.join(RESULTS_DIR, "run_*")) 
              if os.path.exists(os.path.join(d, "trained_model.pkl"))]
RUN_DIR = max(valid_dirs, key=os.path.getctime)


# MANUAL_RUN_NAME = "run_20261007_070839_RotatE_dim128"
# RUN_DIR = os.path.join(RESULTS_DIR, MANUAL_RUN_NAME)

print(f"Loading KGE model from: {os.path.basename(RUN_DIR)}")

# Load Knowledge Graph & PyKEEN Model
g = Graph()
g.parse(KG_PATH, format="turtle")
model = torch.load(os.path.join(RUN_DIR, "trained_model.pkl"), weights_only=False)

entity_to_id = {}
with gzip.open(os.path.join(RUN_DIR, "training_triples", "entity_to_id.tsv.gz"), "rt", encoding="utf-8") as f:
    next(f)
    for line in f:
        e_id, e_uri = line.strip().split("\t")
        entity_to_id[e_uri] = int(e_id)

# Extract features and target variable
X_vectors, y_prices = [], []
for flat_uri in g.subjects(RDF.type, EX.Flat):
    flat_str = str(flat_uri)
    price_lit = g.value(flat_uri, EX.hasPrice)
    
    if price_lit and flat_str in entity_to_id:
        e_id = entity_to_id[flat_str]
        vector = model.entity_representations[0](torch.tensor([e_id], device=model.device)).detach().cpu().numpy()[0]
        
        vector_real = np.concatenate([vector.real, vector.imag]) if np.iscomplexobj(vector) else vector
        X_vectors.append(vector_real)
        y_prices.append(float(price_lit))

X, y = np.array(X_vectors), np.array(y_prices)

# Train/Test Split for portfolio evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Regressor...")
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Print evaluation results
preds_test = regressor.predict(X_test)
print("Evaluation results on test set:")
print(f" - MAE: {mean_absolute_error(y_test, preds_test):.2f} EUR")
print(f" - R2 Score: {r2_score(y_test, preds_test):.4f}")

# Save the regressor model inside the run directory
model_output_path = os.path.join(RUN_DIR, "price_regressor_.pkl")
joblib.dump(regressor, model_output_path)
print(f"\nRegressor successfully saved to:\n{model_output_path}")