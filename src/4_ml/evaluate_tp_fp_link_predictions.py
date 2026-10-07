"""
Link Prediction Evaluation Pipeline (True/False Positives).
This script evaluates the generative power of the trained Knowledge Graph Embedding (KGE) model.
Instead of predicting prices, it predicts structural edges (links) within the graph.
Specifically, it scores the plausibility of "ex:isNearStop" connections between Flats and Stops.

It identifies:
- True Positives (TP): The model assigns a high score, and the connection actually exists in the KG.
- False Positives (FP): The model assigns a high score, but the connection is missing in the KG.
  (Note: In KG completion, highly scored False Positives are often valuable "missing links" 
  or highly plausible connections rather than sheer errors).
"""

import os
import glob
import gzip
import torch
import numpy as np
from rdflib import Graph, Namespace, RDF

# ==============================================================================
# Path Definitions & Setup
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Updated folder structure targeting the fully logical graph (3_logic)
KG_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "src", "3_logic", "inferred_vienna_kg.ttl"))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

EX = Namespace("http://vienna-realestate.org/kg/")


# ==============================================================================
# 1. Locate the latest valid PyKEEN run directory
# ==============================================================================
# # Dynamically scan the results directory for completed training runs
# valid_dirs = [d for d in glob.glob(os.path.join(RESULTS_DIR, "run_*")) 
#               if os.path.exists(os.path.join(d, "trained_model.pkl"))]

# if not valid_dirs:
#     raise FileNotFoundError("No valid run directory with trained_model.pkl found in results!")

MANUAL_RUN_NAME = "run_20261007_070839_RotatE_dim128"
RUN_DIR = os.path.join(RESULTS_DIR, MANUAL_RUN_NAME)
OUTPUT_FILE = os.path.join(RUN_DIR, "tp_fp_evaluation.txt")

print(f"Analyzing run: {os.path.basename(RUN_DIR)}")


# ==============================================================================
# 2. Load Trained KGE Model & Base Knowledge Graph
# ==============================================================================
# Load the PyTorch KGE model into memory
model = torch.load(os.path.join(RUN_DIR, "trained_model.pkl"), weights_only=False)

print("Loading Knowledge Graph...")
g = Graph()
g.parse(KG_PATH, format="turtle")


# ==============================================================================
# 3. Load PyKEEN Integer Mappings (Entities and Relations)
# ==============================================================================
# PyKEEN translates all RDF URIs into internal integer IDs for tensor math.
# We must load both the entity mapping (nodes) and relation mapping (edges).
entity_to_id = {}
with gzip.open(os.path.join(RUN_DIR, "training_triples", "entity_to_id.tsv.gz"), "rt", encoding="utf-8") as f:
    next(f)
    for line in f:
        e_id, e_uri = line.strip().split("\t")
        entity_to_id[e_uri] = int(e_id)

relation_to_id = {}
with gzip.open(os.path.join(RUN_DIR, "training_triples", "relation_to_id.tsv.gz"), "rt", encoding="utf-8") as f:
    next(f)
    for line in f:
        r_id, r_uri = line.strip().split("\t")
        relation_to_id[r_uri] = int(r_id)

# Identify the integer ID for the specific relation we want to predict/evaluate
target_rel_uri = str(EX.isNearStop)

if target_rel_uri not in relation_to_id:
    raise ValueError(f"Relation {target_rel_uri} not found in relation_to_id mapping!")

r_id = relation_to_id[target_rel_uri]


# ==============================================================================
# 4. Extract Ground Truth Data
# ==============================================================================
# Build a fast-lookup set of all actual 'isNearStop' edges that exist in the graph.
ground_truth_edges = set()
for s, p, o in g.triples((None, EX.isNearStop, None)):
    ground_truth_edges.add((str(s), str(o)))

# Fetch all candidate Flats and Stops from the graph that the model has seen during training
all_flats = [str(s) for s in g.subjects(RDF.type, EX.Flat) if str(s) in entity_to_id]
all_stops = [str(s) for s in g.subjects(RDF.type, EX.TransitStop) if str(s) in entity_to_id]

print(f"Evaluating link prediction for relation: ex:isNearStop ({len(ground_truth_edges)} actual triples)...")


# ==============================================================================
# 5. Evaluate Link Plausibility (Scoring HRT Triples)
# ==============================================================================
scored_triples = []

# Evaluate a subset of candidate pairs (Flat, isNearStop, Stop) to save computation time.
# We slice [:30] because scoring a full Cartesian product (N_flats * M_stops) 
# would take significantly longer without batch-tensor optimizations.
for flat_uri in all_flats[:30]:  # Sample candidate subset for speed
    h_id = entity_to_id[flat_uri] # Head ID
    
    for stop_uri in all_stops[:30]:
        t_id = entity_to_id[stop_uri] # Tail ID
        
        # Prepare a PyTorch tensor batch representing a single (Head, Relation, Tail) triple.
        # Format required by PyKEEN: shape (1, 3) -> [[h_id, r_id, t_id]]
        hrt_batch = torch.tensor([[h_id, r_id, t_id]], device=model.device)
        
        # Compute the link prediction score.
        # Higher score = the model believes this relationship is highly plausible based on topology.
        # .detach() isolates it from gradients, .cpu() moves it to RAM, .item() extracts the python float.
        score = model.score_hrt(hrt_batch).detach().cpu().item()
        
        # Check if this highly plausible edge actually exists in our known dataset
        is_true_edge = (flat_uri, stop_uri) in ground_truth_edges
        
        scored_triples.append({
            "flat_uri": flat_uri,
            "stop_uri": stop_uri,
            "score": score,
            "is_true_edge": is_true_edge
        })

# Sort all evaluated triples by prediction score in descending order (highest score first)
scored_triples.sort(key=lambda x: x["score"], reverse=True)


# ==============================================================================
# 6. Extract and Export TP / FP Examples
# ==============================================================================
# Find True Positive: The model gave it a high score, and it IS a real connection in the graph.
# This proves the model learned the underlying spatial/transit patterns correctly.
tp_candidates = [t for t in scored_triples if t["is_true_edge"]]

# Find False Positive: The model gave it a high score, but it is NOT in the graph.
# In traditional ML, this is an error. In Knowledge Graphs, this is the magic of "Link Prediction":
# The model suggests that based on structural similarities, this flat *should* be near this stop.
fp_candidates = [t for t in scored_triples if not t["is_true_edge"]]

output_lines = []
output_lines.append("=== LINK PREDICTION PORTFOLIO EXAMPLES (ex:isNearStop) ===\n")

# Format True Positive output
if tp_candidates:
    tp = tp_candidates[0]
    output_lines.append("1. TRUE POSITIVE LINK PREDICTION:")
    output_lines.append(f"   Head (Flat): {tp['flat_uri']}")
    output_lines.append(f"   Relation:   {target_rel_uri}")
    output_lines.append(f"   Tail (Stop): {tp['stop_uri']}")
    output_lines.append(f"   Model Prediction Score: {tp['score']:.4f}")
    output_lines.append("   Explanation: High prediction score and the edge actually exists in the KG.\n")
else:
    output_lines.append("1. TRUE POSITIVE LINK PREDICTION: None found in sample.\n")

# Format False Positive output
if fp_candidates:
    fp = fp_candidates[0]
    output_lines.append("2. FALSE POSITIVE LINK PREDICTION:")
    output_lines.append(f"   Head (Flat): {fp['flat_uri']}")
    output_lines.append(f"   Relation:   {target_rel_uri}")
    output_lines.append(f"   Tail (Stop): {fp['stop_uri']}")
    output_lines.append(f"   Model Prediction Score: {fp['score']:.4f}")
    output_lines.append("   Explanation: High prediction score, but the edge DOES NOT exist in the real KG.\n")
else:
    output_lines.append("2. FALSE POSITIVE LINK PREDICTION: None found in sample.\n")

output_lines.append("-" * 65)

# Print the results to the terminal
full_output = "\n".join(output_lines)
print(full_output)

# Save the findings to a text file inside the run directory for the portfolio report
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(full_output)

print(f"\nResults successfully saved to: {OUTPUT_FILE}")