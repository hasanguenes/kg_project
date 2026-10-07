import os
import glob
import gzip
import torch
import numpy as np
from rdflib import Graph, Namespace, RDF

# --- Path Definitions ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Updated folder structure (3_logic)
KG_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "src", "3_logic", "inferred_vienna_kg.ttl"))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

EX = Namespace("http://vienna-realestate.org/kg/")

# 1. Locate latest run directory
valid_dirs = [d for d in glob.glob(os.path.join(RESULTS_DIR, "run_*")) 
              if os.path.exists(os.path.join(d, "trained_model.pkl"))]

if not valid_dirs:
    raise FileNotFoundError("No valid run directory with trained_model.pkl found in results!")

LATEST_RUN_DIR = max(valid_dirs, key=os.path.getctime)
OUTPUT_FILE = os.path.join(LATEST_RUN_DIR, "tp_fp_evaluation.txt")

print(f"Analyzing run: {os.path.basename(LATEST_RUN_DIR)}")

# 2. Load Trained KGE Model & Knowledge Graph
model = torch.load(os.path.join(LATEST_RUN_DIR, "trained_model.pkl"), weights_only=False)

print("Loading Knowledge Graph...")
g = Graph()
g.parse(KG_PATH, format="turtle")

# 3. Load Mappings
entity_to_id = {}
with gzip.open(os.path.join(LATEST_RUN_DIR, "training_triples", "entity_to_id.tsv.gz"), "rt", encoding="utf-8") as f:
    next(f)
    for line in f:
        e_id, e_uri = line.strip().split("\t")
        entity_to_id[e_uri] = int(e_id)

relation_to_id = {}
with gzip.open(os.path.join(LATEST_RUN_DIR, "training_triples", "relation_to_id.tsv.gz"), "rt", encoding="utf-8") as f:
    next(f)
    for line in f:
        r_id, r_uri = line.strip().split("\t")
        relation_to_id[r_uri] = int(r_id)

target_rel_uri = str(EX.isNearStop)

if target_rel_uri not in relation_to_id:
    raise ValueError(f"Relation {target_rel_uri} not found in relation_to_id mapping!")

r_id = relation_to_id[target_rel_uri]

# Extract ground truth triples for ex:isNearStop
ground_truth_edges = set()
for s, p, o in g.triples((None, EX.isNearStop, None)):
    ground_truth_edges.add((str(s), str(o)))

all_flats = [str(s) for s in g.subjects(RDF.type, EX.Flat) if str(s) in entity_to_id]
all_stops = [str(s) for s in g.subjects(RDF.type, EX.TransitStop) if str(s) in entity_to_id]

print(f"Evaluating link prediction for relation: ex:isNearStop ({len(ground_truth_edges)} actual triples)...")

scored_triples = []

# Evaluate candidate pairs (Flat, isNearStop, Stop)
for flat_uri in all_flats[:30]:  # Sample candidate subset for speed
    h_id = entity_to_id[flat_uri]
    
    for stop_uri in all_stops[:30]:
        t_id = entity_to_id[stop_uri]
        
        # Prepare batch tensor (1, 3) -> [h_id, r_id, t_id]
        hrt_batch = torch.tensor([[h_id, r_id, t_id]], device=model.device)
        
        # Compute link prediction score using PyKEEN model
        score = model.score_hrt(hrt_batch).detach().cpu().item()
        
        is_true_edge = (flat_uri, stop_uri) in ground_truth_edges
        
        scored_triples.append({
            "flat_uri": flat_uri,
            "stop_uri": stop_uri,
            "score": score,
            "is_true_edge": is_true_edge
        })

# Sort triples by prediction score descending (highest score = most plausible)
scored_triples.sort(key=lambda x: x["score"], reverse=True)

# Find True Positive (highest score, actually exists)
tp_candidates = [t for t in scored_triples if t["is_true_edge"]]
# Find False Positive (highest score, does NOT exist in graph)
fp_candidates = [t for t in scored_triples if not t["is_true_edge"]]

output_lines = []
output_lines.append("=== LINK PREDICTION PORTFOLIO EXAMPLES (ex:isNearStop) ===\n")

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

# Print to console and write to file
full_output = "\n".join(output_lines)
print(full_output)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(full_output)

print(f"\nResults successfully saved to: {OUTPUT_FILE}")