"""
Representative Embedding Vectors Extraction Pipeline.
This script extracts high-dimensional embedding vectors for five carefully 
selected, representative entities from the trained Knowledge Graph Embedding model:
1. A specific transit stop ('Reumannplatz')
2. A specific residential flat near that stop ('flat_gen_65')
3. Semantic ontology classes ('WellConnectedFlat', 'TransitStop', 'Flat')

It formats the complex numbers into readable mathematical notation (Real + Imaginary i) 
and exports the result to a text file for direct inclusion in the portfolio report.
"""

import os
import glob
import gzip
import torch
import numpy as np
from rdflib import Graph, Namespace, RDF, RDFS

# ==============================================================================
# Path Definitions & Setup
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Updated folder structure targeting the fully reasoned knowledge graph (3_logic)
KG_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "src", "3_logic", "inferred_vienna_kg.ttl"))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

EX = Namespace("http://vienna-realestate.org/kg/")

# ==============================================================================
# 1. Locate the latest run directory
# ==============================================================================
# Scan the results directory to automatically find the most recent valid model run
valid_dirs = [d for d in glob.glob(os.path.join(RESULTS_DIR, "run_*")) 
              if os.path.exists(os.path.join(d, "trained_model.pkl"))]

if not valid_dirs:
    raise FileNotFoundError("No valid run directory with trained_model.pkl found in results!")

LATEST_RUN_DIR = max(valid_dirs, key=os.path.getctime)
OUTPUT_FILE = os.path.join(LATEST_RUN_DIR, "embedding_examples.txt")

print(f"Loading KGE model from: {os.path.basename(LATEST_RUN_DIR)}")

# Load the trained PyTorch KGE model into memory
model = torch.load(os.path.join(LATEST_RUN_DIR, "trained_model.pkl"), weights_only=False)

# ==============================================================================
# 2. Load Entity and Relation mappings
# ==============================================================================
# Reconstruct the mapping that links real-world RDF URIs to PyTorch integer IDs
entity_to_id = {}
with gzip.open(os.path.join(LATEST_RUN_DIR, "training_triples", "entity_to_id.tsv.gz"), "rt", encoding="utf-8") as f:
    next(f)  # Skip TSV header
    for line in f:
        e_id, e_uri = line.strip().split("\t")
        entity_to_id[e_uri] = int(e_id)

# ==============================================================================
# 3. Parse Knowledge Graph
# ==============================================================================
# Load the RDF graph to search for specific human-readable labels and types
print("Loading Knowledge Graph...")
g = Graph()
g.parse(KG_PATH, format="turtle")

# ==============================================================================
# 4. Search for Reumannplatz transit stop (STRICTLY constrained to EX.TransitStop)
# ==============================================================================
reumannplatz_uri = None
for s, p, o in g.triples((None, RDFS.label, None)):
    if "Reumannplatz" in str(o) and (s, RDF.type, EX.TransitStop) in g:
        reumannplatz_uri = s
        break

# Fallback URI search filtered strictly by EX.TransitStop if the label lookup fails
if not reumannplatz_uri:
    for s in g.subjects(RDF.type, EX.TransitStop):
        if "reumannplatz" in str(s).lower():
            reumannplatz_uri = s
            break

# ==============================================================================
# 5. Define Target Entities for Portfolio Inspection
# ==============================================================================
# Explicitly target node: flat_gen_65 (a representative flat near Reumannplatz)
target_flat_uri = str(EX.flat_gen_65)

# Curated list of 5 key entities and conceptual classes to showcase in the report
SELECTED_URIS = [
    ("Reumannplatz Stop", str(reumannplatz_uri) if reumannplatz_uri else None),
    ("flat_gen_65 (flat near to Reumannplatz Stop)", target_flat_uri),
    ("Class: WellConnectedFlat", str(EX.WellConnectedFlat)),
    ("Class: TransitStop", str(EX.TransitStop)),
    ("Class: Flat", str(EX.Flat))
]

output_lines = []
output_lines.append("=== 5 REPRESENTATIVE EMBEDDING VECTORS FOR PORTFOLIO ===\n")

# Iterate through selected items and extract their raw vector representations from PyTorch
for label, uri in SELECTED_URIS:
    if not uri or uri not in entity_to_id:
        output_lines.append(f"⚠️ [{label}] URI not found in KGE mapping: {uri}\n")
        continue

    e_id = entity_to_id[uri]
    
    # Query the embedding representation layer for this specific entity ID, 
    # detach it from gradient history, move it to standard RAM, and convert to NumPy
    emb = model.entity_representations[0](torch.tensor([e_id], device=model.device)).detach().cpu().numpy()[0]
    
    output_lines.append(f"Item: {label}")
    output_lines.append(f"  URI: {uri}")
    output_lines.append(f"  Total Dimension: {len(emb)}")

    # --------------------------------------------------------------------------
    # RotatE Complex Number Formatting
    # --------------------------------------------------------------------------
    # RotatE outputs vectors of complex numbers (Real + Imaginary parts).
    # We format them cleanly as "Re + Im i" with 4 decimal places for the portfolio report.
    if np.iscomplexobj(emb):
        formatted_elements = []
        for val in emb[:5]:  # Preview the first 5 dimensions
            re = float(val.real)
            im = float(val.imag)
            sign = "+" if im >= 0 else "-"
            formatted_elements.append(f"{re:.4f} {sign} {abs(im):.4f}i")
        
        vector_str = ", ".join(formatted_elements)
        output_lines.append(f"  Vector (first 5): [{vector_str}, ...]")
    else:
        # Fallback for standard real-valued models
        formatted_elements = [f"{float(x):.4f}" for x in emb[:5]]
        vector_str = ", ".join(formatted_elements)
        output_lines.append(f"  Vector (first 5): [{vector_str}, ...]")
    
    output_lines.append("-" * 65)

# Print results to the console
full_output = "\n".join(output_lines)
print(full_output)

# Save formatted examples to a text file inside the active model run directory
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(full_output)

print(f"\nResults successfully saved to: {OUTPUT_FILE}")