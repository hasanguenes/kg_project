import os
import glob
import gzip
import torch
import numpy as np
from rdflib import Graph, Namespace, RDF, RDFS

# --- Path Definitions ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Updated folder structure (3_logic)
KG_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "src", "3_logic", "inferred_vienna_kg.ttl"))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

EX = Namespace("http://vienna-realestate.org/kg/")

# 1. Locate the latest run directory
valid_dirs = [d for d in glob.glob(os.path.join(RESULTS_DIR, "run_*")) 
              if os.path.exists(os.path.join(d, "trained_model.pkl"))]

if not valid_dirs:
    raise FileNotFoundError("No valid run directory with trained_model.pkl found in results!")

LATEST_RUN_DIR = max(valid_dirs, key=os.path.getctime)
OUTPUT_FILE = os.path.join(LATEST_RUN_DIR, "embedding_examples.txt")

print(f"Loading KGE model from: {os.path.basename(LATEST_RUN_DIR)}")

model = torch.load(os.path.join(LATEST_RUN_DIR, "trained_model.pkl"), weights_only=False)

# 2. Load Entity and Relation mappings
entity_to_id = {}
with gzip.open(os.path.join(LATEST_RUN_DIR, "training_triples", "entity_to_id.tsv.gz"), "rt", encoding="utf-8") as f:
    next(f)
    for line in f:
        e_id, e_uri = line.strip().split("\t")
        entity_to_id[e_uri] = int(e_id)

# 3. Parse Knowledge Graph
print("Loading Knowledge Graph...")
g = Graph()
g.parse(KG_PATH, format="turtle")

# 4. Search for Reumannplatz transit stop (STRICTLY constrained to EX.TransitStop)
reumannplatz_uri = None
for s, p, o in g.triples((None, RDFS.label, None)):
    if "Reumannplatz" in str(o) and (s, RDF.type, EX.TransitStop) in g:
        reumannplatz_uri = s
        break

# Fallback URI search filtered strictly by EX.TransitStop
if not reumannplatz_uri:
    for s in g.subjects(RDF.type, EX.TransitStop):
        if "reumannplatz" in str(s).lower():
            reumannplatz_uri = s
            break

# 5. Explicitly target node: flat_gen_65
target_flat_uri = str(EX.flat_gen_65)

# Selected 5 entities/classes (using flat_gen_65 & verified TransitStop)
SELECTED_URIS = [
    ("Reumannplatz Stop", str(reumannplatz_uri) if reumannplatz_uri else None),
    ("flat_gen_65 (flat near to Reumannplatz Stop)", target_flat_uri),
    ("Class: WellConnectedFlat", str(EX.WellConnectedFlat)),
    ("Class: TransitStop", str(EX.TransitStop)),
    ("Class: Flat", str(EX.Flat))
]

output_lines = []
output_lines.append("=== 5 REPRESENTATIVE EMBEDDING VECTORS FOR PORTFOLIO ===\n")

for label, uri in SELECTED_URIS:
    if not uri or uri not in entity_to_id:
        output_lines.append(f"⚠️ [{label}] URI not found in KGE mapping: {uri}\n")
        continue

    e_id = entity_to_id[uri]
    emb = model.entity_representations[0](torch.tensor([e_id], device=model.device)).detach().cpu().numpy()[0]
    
    output_lines.append(f"Item: {label}")
    output_lines.append(f"  URI: {uri}")
    output_lines.append(f"  Total Dimension: {len(emb)}")

    # Format vector elements strictly with 4 decimal places (Re + Im*i)
    if np.iscomplexobj(emb):
        formatted_elements = []
        for val in emb[:5]:
            re = float(val.real)
            im = float(val.imag)
            sign = "+" if im >= 0 else "-"
            formatted_elements.append(f"{re:.4f} {sign} {abs(im):.4f}i")
        
        vector_str = ", ".join(formatted_elements)
        output_lines.append(f"  Vector (first 5): [{vector_str}, ...]")
    else:
        formatted_elements = [f"{float(x):.4f}" for x in emb[:5]]
        vector_str = ", ".join(formatted_elements)
        output_lines.append(f"  Vector (first 5): [{vector_str}, ...]")
    
    output_lines.append("-" * 65)

# Print to console and write to file
full_output = "\n".join(output_lines)
print(full_output)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(full_output)

print(f"\nResults successfully saved to: {OUTPUT_FILE}")