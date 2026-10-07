"""
Embedding Evaluation and Visualization Pipeline.
This script evaluates the quality of the learned Knowledge Graph Embeddings (KGE) 
by analyzing their vector representations. It performs two main tasks:
1. Cosine Similarity Matrix: Measures the exact mathematical closeness of specific nodes.
2. Principal Component Analysis (PCA): Reduces the high-dimensional vectors (e.g., 256D) 
   down to 2D to visually verify if the model successfully learned to cluster 
   conceptually or geographically similar entities (e.g., Prime District Flats vs. 10th District Flats).
"""

import gzip
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdflib import RDF, Graph, Literal, Namespace, URIRef
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import torch

# ==============================================================================
# 1. Path Definitions & Model Loading
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

KG_PATH = os.path.join(PROJECT_ROOT, "src", "3_logic", "inferred_vienna_kg.ttl")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Manual Override: Hardcode a specific run to ensure reproducibility for the evaluation plots.
# MANUAL_RUN_NAME = "run_20261007_070839_RotatE_dim128"
valid_dirs = [d for d in glob.glob(os.path.join(RESULTS_DIR, "run_*")) 
              if os.path.exists(os.path.join(d, "trained_model.pkl"))]

if not valid_dirs:
    raise FileNotFoundError("No valid run directory with trained_model.pkl found in results!")

LATEST_RUN_DIR = max(valid_dirs, key=os.path.getctime)

print(f"Loading model from: {os.path.basename(LATEST_RUN_DIR)}")

# Parse Knowledge Graph to extract human-readable labels for our entities
print("Parsing Knowledge Graph...")
g = Graph()
g.parse(KG_PATH, format="turtle")

# Load the trained PyTorch KGE model into memory
model = torch.load(
    os.path.join(LATEST_RUN_DIR, "trained_model.pkl"), weights_only=False
)

# Load the entity-to-ID mapping to query specific vectors from the model
entity_to_id = {}
mapping_path = os.path.join(
    LATEST_RUN_DIR, "training_triples", "entity_to_id.tsv.gz"
)

with gzip.open(mapping_path, "rt", encoding="utf-8") as f:
    next(f)  # Skip TSV header
    for line in f:
        e_id, e_uri = line.strip().split("\t")
        entity_to_id[e_uri] = int(e_id)

EX = Namespace("http://vienna-realestate.org/kg/")

def get_entity_vector(uri):
    """
    Extracts and formats the embedding vector for a given graph URI.
    
    If the model uses RotatE (which operates in the complex plane), the vector 
    contains complex numbers (real + imaginary). Since standard visualization 
    and similarity metrics (like PCA and Cosine Similarity in Scikit-Learn) 
    require real numbers, this function concatenates the real and imaginary 
    parts into a single, flattened, real-valued array of double the original length.
    """
    if uri not in entity_to_id:
        return None
        
    e_id = entity_to_id[uri]
    # Move the integer ID to a tensor on the correct processing device
    tensor_id = torch.tensor([e_id], device=model.device)
    
    # Query the embedding layer, detach from the computation graph, and move to CPU
    raw_vec = (
        model.entity_representations[0](tensor_id).detach().cpu().numpy()[0]
    )
    
    # RotatE complex number resolution
    if np.iscomplexobj(raw_vec):
        return np.concatenate([raw_vec.real, raw_vec.imag])
    return raw_vec


# ==============================================================================
# 2. Select Extended Entity Groups (Clean Labels)
# ==============================================================================
# We deliberately select specific groups of entities to see if the model has 
# learned spatial (geography) and semantic (ontology) relationships.
target_entities = []  # Stores Tuples: (URI, Clean_Label, Group_Category)

# --- Topic 1: Transit Stops (Karlsplatz, Reumannplatz & Praterstern) ---
# We scan the graph for specific prominent transit hubs to see if the model 
# clusters stops that belong to the same geographical area.
for s, p, o in g:
    if isinstance(o, Literal):
        val = str(o).lower()
        s_str = str(s)
        if s_str in entity_to_id:
            suffix = s_str.split("/")[-1]
            if "karlsplatz" in val and not any(e[0] == s_str for e in target_entities):
                target_entities.append((s_str, suffix, "Stop: Karlsplatz"))
            elif "reumannplatz" in val and not any(e[0] == s_str for e in target_entities):
                target_entities.append((s_str, suffix, "Stop: Reumannplatz"))
            elif "praterstern" in val and not any(e[0] == s_str for e in target_entities):
                target_entities.append((s_str, suffix, "Stop: Praterstern"))

# --- Topic 2: Residential Counterpoles (District 10 vs. Prime Districts) ---
# We want to check if the embeddings capture socio-economic/geographic differences 
# by comparing flats in the 10th district (Favoriten) against prime inner-city/wealthy 
# districts (1st & 19th).
district_10_flats = []
prime_district_flats = []

for s, p, o in g:
    s_str = str(s)
    o_str = str(o).lower()
    if "flat" in s_str.lower() and s_str in entity_to_id:
        if ("1100" in o_str or "district_10" in o_str or "district 10" in o_str) and s_str not in district_10_flats:
            district_10_flats.append(s_str)
        elif ("1010" in o_str or "1190" in o_str or "district_1" in o_str or "district_19" in o_str) and s_str not in prime_district_flats:
            prime_district_flats.append(s_str)

# Fallback: If no prime flats are found based on strings, grab a few generic ones to prevent crashes
if not prime_district_flats:
    prime_district_flats = [u for u in entity_to_id.keys() if "flat" in u.lower() and u not in district_10_flats][:4]

# Add a subset (max 4) of these flats to our target list
for flat_uri in district_10_flats[:4]:
    suffix = flat_uri.split("/")[-1]
    target_entities.append((flat_uri, suffix, "Flat: 10th District"))

for flat_uri in prime_district_flats[:4]:
    suffix = flat_uri.split("/")[-1]
    target_entities.append((flat_uri, suffix, "Flat: Prime District"))

# --- Topic 3: Concepts & Ontology Classes ---
# Finally, we include abstract structural nodes. Do "Flats" cluster near the 
# "Flat" class node? Does "WellConnectedFlat" sit somewhere distinct?
concept_classes = ["TransitStop", "Flat", "WellConnectedFlat"]
for cls in concept_classes:
    cls_uri = str(EX[cls])
    if cls_uri in entity_to_id:
        target_entities.append((cls_uri, cls, "Ontology Class"))

print(f"\nExtracted {len(target_entities)} total entities across extended categories.")


# ==============================================================================
# 3. Extract Embeddings & Compute Cosine Similarity Matrix
# ==============================================================================
vectors = []
labels = []
categories = []

# Fetch the numerical vectors for all identified targets
for uri, lbl, cat in target_entities:
    vec = get_entity_vector(uri)
    if vec is not None:
        vectors.append(vec)
        labels.append(lbl)
        categories.append(cat)

X = np.array(vectors)

if len(X) >= 2:
    # Compute Cosine Similarity: Measures the angle between two vectors. 
    # Values close to 1.0 mean the vectors point in the exact same direction (highly similar).
    sim_matrix = cosine_similarity(X)
    df_sim = pd.DataFrame(sim_matrix, index=labels, columns=labels)

    print("\n" + "=" * 80)
    print("COSINE SIMILARITY MATRIX")
    print("=" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    # Print a rounded matrix to the console for quick inspection
    print(df_sim.round(3))
    print("=" * 80 + "\n")

    # ==============================================================================
    # 4. Clean PCA Visualization
    # ==============================================================================
    # PCA (Principal Component Analysis) reduces our high-dimensional embedding space 
    # (e.g., 256 dimensions) down to 2 dimensions by finding the axes of maximum variance.
    # This allows us to plot a complex vector space on a standard 2D scatter plot.
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    # Initialize plotting canvas
    plt.figure(figsize=(13, 8.5))

    unique_cats = sorted(list(set(categories)))
    color_map = plt.cm.get_cmap("tab10", len(unique_cats))

    # Scatter plot loop: plots points category by category so we get a grouped legend
    for idx, cat in enumerate(unique_cats):
        indices = [i for i, c in enumerate(categories) if c == cat]
        plt.scatter(
            X_2d[indices, 0],
            X_2d[indices, 1],
            label=cat,
            s=130,  # Marker size
            edgecolors="k", # Black border around markers
            alpha=0.88,
            color=color_map(idx)
        )

    # Add clean text labels directly next to the scatter points
    for i, label in enumerate(labels):
        plt.annotate(
            label,
            (X_2d[i, 0], X_2d[i, 1]),
            fontsize=8,
            fontweight="bold",
            alpha=0.85,
            xytext=(6, 6), # Offset the text slightly from the dot
            textcoords="offset points",
        )

    # Simplified title and informative subtitle
    plt.suptitle("Principal Component Analysis", fontsize=14, fontweight="bold", y=0.94)
    plt.title("Vector Space Projection comparing Vienna Public Transport Stops, District Properties, and Ontology Classes", fontsize=10, pad=10)

    # Label the abstract PCA axes
    plt.xlabel("Principal Component 1", fontsize=10)
    plt.ylabel("Principal Component 2", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Render legend based on the categories
    plt.legend(loc="best", fontsize=9, title="Entity Group")

    # Save the output image inside the active model run directory
    plot_path = os.path.join(LATEST_RUN_DIR, "extended_similarity_pca_clean.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Clean PCA plot saved to: {plot_path}")
    
    # Display the plot window
    plt.show()