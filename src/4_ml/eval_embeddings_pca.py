import gzip
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdflib import RDF, Graph, Literal, Namespace, URIRef
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import torch

# ---------------------------------------------------------
# 1. Path Definitions & Model Loading
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

KG_PATH = os.path.join(PROJECT_ROOT, "src", "3_logic", "inferred_vienna_kg.ttl")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# MANUAL_RUN_NAME = "run_20261007_044920_RotatE_dim256"
MANUAL_RUN_NAME = "run_20261007_070839_RotatE_dim128"
LATEST_RUN_DIR = os.path.join(RESULTS_DIR, MANUAL_RUN_NAME)

if not os.path.exists(LATEST_RUN_DIR):
    raise FileNotFoundError(f"Directory not found: {LATEST_RUN_DIR}")

print(f"Loading model from: {os.path.basename(LATEST_RUN_DIR)}")

# Parse Knowledge Graph
print("Parsing Knowledge Graph...")
g = Graph()
g.parse(KG_PATH, format="turtle")

# Load PyKEEN model and entity ID mapping
model = torch.load(
    os.path.join(LATEST_RUN_DIR, "trained_model.pkl"), weights_only=False
)
entity_to_id = {}
mapping_path = os.path.join(
    LATEST_RUN_DIR, "training_triples", "entity_to_id.tsv.gz"
)

with gzip.open(mapping_path, "rt", encoding="utf-8") as f:
    next(f)
    for line in f:
        e_id, e_uri = line.strip().split("\t")
        entity_to_id[e_uri] = int(e_id)

EX = Namespace("http://vienna-realestate.org/kg/")

def get_entity_vector(uri):
    """Extract concatenated real vector for RotatE embeddings."""
    if uri not in entity_to_id:
        return None
    e_id = entity_to_id[uri]
    tensor_id = torch.tensor([e_id], device=model.device)
    raw_vec = (
        model.entity_representations[0](tensor_id).detach().cpu().numpy()[0]
    )
    if np.iscomplexobj(raw_vec):
        return np.concatenate([raw_vec.real, raw_vec.imag])
    return raw_vec


# ---------------------------------------------------------
# 2. Select Extended Entity Groups (Clean Labels)
# ---------------------------------------------------------
target_entities = []  # Tuples: (URI, Clean_Label, Group_Category)

# --- Topic 1: Transit Stops (Karlsplatz, Reumannplatz & Praterstern) ---
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

if not prime_district_flats:
    prime_district_flats = [u for u in entity_to_id.keys() if "flat" in u.lower() and u not in district_10_flats][:4]

for flat_uri in district_10_flats[:4]:
    suffix = flat_uri.split("/")[-1]
    target_entities.append((flat_uri, suffix, "Flat: 10th District"))

for flat_uri in prime_district_flats[:4]:
    suffix = flat_uri.split("/")[-1]
    target_entities.append((flat_uri, suffix, "Flat: Prime District"))

# --- Concepts & Classes ---
concept_classes = ["TransitStop", "Flat", "WellConnectedFlat"]
for cls in concept_classes:
    cls_uri = str(EX[cls])
    if cls_uri in entity_to_id:
        target_entities.append((cls_uri, cls, "Ontology Class"))

print(f"\nExtracted {len(target_entities)} total entities across extended categories.")

# ---------------------------------------------------------
# 3. Extract Embeddings & Compute Cosine Similarity Matrix
# ---------------------------------------------------------
vectors = []
labels = []
categories = []

for uri, lbl, cat in target_entities:
    vec = get_entity_vector(uri)
    if vec is not None:
        vectors.append(vec)
        labels.append(lbl)
        categories.append(cat)

X = np.array(vectors)

if len(X) >= 2:
    sim_matrix = cosine_similarity(X)
    df_sim = pd.DataFrame(sim_matrix, index=labels, columns=labels)

    print("\n" + "=" * 80)
    print("COSINE SIMILARITY MATRIX")
    print("=" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df_sim.round(3))
    print("=" * 80 + "\n")

    # ---------------------------------------------------------
    # 4. Clean PCA Visualisation
    # ---------------------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(13, 8.5))

    unique_cats = sorted(list(set(categories)))
    color_map = plt.cm.get_cmap("tab10", len(unique_cats))

    for idx, cat in enumerate(unique_cats):
        indices = [i for i, c in enumerate(categories) if c == cat]
        plt.scatter(
            X_2d[indices, 0],
            X_2d[indices, 1],
            label=cat,
            s=130,
            edgecolors="k",
            alpha=0.88,
            color=color_map(idx)
        )

    # Clean label text directly next to points
    for i, label in enumerate(labels):
        plt.annotate(
            label,
            (X_2d[i, 0], X_2d[i, 1]),
            fontsize=8,
            fontweight="bold",
            alpha=0.85,
            xytext=(6, 6),
            textcoords="offset points",
        )

    # Simplified title and informative subtitle
    plt.suptitle("Principal Component Analysis", fontsize=14, fontweight="bold", y=0.94)
    plt.title("Vector Space Projection comparing Vienna Public Transport Stops, District Properties, and Ontology Classes", fontsize=10, pad=10)

    plt.xlabel("Principal Component 1", fontsize=10)
    plt.ylabel("Principal Component 2", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=9, title="Entity Group")

    plot_path = os.path.join(LATEST_RUN_DIR, "extended_similarity_pca_clean.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Clean PCA plot saved to: {plot_path}")
    plt.show()