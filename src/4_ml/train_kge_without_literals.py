"""
Filtered Knowledge Graph Embeddings Training Pipeline.
This script trains a RotatE embedding model using PyKEEN. 
Crucially, it filters out literal values (like exact coordinates, prices, and labels) 
before training, ensuring the model focuses purely on the structural topology 
of the graph rather than learning meaningless embeddings for unique float/string values.
"""

import os
import time
import numpy as np
import torch
from rdflib import Graph

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# ==============================================================================
# Dynamic Path Resolution & Timestamped Output Folder
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

MODEL_NAME = "RotatE"
EMBEDDING_DIM = 64  # Reduced dimensionality compared to previous runs for faster convergence
EPOCHS = 700

# Create timestamp-based subdirectory to organize experimental runs without overwriting
timestamp = time.strftime("%Y%m%d_%H%M%S")
RUN_FOLDER_NAME = f"run_{timestamp}_{MODEL_NAME}_dim{EMBEDDING_DIM}"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", RUN_FOLDER_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

script_start_time = time.time()
print(
    f"[{time.strftime('%H:%M:%S')}] Starting Knowledge Graph Embeddings"
    " Training Pipeline (Filtered Structural Graph)..."
)
print(f"[{time.strftime('%H:%M:%S')}] Output Directory: {OUTPUT_DIR}")


# ==============================================================================
# 1. Input Path Definition
# ==============================================================================
INPUT_TTL_PATH = os.path.join(
    PROJECT_ROOT, "src", "3_logic", "inferred_vienna_kg.ttl"
)


# ==============================================================================
# 2. Load Triples, Filter Literals, & Convert to PyKEEN TriplesFactory
# ==============================================================================
print(
    f"[{time.strftime('%H:%M:%S')}] Parsing Turtle file via RDFLib from"
    f" {INPUT_TTL_PATH}..."
)

g = Graph()
g.parse(INPUT_TTL_PATH, format="turtle")

# Define literal predicates to filter out.
# WHY FILTER? Knowledge Graph Embedding (KGE) models like RotatE are designed to learn 
# latent structural relationships between entities. If we include literal values 
# (e.g., exact prices like "1250.50" or latitudes like "48.2082"), the model treats 
# every unique number as a distinct, disconnected node in the graph. This creates 
# extreme sparsity and degrades the model's ability to learn meaningful graph topology.
LITERAL_PREDICATES = {
    "http://vienna-realestate.org/kg/hasPrice",
    "http://vienna-realestate.org/kg/hasSize",
    "http://vienna-realestate.org/kg/hasRooms",
    "http://vienna-realestate.org/kg/latitude",
    "http://vienna-realestate.org/kg/longitude",
    "http://vienna-realestate.org/kg/timeToCenterMinutes",
    "http://vienna-realestate.org/kg/timeToEducationHubMinutes",
    "http://vienna-realestate.org/kg/timeToMainStationMinutes",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://vienna-realestate.org/kg/willhabenId"
}

triples_list = []
for s, p, o in g:
    # Only keep the triple if the predicate is structural (i.e., not in the literal list).
    # This leaves relations like 'isNearStop', 'hasAlternativeFlatInProximity', 
    # 'hasConnection', and 'RDF.type'.
    if str(p) not in LITERAL_PREDICATES:
        triples_list.append([str(s), str(p), str(o)])

triples_array = np.array(triples_list, dtype=str)
triples_factory = TriplesFactory.from_labeled_triples(triples_array)

# Automatically split the dataset into training (80%), testing (10%), and validation (10%) sets
# using a fixed random seed to ensure reproducible splits across different runs.
training_factory, testing_factory, validation_factory = (
    triples_factory.split(ratios=[0.8, 0.1, 0.1], random_state=42)
)

print(f"[{time.strftime('%H:%M:%S')}] Filtered Triples Loaded Successfully:")
print(f"  - Total Structural Triples: {triples_factory.num_triples}")
print(f"  - Training Triples: {training_factory.num_triples}")
print(f"  - Testing Triples: {testing_factory.num_triples}")
print(f"  - Validation Triples: {validation_factory.num_triples}")
print(f"  - Unique Entities: {triples_factory.num_entities}")
print(f"  - Unique Relations: {triples_factory.num_relations}")


# ==============================================================================
# 3. Train Knowledge Graph Embedding (KGE) Model
# ==============================================================================
print(
    f"[{time.strftime('%H:%M:%S')}] Training {MODEL_NAME} model for"
    f" {EPOCHS} epochs..."
)

# Execute the PyKEEN training pipeline over a fixed number of epochs
result = pipeline(
    training=training_factory,
    testing=testing_factory,
    validation=validation_factory,
    model=MODEL_NAME,
    model_kwargs=dict(embedding_dim=EMBEDDING_DIM),
    
    # Loss for RotatE: Negative Sampling Self-Adversarial Loss (NSSALoss)
    loss="NSSALoss",
    loss_kwargs=dict(margin=9.0, adversarial_temperature=1.0),
    
    # Optimizer & Learning Rate
    optimizer="Adam",
    optimizer_kwargs=dict(lr=0.0005),
    
    # Negative Sampling: Generate 32 corrupted (false) triples for every true triple
    # This teaches the model to push false relationships apart in the vector space.
    negative_sampler="basic",
    negative_sampler_kwargs=dict(num_negs_per_pos=32),
    
    training_kwargs=dict(num_epochs=EPOCHS, batch_size=256),
    evaluation_kwargs=dict(batch_size=256),
    random_seed=42,
)

# Save pipeline results (model weights, configuration, metrics) into the timestamped directory
result.save_to_directory(OUTPUT_DIR)
print(f"[{time.strftime('%H:%M:%S')}] Model & Results saved to: {OUTPUT_DIR}")


# ==============================================================================
# 4. Extract and Display Evaluation Metrics
# ==============================================================================
# Retrieve link prediction ranking metrics calculated on the testing set.
mrr = result.metric_results.get_metric("mean_reciprocal_rank")
hits_1 = result.metric_results.get_metric("hits_at_1")
hits_5 = result.metric_results.get_metric("hits_at_5")
hits_10 = result.metric_results.get_metric("hits_at_10")

print("\n" + "=" * 50)
print("EVALUATION METRICS (Link Prediction):")
print(f"  - Mean Reciprocal Rank (MRR): {mrr:.4f}")
if hits_1 is not None:
    print(f"  - Hits@1:  {hits_1:.4f}")
if hits_5 is not None:
    print(f"  - Hits@5:  {hits_5:.4f}")
if hits_10 is not None:
    print(f"  - Hits@10: {hits_10:.4f}")
print("=" * 50 + "\n")


# ==============================================================================
# 5. Extract 5 Embedding Examples for Portfolio Report
# ==============================================================================
print(
    f"[{time.strftime('%H:%M:%S')}] Extracting 5 Embedding Examples for"
    " Portfolio Report..."
)
model = result.model
entity_to_id = training_factory.entity_to_id

# Grab the first 5 entity labels from the mapping dictionary
sample_entities = list(entity_to_id.keys())[:5]

print("\n--- 5 EMBEDDING VECTOR EXAMPLES ---")
for i, entity in enumerate(sample_entities, 1):
    entity_id = entity_to_id[entity]
    
    # 1. Map entity string to internal PyKEEN ID.
    # 2. Query the PyTorch model for the embedding vector on the appropriate device.
    # 3. Detach from the gradient graph and move to CPU memory.
    # 4. Convert to a NumPy array for readable output.
    embedding_vector = (
        model.entity_representations[0](
            torch.tensor([entity_id], device=model.device)
        )
        .detach()
        .cpu()
        .numpy()[0]
    )
    
    # RotatE models relationships as rotations in the complex plane. 
    # Therefore, the underlying embeddings are arrays of complex numbers.
    if np.iscomplexobj(embedding_vector):
        # Format the complex numbers displaying both Real and Imaginary parts
        vector_preview = [f"{val.real:.3f}+{val.imag:.3f}j" for val in embedding_vector[:5]]
    else:
        # Fallback if a different model without complex numbers is used in the future
        vector_preview = [round(float(val), 4) for val in embedding_vector[:5]]
        
    print(
        f"Example {i}: Entity <{entity}>\n  -> Vector Shape:"
        f" {embedding_vector.shape}, First 5 Dimensions: {vector_preview}\n"
    )

total_duration = time.time() - script_start_time
print(
    f"\n[{time.strftime('%H:%M:%S')}] ML Pipeline Execution Completed in"
    f" {total_duration:.2f} seconds!"
)