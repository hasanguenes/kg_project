import os
import time
import numpy as np
import torch
from rdflib import Graph

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# ---------------------------------------------------------
# Dynamic Path Resolution & Timestamped Output Folder
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

MODEL_NAME = "RotatE"
EMBEDDING_DIM = 64
EPOCHS = 700

# Create timestamp-based subdirectory
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

# ---------------------------------------------------------
# 1. Input Path Definition
# ---------------------------------------------------------
INPUT_TTL_PATH = os.path.join(
    PROJECT_ROOT, "src", "3_logic", "inferred_vienna_kg.ttl"
)

# ---------------------------------------------------------
# 2. Load Triples, Filter Literals, & Convert to PyKEEN TriplesFactory
# ---------------------------------------------------------
print(
    f"[{time.strftime('%H:%M:%S')}] Parsing Turtle file via RDFLib from"
    f" {INPUT_TTL_PATH}..."
)

g = Graph()
g.parse(INPUT_TTL_PATH, format="turtle")

# Define literal predicates to filter out so KGE focuses purely on topology
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
    # Skip literal properties to prevent KGE from treating raw numbers as discrete entities
    if str(p) not in LITERAL_PREDICATES:
        triples_list.append([str(s), str(p), str(o)])

triples_array = np.array(triples_list, dtype=str)
triples_factory = TriplesFactory.from_labeled_triples(triples_array)

# Split dataset automatically (80/10/10)
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

# ---------------------------------------------------------
# 3. Train Knowledge Graph Embedding (KGE) Model
# ---------------------------------------------------------
print(
    f"[{time.strftime('%H:%M:%S')}] Training {MODEL_NAME} model for"
    f" {EPOCHS} epochs..."
)

result = pipeline(
    training=training_factory,
    testing=testing_factory,
    validation=validation_factory,
    model=MODEL_NAME,
    model_kwargs=dict(embedding_dim=EMBEDDING_DIM),
    
    # Loss for RotatE
    loss="NSSALoss",
    loss_kwargs=dict(margin=9.0, adversarial_temperature=1.0),
    
    # Optimizer & Learning Rate
    optimizer="Adam",
    optimizer_kwargs=dict(lr=0.0005),
    
    # Negative Sampling
    negative_sampler="basic",
    negative_sampler_kwargs=dict(num_negs_per_pos=32),
    
    training_kwargs=dict(num_epochs=EPOCHS, batch_size=256),
    evaluation_kwargs=dict(batch_size=256),
    random_seed=42,
)

# Save pipeline results into the timestamped directory
result.save_to_directory(OUTPUT_DIR)
print(f"[{time.strftime('%H:%M:%S')}] Model & Results saved to: {OUTPUT_DIR}")

# ---------------------------------------------------------
# 4. Extract and Display Evaluation Metrics
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 5. Extract 5 Embedding Examples for Portfolio Report
# ---------------------------------------------------------
print(
    f"[{time.strftime('%H:%M:%S')}] Extracting 5 Embedding Examples for"
    " Portfolio Report..."
)
model = result.model
entity_to_id = training_factory.entity_to_id

sample_entities = list(entity_to_id.keys())[:5]

print("\n--- 5 EMBEDDING VECTOR EXAMPLES ---")
for i, entity in enumerate(sample_entities, 1):
    entity_id = entity_to_id[entity]
    embedding_vector = (
        model.entity_representations[0](
            torch.tensor([entity_id], device=model.device)
        )
        .detach()
        .cpu()
        .numpy()[0]
    )
    
    if np.iscomplexobj(embedding_vector):
        vector_preview = [f"{val.real:.3f}+{val.imag:.3f}j" for val in embedding_vector[:5]]
    else:
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