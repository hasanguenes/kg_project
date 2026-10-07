"""
Knowledge Graph Embeddings Training and Link Prediction Pipeline.
This script uses PyKEEN to train a RotatE embedding model on the inferred Vienna 
Real Estate Knowledge Graph. It includes dataset splitting, model training with 
early stopping, evaluation, embedding extraction, and link prediction.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from rdflib import Graph

from pykeen.pipeline import pipeline
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory

# ---------------------------------------------------------
# Dynamic Path Resolution & Timestamped Output Folder
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

MODEL_NAME = "RotatE"
EMBEDDING_DIM = 256
EPOCHS = 500

# Create a timestamp-based subfolder (e.g., run_20261006_161500_RotatE_dim256)
# This prevents overwriting previous model runs and keeps experiments organized
timestamp = time.strftime("%Y%m%d_%H%M%S")
RUN_FOLDER_NAME = f"run_{timestamp}_{MODEL_NAME}_dim{EMBEDDING_DIM}"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", RUN_FOLDER_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

script_start_time = time.time()
print(
    f"[{time.strftime('%H:%M:%S')}] Starting Knowledge Graph Embeddings"
    " Training Pipeline..."
)
print(f"[{time.strftime('%H:%M:%S')}] Output Directory: {OUTPUT_DIR}")

# ---------------------------------------------------------
# 1. Input Path Definition
# ---------------------------------------------------------
INPUT_TTL_PATH = os.path.join(
    PROJECT_ROOT, "src", "3_logic", "inferred_vienna_kg.ttl"
)

# ---------------------------------------------------------
# 2. Load Triples using RDFLib and convert to PyKEEN TriplesFactory
# ---------------------------------------------------------
print(
    f"[{time.strftime('%H:%M:%S')}] Parsing Turtle file via RDFLib from"
    f" {INPUT_TTL_PATH}..."
)

g = Graph()
g.parse(INPUT_TTL_PATH, format="turtle")

# Extract all triples from the RDFLib graph into a basic string-based list
triples_list = [[str(s), str(p), str(o)] for s, p, o in g]
triples_array = np.array(triples_list, dtype=str)

# Convert the string array into a PyKEEN TriplesFactory.
# This assigns unique integer IDs to every entity and relation, which is 
# required for the PyTorch embedding layers.
triples_factory = TriplesFactory.from_labeled_triples(triples_array)

# Automatically split the dataset into training (80%), testing (10%), and validation (10%) sets
# using a fixed random seed to ensure reproducible splits across different runs.
training_factory, testing_factory, validation_factory = (
    triples_factory.split(ratios=[0.8, 0.1, 0.1], random_state=42)
)

print(f"[{time.strftime('%H:%M:%S')}] Triples Loaded Successfully:")
print(f"  - Total Triples: {triples_factory.num_triples}")
print(f"  - Training Triples: {training_factory.num_triples}")
print(f"  - Testing Triples: {testing_factory.num_triples}")
print(f"  - Validation Triples: {validation_factory.num_triples}")
print(f"  - Unique Entities: {triples_factory.num_entities}")
print(f"  - Unique Relations: {triples_factory.num_relations}")

# # ---------------------------------------------------------
# # 3. Train Knowledge Graph Embedding (KGE) Model (Old Configuration)
# # ---------------------------------------------------------
# print(
#     f"[{time.strftime('%H:%M:%S')}] Training {MODEL_NAME} model for"
#     f" {EPOCHS} epochs..."
# )

# result = pipeline(
#     training=training_factory,
#     testing=testing_factory,
#     validation=validation_factory,
#     model=MODEL_NAME,
#     model_kwargs=dict(embedding_dim=EMBEDDING_DIM),
    
#     # Loss function for RotatE (Negative Sampling Self-Adversarial Loss)
#     loss="NSSALoss",
#     loss_kwargs=dict(margin=9.0, adversarial_temperature=1.0),
    
#     # Optimizer & Learning Rate
#     optimizer="Adam",
#     optimizer_kwargs=dict(lr=0.0005),
    
#     # Negative Sampling
#     negative_sampler="basic",
#     negative_sampler_kwargs=dict(num_negs_per_pos=32),
    
#     training_kwargs=dict(num_epochs=EPOCHS, batch_size=256),
#     evaluation_kwargs=dict(batch_size=256),
#     random_seed=42,
# )

# ---------------------------------------------------------
# 3. Train Knowledge Graph Embedding (KGE) Model (Active Configuration)
# ---------------------------------------------------------
MODEL_NAME = "RotatE"
EMBEDDING_DIM = 256
EPOCHS = 1000  # Maximum number of epochs allowed

print(
    f"[{time.strftime('%H:%M:%S')}] Training {MODEL_NAME} model for"
    f" {EPOCHS} epochs (Evaluating every 100 epochs)..."
)

# The PyKEEN pipeline handles the complete lifecycle: 
# Training loop, negative sampling, evaluation on test set, and early stopping.
result = pipeline(
    training=training_factory,
    testing=testing_factory,
    validation=validation_factory,
    model=MODEL_NAME,
    model_kwargs=dict(embedding_dim=EMBEDDING_DIM),
    
    # Loss function for RotatE: Negative Sampling Self-Adversarial Loss (NSSALoss)
    # Often yields the best performance for RotatE models by dynamically weighting harder negative samples.
    loss="NSSALoss",
    loss_kwargs=dict(margin=9.0, adversarial_temperature=1.0),
    
    # Optimizer & Learning Rate
    optimizer="Adam",
    optimizer_kwargs=dict(lr=0.0005),
    
    # Negative Sampling: Generate 32 false triples for every true triple to teach the model what NOT to predict
    negative_sampler="basic",
    negative_sampler_kwargs=dict(num_negs_per_pos=32),
    
    # Early Stopper: Automatically evaluates the model against the validation set to prevent overfitting
    stopper="early",
    stopper_kwargs=dict(
        frequency=100,       # Evaluate and print logs EVERY 100 EPOCHS
        patience=3,          # Stop training if no progress is made after 3 consecutive checks (i.e., 300 epochs)
        relative_delta=0.002 # Minimum required improvement of the MRR metric (0.2%) to reset patience
    ),
    
    training_kwargs=dict(num_epochs=EPOCHS, batch_size=256),
    evaluation_kwargs=dict(batch_size=256),
    random_seed=42,
)

# Save pipeline results (model checkpoints, metadata, and evaluation metrics) into the timestamped directory
result.save_to_directory(OUTPUT_DIR)
print(f"[{time.strftime('%H:%M:%S')}] Model & Results saved to: {OUTPUT_DIR}")

# ---------------------------------------------------------
# 4. Extract and Display Evaluation Metrics
# ---------------------------------------------------------
# The test set was evaluated automatically at the end of the pipeline.
# Extract standard ranking metrics from the result object.
mrr = result.metric_results.get_metric("mean_reciprocal_rank")
hits_1 = result.metric_results.get_metric("hits_at_1")
hits_5 = result.metric_results.get_metric("hits_at_5")
hits_10 = result.metric_results.get_metric("hits_at_10")

print("\n" + "=" * 50)
print("EVALUATION METRICS (Link Prediction):")
print(f"  - Mean Reciprocal Rank (MRR): {mrr:.4f}")
print(f"  - Hits@1:  {hits_1:.4f}")
print(f"  - Hits@5:  {hits_5:.4f}")
print(f"  - Hits@10: {hits_10:.4f}")
print("=" * 50 + "\n")

# ---------------------------------------------------------
# 5. Extract 5 concrete examples of Node representations
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
    
    # Process to extract raw vector values from the PyTorch model:
    # 1. Move the ID to a tensor on the correct device (CPU/GPU)
    # 2. Query the embedding layer
    # 3. .detach(): Remove tensor from computational graph (no gradients needed)
    # 4. .cpu(): Move tensor back to CPU memory (if it was on GPU)
    # 5. .numpy(): Convert to standard NumPy array
    embedding_vector = (
        model.entity_representations[0](
            torch.tensor([entity_id], device=model.device)
        )
        .detach()
        .cpu()
        .numpy()[0]
    )
    
    # RotatE models relationships as rotations in the complex plane, meaning 
    # the resulting embeddings contain complex numbers (real + imaginary parts).
    if np.iscomplexobj(embedding_vector):
        # Show Real (Re) and Imaginary (Im) parts explicitly
        vector_preview = [f"{val.real:.3f}+{val.imag:.3f}j" for val in embedding_vector[:5]]
    else:
        vector_preview = [round(float(val), 4) for val in embedding_vector[:5]]
        
    print(
        f"Example {i}: Entity <{entity}>\n  -> Vector Shape:"
        f" {embedding_vector.shape}, First 5 Dimensions: {vector_preview}\n"
    )

# ---------------------------------------------------------
# 6. Link Prediction for KG Completion
# ---------------------------------------------------------
# Here we use the trained model to infer new links (edges) that do not currently exist in the graph.
print(
    f"[{time.strftime('%H:%M:%S')}] Performing Link Prediction for KG"
    " Completion..."
)

target_relation = "http://vienna-realestate.org/kg/isNearStop"

if target_relation in training_factory.relation_to_id:
    print(f"Predicting missing links for relation: {target_relation}")

    # 1. Retrieve a specific Head-Entity (e.g., a flat) directly from the training data
    rel_id = training_factory.relation_to_id[target_relation]
    sample_head_id = training_factory.mapped_triples[
        training_factory.mapped_triples[:, 1] == rel_id
    ][0, 0].item()
    sample_head = training_factory.entity_id_to_label[sample_head_id]

    # 2. Retrieve the target predictions as a pandas DataFrame.
    # The model ranks all possible Tail-Entities based on plausibility.
    prediction_df = predict_target(
        model=model,
        head=sample_head,
        relation=target_relation,
        triples_factory=training_factory,
    ).df

    # 3. Filter: Only display predicted edges (links) that were NOT already present in the training set
    if "in_training" in prediction_df.columns:
        new_predictions = prediction_df[~prediction_df["in_training"]].head(10)
    else:
        new_predictions = prediction_df.head(10)

    print(f"\n--- TOP PREDICTED TARGET STOPS FOR HEAD <{sample_head}> ---")
    for idx, row in new_predictions.iterrows():
        tail_val = row.get("tail_label", row.get("target_label", "Unknown"))
        score_val = row.get("score", 0.0)
        print(
            f"Head: {sample_head} | Relation: isNearStop | Tail:"
            f" {tail_val} | Score: {score_val:.4f}"
        )
else:
    print(f"Warning: Relation '{target_relation}' not found in the training dataset.")

total_duration = time.time() - script_start_time
print(
    f"\n[{time.strftime('%H:%M:%S')}] ML Pipeline Execution Completed in"
    f" {total_duration:.2f} seconds!"
)