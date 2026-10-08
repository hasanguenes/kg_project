# Code Pipeline Overview

This folder contains the Python scripts responsible for Knowledge Graph Embedding (KGE) training, evaluation, price regression, and map data generation.

## Script Descriptions

* **`train_kge_without_literals.py`**: Parses the inferred Turtle graph, filters out literal attributes (like prices and coordinates) to focus purely on structural topology, and trains a RotatE embedding model via PyKEEN[cite: 3].
* **`train_kge.py`**: Alternative training script that includes literal values or standard configurations for baseline comparisons.
* **`train_regressor.py`**: Implements the extended late fusion approach by flattening complex entity vectors and training a Random Forest Regressor to predict property prices[cite: 5].
* **`predict_prices.py`**: Uses the trained regressor and KGE models to estimate and evaluate prices for residential properties.
* **`predict_prices_incl_literals.py`**: Extended price prediction script incorporating literal attributes alongside structural features.
* **`evaluate_embeddings_pca.py`**: Performs Principal Component Analysis (PCA) on the complex embedding vectors to visualize structural groupings and class separation.
* **`evaluate_tp_fp_link_predictions.py`**: Evaluates link prediction performance by analyzing true positive and false positive relations (such as `isNearStop`).
* **`export_embedding_examples.py`**: Extracts representative complex embedding vectors (real and imaginary parts) for portfolio documentation and reporting.
* **`generate_map_data.py`**: Combines ML price predictions, KGE vectors, and semantic graph rules into a structured JSON payload (`map_data.json`) for frontend map rendering[cite: 4].