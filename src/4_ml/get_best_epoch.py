import json

# Pfad zu den Pipeline-Ergebnissen
results_json_path = r"C:\Users\guene\OneDrive\Studium\Master\2. Semester\Knowledge Graphs\Portfolio\Projekt\kgcourse-project-starting-template-main\src\4_ml\results\run_20261006_225352_RotatE_dim256\results.json"

with open(results_json_path, "r") as f:
    data = json.load(f)

# Falls Stopper/Validation-Historie vorhanden ist:
if "stopper" in data:
    print("Stopper History:", data["stopper"])

# Zeigt alle Metriken der finalen Evaluation (Epoche 1000)
print("\nFinal Test Metrics:")
print("MRR:", data["metrics"]["both"]["realistic"]["inverse_harmonic_mean_rank"])
print("Hits@10:", data["metrics"]["both"]["realistic"]["hits_at_10"])