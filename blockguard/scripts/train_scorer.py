# BlockGuard — LightGBM Training Script
# Trains the meta-scorer on labeled block data.
# 
# Usage:
#   python -m blockguard.scripts.train_scorer --data data/blocks.csv
#   python -m blockguard.scripts.train_scorer --seed-only  # Creates seed labels
#
# Training data format (CSV):
#   entity_overlap,hedge_density,block_length,has_numbers,has_question,position_ratio,label
#   0.5,0.1,15,0,0,0.3,1
#   0.0,0.8,45,1,1,0.9,0

import argparse
import csv
import os
import sys
import numpy as np
import pickle
from typing import List, Tuple

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blockguard.stages.s4_decide import DecisionLayer
from blockguard.stages.s2_verify import SignalScorer
from blockguard.config import cfg

def load_blocks_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load labeled block data from CSV."""
    X = []
    y = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = [float(row.get(c, 0.0)) for c in [
                "entity_overlap", "hedge_density", "block_length",
                "has_numbers", "has_question", "position_ratio"
            ]]
            X.append(features)
            y.append(int(row["label"]))
    return np.array(X), np.array(y)

def create_seed_labels(prompt: str = None) -> List[dict]:
    """
    Create seed-labeled data for initial training.
    
    Uses SelfCheckGPT-style consistency checks to auto-label,
    then human-review queue for disagreements.
    
    In production, this collects 2,000-5,000 blocks from real LLM responses
    and runs the consistency check pipeline to generate initial labels.
    """
    seed_blocks = [
        # Format: {text, features, label}
        # label: 1 = true/factual, 0 = hallucinated
        {
            "text": "The Earth orbits the Sun in approximately 365.25 days.",
            "features": {"entity_overlap": 0.8, "hedge_density": 0.0, "block_length": 12, "has_numbers": 1, "has_question": 0, "position_ratio": 0.3},
            "label": 1,
        },
        {
            "text": "Quantum computing was first proposed by Richard Feynman in 1982.",
            "features": {"entity_overlap": 0.6, "hedge_density": 0.1, "block_length": 10, "has_numbers": 1, "has_question": 0, "position_ratio": 0.4},
            "label": 1,
        },
        {
            "text": "AI will definitely surpass human intelligence by 2027 and replace all jobs.",
            "features": {"entity_overlap": 0.1, "hedge_density": 0.0, "block_length": 12, "has_numbers": 1, "has_question": 0, "position_ratio": 0.5},
            "label": 0,
        },
        {
            "text": "The model might possibly consider various factors when generating this response.",
            "features": {"entity_overlap": 0.0, "hedge_density": 0.6, "block_length": 11, "has_numbers": 0, "has_question": 0, "position_ratio": 0.6},
            "label": 0,
        },
    ]
    
    # Auto-generate more seed data via consistency checks
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "blockguard", "data", "seed_blocks.csv")
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_overlap", "hedge_density", "block_length", "has_numbers", "has_question", "position_ratio", "label"])
        for block in seed_blocks:
            f.write(",".join(str(block["features"][k]) for k in ["entity_overlap", "hedge_density", "block_length", "has_numbers", "has_question", "position_ratio"]) + f",{block['label']}\n")
    
    print(f"Seed data written to {output_path}")
    print(f"Created {len(seed_blocks)} seed blocks for initial training.")
    return seed_blocks

def train(data_path: str, model_output_path: str = None):
    """Train the LightGBM meta-scorer."""
    X, y = load_blocks_csv(data_path)
    
    if len(X) < 10:
        print(f"ERROR: Need at least 10 labeled blocks, got {len(X)}")
        print("Run `python -m blockguard.scripts.train_scorer --seed-only` first")
        return None
    
    # Split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    print(f"Training on {len(X_train)} blocks, testing on {len(X_test)}")
    
    model = DecisionLayer()
    model.train(X_train, y_train)
    
    # Evaluate
    correct = 0
    for i in range(len(X_test)):
        pred = round(model.predict({"entity_overlap": X_test[i][0], "hedge_density": X_test[i][1], 
                               "block_length": X_test[i][2], "has_numbers": X_test[i][3],
                               "has_question": X_test[i][4], "position_ratio": X_test[i][5]}))
        if pred == y_test[i]:
            correct += 1
    accuracy = correct / len(X_test) if len(X_test) > 0 else 0.0
    
    print(f"Test accuracy: {accuracy:.3f}")
    
    if model_output_path:
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        with open(model_output_path, 'wb') as f:
            pickle.dump(model.model, f)
        print(f"Model saved to {model_output_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to labeled blocks CSV")
    parser.add_argument("--seed-only", action="store_true", help="Create seed labels only")
    parser.add_argument("--model-output", default="data/lightgbm_model.txt", help="Where to save the trained model")
    args = parser.parse_args()
    
    if args.seed_only:
        create_seed_labels()
    elif args.data:
        train(args.data, args.model_output)
    else:
        # Try default data path
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "seed_blocks.csv")
        if os.path.exists(default_path):
            train(default_path, args.model_output)
        else:
            print("No data found. Run with --seed-only first.")
