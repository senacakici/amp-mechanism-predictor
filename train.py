"""
train.py
--------
Train and save multi-label AMP mechanism classifiers.

Usage:
    python src/models/train.py --model binary_relevance --features physicochemical
    python src/models/train.py --model classifier_chain --features all
    python src/models/train.py --model mlp --features all
"""

import argparse
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.features import featurize_dataframe, AMINO_ACIDS
from evaluation.metrics import evaluate_multilabel

MECHANISM_LABELS = [
    "membrane_disruption",
    "membrane_depolarization",
    "cell_wall_synthesis",
    "protein_synthesis",
    "dna_rna_targeting",
    "cell_division",
    "metabolic_disruption",
    "immunomodulatory",
]

DATA_PATH = Path(__file__).parent.parent.parent / "data/processed/amp_multilabel_dataset.csv"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def build_model(model_type: str):
    """Return a scikit-learn multi-label classifier."""
    base_estimators = {
        "lr": LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    }
    base = base_estimators["rf"]  # default

    if model_type == "binary_relevance":
        # Independent classifier per label
        return MultiOutputClassifier(base, n_jobs=-1)
    
    elif model_type == "classifier_chain":
        # Each label classifier uses previous predictions as features
        return ClassifierChain(
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            order="random",
            random_state=42,
        )
    
    elif model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        # Multi-output MLP
        return MultiOutputClassifier(
            MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            ),
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_feature_columns(X: pd.DataFrame, feature_set: str) -> list:
    """Select feature columns based on feature set name."""
    if feature_set == "physicochemical":
        return [c for c in X.columns if not c.startswith("AAC_") and not c.startswith("DPC_")]
    elif feature_set == "aac":
        return [c for c in X.columns if c.startswith("AAC_")]
    elif feature_set == "all":
        return list(X.columns)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")


def main(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading dataset from {DATA_PATH}")
    if not DATA_PATH.exists():
        print("Dataset not found. Run: python src/data/prepare_dataset.py")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} samples")

    # --- Featurize ---
    print("Extracting features...")
    X_raw = featurize_dataframe(df, seq_col="sequence", include_dpc=False)
    y = df[MECHANISM_LABELS].values

    feat_cols = get_feature_columns(X_raw, args.features)
    X = X_raw[feat_cols].values
    print(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Label matrix:   {y.shape[0]} samples × {y.shape[1]} labels")

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Build pipeline ---
    clf = build_model(args.model)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])

    # --- Train ---
    print(f"\nTraining {args.model} classifier...")
    pipeline.fit(X_train, y_train)
    print("  Training complete.")

    # --- Evaluate ---
    print("\nEvaluating on test set...")
    y_pred = pipeline.predict(X_test)
    metrics = evaluate_multilabel(y_test, y_pred, label_names=MECHANISM_LABELS)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  Hamming Loss:          {metrics['hamming_loss']:.4f}")
    print(f"  Subset Accuracy:       {metrics['subset_accuracy']:.4f}")
    print(f"  Macro F1:              {metrics['macro_f1']:.4f}")
    print(f"  Micro F1:              {metrics['micro_f1']:.4f}")
    print(f"  Label Ranking AP:      {metrics['lrap']:.4f}")
    print("\nPer-label F1:")
    for label, f1 in metrics["per_label_f1"].items():
        print(f"  {label:<30} {f1:.4f}")

    # --- Save ---
    model_name = f"{args.model}_{args.features}"
    model_path = RESULTS_DIR / f"{model_name}.pkl"
    metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"

    with open(model_path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "feature_cols": feat_cols}, f)

    with open(metrics_path, "w") as f:
        # Convert numpy types for JSON serialisation
        json_metrics = {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in metrics.items()}
        json.dump(json_metrics, f, indent=2)

    print(f"\nModel saved to:   {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AMP mechanism classifier")
    parser.add_argument(
        "--model",
        choices=["binary_relevance", "classifier_chain", "mlp"],
        default="binary_relevance",
        help="Multi-label classification strategy",
    )
    parser.add_argument(
        "--features",
        choices=["physicochemical", "aac", "all"],
        default="all",
        help="Feature set to use",
    )
    args = parser.parse_args()
    main(args)
