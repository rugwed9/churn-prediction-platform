"""Training script - runs the full ML pipeline.

Usage:
    python scripts/train.py [--config configs/default.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import create_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run churn prediction training pipeline")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    pipeline = create_pipeline(args.config)

    print("=" * 60)
    print("Churn Prediction Platform - Training Pipeline")
    print("=" * 60)

    results = pipeline.run()

    print(f"\nDataset: {results['dataset']['n_samples']} samples, "
          f"{results['dataset']['n_features']} features, "
          f"{results['dataset']['churn_rate']:.1%} churn rate")

    print(f"\n{'Model':<15} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8} {'Recall':>8}")
    print("-" * 55)
    for name, metrics in results["models"].items():
        print(f"{name:<15} {metrics['auc_roc']:>8.4f} {metrics['auc_pr']:>8.4f} "
              f"{metrics['f1']:>8.4f} {metrics['recall']:>8.4f}")

    print(f"\nTop features:")
    for feat, imp in list(results["feature_importance"].items())[:5]:
        print(f"  {feat:<30} {imp:.4f}")

    print(f"\nPipeline completed in {results['pipeline_duration_seconds']}s")
    print(f"Models saved to models/registry/")

    # Save results
    output = Path("data/processed/training_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
