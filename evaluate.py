import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler


def load_original_data(data_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(data_path)
    # for alignement with outputs files
    X = df.drop(columns=["city_name", "country"])
    X_scaled = StandardScaler().fit_transform(X)
    return df, X_scaled


def load_embedding(emb_path: Path) -> pd.DataFrame:
    emb = pd.read_csv(emb_path)

    required_cols = {"city_name", "x", "y"}
    missing = required_cols - set(emb.columns)
    if missing:
        raise ValueError(f"{emb_path.name} missing columns: {missing}. Expected at least {required_cols}")

    emb = emb[["city_name", "x", "y"]].copy()
    emb["city_name"] = emb["city_name"].astype(str)

    if emb["city_name"].duplicated().any():
        raise ValueError(f"{emb_path.name} has duplicated city_name values. Fix export to be unique.")

    return emb


def score_method(
    method_name: str,
    X_scaled: np.ndarray,
    df_original: pd.DataFrame,
    emb: pd.DataFrame,
    n_neighbors: int,
) -> float:
    original_names = df_original["city_name"].astype(str)
    emb = emb.set_index("city_name").reindex(original_names)

    if emb.isna().any().any():
        missing_count = int(emb.isna().any(axis=1).sum())
        raise ValueError(
            f"{method_name}: {missing_count} cities missing in embedding. "
            f"Ensure embedding contains exactly the same city_name values as the dataset."
        )

    X_2d = emb[["x", "y"]].to_numpy(dtype=float)

    # Trustworthiness attend X (HD) et X_embedded (LD)
    return float(trustworthiness(X_scaled, X_2d, n_neighbors=n_neighbors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dimensionality reduction methods using trustworthiness.")
    parser.add_argument("--data", default="data/city_lifestyle_dataset.csv", help="Path to original CSV dataset")
    parser.add_argument("--outputs", default="outputs", help="Directory containing *_emb_2d.csv files")
    parser.add_argument("--neighbors", type=int, default=10, help="n_neighbors for trustworthiness (default: 10)")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.outputs)

    df_original, X_scaled = load_original_data(data_path)

    methods = {
        "PCA": out_dir / "pca_emb_2d.csv",
        "TSNE": out_dir / "tsne_emb_2d.csv",
        "UMAP": out_dir / "umap_emb_2d.csv",
    }

    print(f"Dataset: {data_path} | n={len(df_original)}")
    print(f"Trustworthiness with n_neighbors={args.neighbors}\n")

    scores: list[tuple[str, float]] = []

    for name, path in methods.items():
        if not path.exists():
            print(f"- {name}: SKIPPED (missing file: {path})")
            continue

        emb = load_embedding(path)
        score = score_method(name, X_scaled, df_original, emb, n_neighbors=args.neighbors)
        scores.append((name, score))
        print(f"- {name}: {score:.4f}")

    if scores:
        best = max(scores, key=lambda t: t[1])
        print(f"\nBest: {best[0]} ({best[1]:.4f})")
    else:
        print("No embeddings found. Expected files like outputs/pca_emb_2d.csv")


if __name__ == "__main__":
    main()