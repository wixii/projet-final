import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def load_original_data(data_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(data_path)
    # for alignment with outputs files
    X = df.drop(columns=["city_name", "country"], errors="ignore")
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


def align_embedding_to_original(df_original: pd.DataFrame, emb: pd.DataFrame, method_name: str) -> np.ndarray:
    """Returns X_2d aligned to df_original order."""
    original_names = df_original["city_name"].astype(str)
    emb = emb.set_index("city_name").reindex(original_names)

    if emb.isna().any().any():
        missing_count = int(emb.isna().any(axis=1).sum())
        raise ValueError(
            f"{method_name}: {missing_count} cities missing in embedding. "
            f"Ensure embedding contains exactly the same city_name values as the dataset."
        )

    return emb[["x", "y"]].to_numpy(dtype=float)


def score_trustworthiness(X_scaled: np.ndarray, X_2d: np.ndarray, n_neighbors: int) -> float:
    # Trustworthiness expects X (HD) and X_embedded (LD)
    return float(trustworthiness(X_scaled, X_2d, n_neighbors=n_neighbors))


def score_knn_accuracy(y: np.ndarray, X_2d: np.ndarray, n_neighbors: int) -> tuple[float, float]:
    """kNN accuracy on the 2D embedding with 5-fold stratified CV. Returns (mean, std)."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(knn, X_2d, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dimensionality reduction methods using trustworthiness + kNN accuracy."
    )
    parser.add_argument("--data", default="data/city_lifestyle_dataset.csv", help="Path to original CSV dataset")
    parser.add_argument("--outputs", default="outputs", help="Directory containing *_emb_2d.csv files")
    parser.add_argument(
        "--neighbors",
        type=int,
        default=10,
        help="n_neighbors for trustworthiness AND kNN (default: 10)",
    )
    parser.add_argument(
        "--label",
        default="country",
        help="Label column for kNN accuracy (default: country). Set to '' to disable kNN.",
    )

    # ✅ Bonus: allow user to choose which methods to evaluate
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["PCA", "TSNE", "UMAP"],
        help="Methods to evaluate (any of: PCA TSNE UMAP). Example: --methods PCA TSNE",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.outputs)

    df_original, X_scaled = load_original_data(data_path)

    # Labels for kNN (enabled if label col exists and not disabled)
    do_knn = bool(args.label) and (args.label in df_original.columns)
    y = None
    if do_knn:
        y = df_original[args.label].astype("category").cat.codes.to_numpy()

    all_methods = {
        "PCA": out_dir / "pca_emb_2d.csv",
        "TSNE": out_dir / "tsne_emb_2d.csv",
        "UMAP": out_dir / "umap_emb_2d.csv",
    }

    # Filter methods based on user selection (case-insensitive)
    wanted = {m.upper() for m in args.methods}
    methods = {k: v for k, v in all_methods.items() if k.upper() in wanted}

    if not methods:
        raise ValueError(
            f"No valid methods selected: {args.methods}. "
            f"Choose among: {list(all_methods.keys())}"
        )

    print(f"Dataset: {data_path} | n={len(df_original)}")
    print(f"Selected methods: {', '.join(methods.keys())}")
    print(f"Trustworthiness with n_neighbors={args.neighbors}")
    if do_knn:
        n_classes = int(pd.Series(y).nunique())
        print(f"kNN accuracy on 2D embedding (label={args.label}, classes={n_classes}, 5-fold CV)\n")
    else:
        print("kNN accuracy: skipped (label column missing or disabled)\n")

    results: list[dict] = []

    for name, path in methods.items():
        if not path.exists():
            print(f"- {name}: SKIPPED (missing file: {path})")
            continue

        emb = load_embedding(path)
        X_2d = align_embedding_to_original(df_original, emb, name)

        tw = score_trustworthiness(X_scaled, X_2d, n_neighbors=args.neighbors)

        if do_knn and y is not None:
            acc_mean, acc_std = score_knn_accuracy(y, X_2d, n_neighbors=args.neighbors)
            print(f"- {name}: trustworthiness={tw:.4f} | knn_acc={acc_mean:.4f} (±{acc_std:.4f})")
            results.append({"method": name, "trustworthiness": tw, "knn_acc": acc_mean, "knn_acc_std": acc_std})
        else:
            print(f"- {name}: trustworthiness={tw:.4f}")
            results.append({"method": name, "trustworthiness": tw})

    if results:
        best_tw = max(results, key=lambda d: d["trustworthiness"])
        print(f"\nBest trustworthiness: {best_tw['method']} ({best_tw['trustworthiness']:.4f})")

        if do_knn and any("knn_acc" in r for r in results):
            best_knn = max((r for r in results if "knn_acc" in r), key=lambda d: d["knn_acc"])
            print(f"Best kNN accuracy: {best_knn['method']} ({best_knn['knn_acc']:.4f})")
    else:
        print("No embeddings found. Expected files like outputs/pca_emb_2d.csv")


if __name__ == "__main__":
    main()
