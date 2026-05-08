"""
visualise.py

Local visualization utility for the CNN lab test dataset.
This file is NOT meant for submission or autograding.

What it does:
- loads train_signals.npy and train_labels.npy from ./data by default
- prints dataset shapes and label counts
- saves sample signal plots for each class
- saves a combined grid of examples across classes
- saves a plot of the class-wise mean signals

Usage:
    python visualise.py
    python visualise.py --data_dir data --out_dir visualisations
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_data(data_dir: Path):
    train_x = np.load(data_dir / "train_signals.npy")
    train_y = np.load(data_dir / "train_labels.npy")
    return train_x, train_y


def plot_label_distribution(y: np.ndarray, out_dir: Path):
    labels, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts)
    plt.xticks(labels)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("Training label distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "label_distribution.png", dpi=160)
    plt.close()


def plot_examples_per_class(X: np.ndarray, y: np.ndarray, out_dir: Path, n_examples: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    classes = sorted(np.unique(y).tolist())

    for cls in classes:
        idx = np.where(y == cls)[0]
        chosen = rng.choice(idx, size=min(n_examples, len(idx)), replace=False)

        fig, axes = plt.subplots(len(chosen), 1, figsize=(
            10, 1.8 * len(chosen)), squeeze=False)
        fig.suptitle(f"Example signals for class {cls}", fontsize=12)

        for row, i in enumerate(chosen):
            ax = axes[row, 0]
            ax.plot(X[i])
            ax.set_xlim(0, X.shape[1] - 1)
            ax.set_ylabel(f"#{i}")
            if row == len(chosen) - 1:
                ax.set_xlabel("Time index")

        plt.tight_layout()
        plt.savefig(out_dir / f"class_{cls}_examples.png", dpi=160)
        plt.close()


def plot_combined_grid(X: np.ndarray, y: np.ndarray, out_dir: Path, per_class: int = 3, seed: int = 1):
    rng = np.random.default_rng(seed)
    classes = sorted(np.unique(y).tolist())

    rows = len(classes)
    cols = per_class
    fig, axes = plt.subplots(rows, cols, figsize=(
        4 * cols, 2.2 * rows), squeeze=False)
    fig.suptitle("Sample signals across all classes", fontsize=13)

    for r, cls in enumerate(classes):
        idx = np.where(y == cls)[0]
        chosen = rng.choice(idx, size=min(per_class, len(idx)), replace=False)

        for c in range(cols):
            ax = axes[r, c]
            if c < len(chosen):
                i = chosen[c]
                ax.plot(X[i])
                ax.set_title(f"class {cls}, idx {i}", fontsize=9)
                ax.set_xlim(0, X.shape[1] - 1)
            else:
                ax.axis("off")

            if r == rows - 1:
                ax.set_xlabel("Time index")
            if c == 0:
                ax.set_ylabel(f"class {cls}")

    plt.tight_layout()
    plt.savefig(out_dir / "combined_examples_grid.png", dpi=160)
    plt.close()


def plot_class_means(X: np.ndarray, y: np.ndarray, out_dir: Path):
    classes = sorted(np.unique(y).tolist())
    plt.figure(figsize=(10, 5))

    for cls in classes:
        cls_x = X[y == cls]
        mean_signal = cls_x.mean(axis=0)
        plt.plot(mean_signal, label=f"class {cls}")

    plt.xlim(0, X.shape[1] - 1)
    plt.xlabel("Time index")
    plt.ylabel("Mean signal value")
    plt.title("Class-wise mean signals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "class_mean_signals.png", dpi=160)
    plt.close()


def plot_class_mean_std_band(X: np.ndarray, y: np.ndarray, out_dir: Path):
    classes = sorted(np.unique(y).tolist())

    for cls in classes:
        cls_x = X[y == cls]
        mean_signal = cls_x.mean(axis=0)
        std_signal = cls_x.std(axis=0)

        plt.figure(figsize=(10, 4))
        x_axis = np.arange(X.shape[1])
        plt.plot(x_axis, mean_signal, label=f"class {cls} mean")
        plt.fill_between(x_axis, mean_signal - std_signal,
                         mean_signal + std_signal, alpha=0.25)
        plt.xlim(0, X.shape[1] - 1)
        plt.xlabel("Time index")
        plt.ylabel("Signal value")
        plt.title(f"Mean ± std signal for class {cls}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"class_{cls}_mean_std.png", dpi=160)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing train_signals.npy and train_labels.npy")
    parser.add_argument("--out_dir", type=str,
                        default="visualisations", help="Directory to save the plots")
    parser.add_argument("--n_examples", type=int, default=6,
                        help="Number of example signals to save per class")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y = load_data(data_dir)

    print("Loaded training data")
    print("train_signals shape:", train_x.shape, "dtype:", train_x.dtype)
    print("train_labels shape :", train_y.shape, "dtype:", train_y.dtype)

    labels, counts = np.unique(train_y, return_counts=True)
    print("Label counts:")
    for label, count in zip(labels, counts):
        print(f"  class {label}: {count}")

    # plot_label_distribution(train_y, out_dir)
    plot_examples_per_class(train_x, train_y, out_dir,
                            n_examples=args.n_examples, seed=0)
    plot_combined_grid(train_x, train_y, out_dir, per_class=3, seed=1)
    # plot_class_means(train_x, train_y, out_dir)
    # plot_class_mean_std_band(train_x, train_y, out_dir)

    print(f"\nSaved plots to: {out_dir.resolve()}")
    print("Generated files:")
    for p in sorted(out_dir.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
