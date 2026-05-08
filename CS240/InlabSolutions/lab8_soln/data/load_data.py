import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def generate_moons_dataset(
	n_samples: int = 2000,
	noise: float = 0.2,
	test_size: float = 0.2,
	random_state: int = 0,
):
	"""Generate train/test moons dataset splits."""
	x, y = make_moons(
		n_samples=n_samples,
		noise=noise,
		random_state=random_state,
	)

	x_train, x_test, y_train, y_test = train_test_split(
		x,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=y,
	)

	# Use stable dtypes for NumPy/Torch loading pipelines.
	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)
	y_train = y_train.astype(np.int64)
	y_test = y_test.astype(np.int64)

	return x_train, y_train, x_test, y_test


def save_numpy(output_dir: Path, x_train, y_train, x_test, y_test) -> Path:
	"""Save dataset as a compressed NumPy archive."""
	output_path = output_dir / "moons_dataset.npz"
	np.savez_compressed(
		output_path,
		x_train=x_train,
		y_train=y_train,
		x_test=x_test,
		y_test=y_test,
	)
	return output_path


def save_torch(output_dir: Path, x_train, y_train, x_test, y_test) -> Path:
	"""Save dataset as Torch tensors in a .pt file."""
	output_path = output_dir / "moons_dataset.pt"
	torch.save(
		{
			"x_train": torch.from_numpy(x_train),
			"y_train": torch.from_numpy(y_train),
			"x_test": torch.from_numpy(x_test),
			"y_test": torch.from_numpy(y_test),
		},
		output_path,
	)
	return output_path


def parse_args():
	parser = argparse.ArgumentParser(
		description="Generate sklearn moons dataset and save locally."
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path.cwd(),
		help="Directory to save output files (default: current working directory).",
	)
	parser.add_argument("--n-samples", type=int, default=2000)
	parser.add_argument("--noise", type=float, default=0.2)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--seed", type=int, default=0)
	return parser.parse_args()


def main():
	args = parse_args()
	args.output_dir.mkdir(parents=True, exist_ok=True)

	x_train, y_train, x_test, y_test = generate_moons_dataset(
		n_samples=args.n_samples,
		noise=args.noise,
		test_size=args.test_size,
		random_state=args.seed,
	)

	npz_path = save_numpy(args.output_dir, x_train, y_train, x_test, y_test)
	pt_path = save_torch(args.output_dir, x_train, y_train, x_test, y_test)

	print(f"Saved NumPy dataset to: {npz_path}")
	print(f"Saved Torch dataset to: {pt_path}")
	print("Shapes:")
	print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
	print(f"  x_test : {x_test.shape}, y_test : {y_test.shape}")


if __name__ == "__main__":
	main()
