# local_runner.py
#
# Local testing script for the 2-hour optimized CNN lab test.
# Keep this file in the same folder as:
#   - ts_cnn.py
#   - signal_ops.py
#   - motif_models.py
#   - motif_training.py
#   - data/
#
# Example usage:
#   python local_runner.py --part a
#   python local_runner.py --part b
#   python local_runner.py --part c
#   python local_runner.py --part d
#   python local_runner.py --part bonus
#   python local_runner.py --part all

import argparse
import importlib.util
import traceback
from pathlib import Path
import numpy as np
import torch


def load_student_module(path="ts_cnn.py"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path} in the current directory.")
    spec = importlib.util.spec_from_file_location("student_ts_cnn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def print_header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def safe_run(name, fn):
    print(f"\n[RUNNING] {name}")
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as e:
        print(f"[FAIL] {name}")
        print(f"Error: {e}")
        traceback.print_exc()


def load_data():
    needed = [
        Path("data/train_signals.npy"),
        Path("data/train_labels.npy"),
        Path("data/val_signals.npy"),
        Path("data/val_labels.npy"),
    ]
    for p in needed:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    train_x = np.load("data/train_signals.npy")
    train_y = np.load("data/train_labels.npy")
    val_x = np.load("data/val_signals.npy")
    val_y = np.load("data/val_labels.npy")
    return train_x, train_y, val_x, val_y


def check_a(student):
    def _run():
        print_header("Part A: signal operations")

        print("conv1d_output_length(128, 5, 1, 2) =", student.conv1d_output_length(128, 5, 1, 2))
        print("maxpool1d_output_length(128, 2, 2) =", student.maxpool1d_output_length(128, 2, 2))

        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        k = np.array([1.0, 0.0, -1.0], dtype=np.float32)
        out = student.manual_convolution_1d(x, k, stride=1, padding=1)
        expected = np.array([-2.0, -2.0, -2.0, 3.0], dtype=np.float32)

        print("\nmanual_convolution_1d")
        print("output   =", out)
        print("expected =", expected)
        print("allclose =", np.allclose(out, expected))
        assert np.allclose(out, expected)

        a = student.prepare_input_batch(np.zeros((128,), dtype=np.float32))
        b = student.prepare_input_batch(np.zeros((5, 128), dtype=np.float32))
        c = student.prepare_input_batch(np.zeros((5, 1, 128), dtype=np.float32))

        print("\nprepare_input_batch shapes:")
        print("(128,)    ->", tuple(a.shape))
        print("(5, 128)  ->", tuple(b.shape))
        print("(5,1,128) ->", tuple(c.shape))

        assert tuple(a.shape) == (1, 1, 128)
        assert tuple(b.shape) == (5, 1, 128)
        assert tuple(c.shape) == (5, 1, 128)
        assert a.dtype == torch.float32

    safe_run("Part A checks", _run)


def check_b(student):
    def _run():
        print_header("Part B: ShallowCNN1D")

        model = student.ShallowCNN1D()
        print(model)

        x = np.zeros((4, 128), dtype=np.float32)
        y = model(x)

        print("\nOutput shape:", tuple(y.shape))
        print("Parameter count:", student.count_parameters(model))
        assert tuple(y.shape) == (4, 4)

    safe_run("Part B checks", _run)


def check_c(student):
    def _run():
        print_header("Part C: BetterCNN1D")

        model = student.BetterCNN1D()
        print(model)

        x = np.zeros((4, 128), dtype=np.float32)
        y = model(x)

        print("\nOutput shape:", tuple(y.shape))
        print("Parameter count:", student.count_parameters(model))
        assert tuple(y.shape) == (4, 4)

        preds = student.predict(model, np.zeros((9, 128), dtype=np.float32), batch_size=4, device="cpu")
        print("predict() output shape:", tuple(preds.shape))
        assert tuple(preds.shape) == (9,)
        assert preds.dtype == torch.long

    safe_run("Part C checks", _run)


def check_d(student):
    def _run():
        print_header("Part D: training")

        train_x, train_y, val_x, val_y = load_data()

        train_x_small = train_x[:512]
        train_y_small = train_y[:512]
        val_x_small = val_x[:256]
        val_y_small = val_y[:256]

        model, history = student.train_better_model(
            train_x_small,
            train_y_small,
            val_x_small,
            val_y_small,
            epochs=3,
            batch_size=64,
            lr=1e-3,
            device="cpu",
        )

        print("History keys:", list(history.keys()))
        print("train_loss:", history["train_loss"])
        print("train_acc :", history["train_acc"])
        print("val_acc   :", history["val_acc"])

        assert set(history.keys()) == {"train_loss", "train_acc", "val_acc"}
        assert len(history["train_loss"]) == 3
        assert len(history["train_acc"]) == 3
        assert len(history["val_acc"]) == 3

    safe_run("Part D checks", _run)


def check_bonus(student):
    def _run():
        print_header("Bonus")

        block = student.ResidualBlock1D(16)
        x = torch.randn(4, 16, 64)
        y = block(x)
        print("ResidualBlock1D output shape:", tuple(y.shape))
        assert tuple(y.shape) == tuple(x.shape)

        model = student.ResidualCNN1D()
        z = model(np.zeros((5, 128), dtype=np.float32))
        print("ResidualCNN1D output shape:", tuple(z.shape))
        assert tuple(z.shape) == (5, 4)

    safe_run("Bonus checks", _run)


def run_all(student):
    check_a(student)
    check_b(student)
    check_c(student)
    check_d(student)
    check_bonus(student)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part",
        type=str,
        default="all",
        choices=["a", "b", "c", "d", "bonus", "all"],
        help="Which part to test",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="ts_cnn.py",
        help="Path to ts_cnn.py",
    )
    args = parser.parse_args()

    print_header("Loading student file")
    print("Using file:", args.file)

    student = load_student_module(args.file)
    print("Loaded successfully.")

    if args.part == "a":
        check_a(student)
    elif args.part == "b":
        check_b(student)
    elif args.part == "c":
        check_c(student)
    elif args.part == "d":
        check_d(student)
    elif args.part == "bonus":
        check_bonus(student)
    else:
        run_all(student)


if __name__ == "__main__":
    main()
