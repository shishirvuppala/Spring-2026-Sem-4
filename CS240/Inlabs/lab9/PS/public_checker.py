"""
Public checker for the 2-hour optimized CNN lab test.

Usage:
    1) Keep this file in the same folder as:
           ts_cnn.py
           signal_ops.py
           motif_models.py
           motif_training.py
           data/
    2) Run:
           python public_checker.py

This checker is intentionally shorter and stricter than local_runner.py.
Passing it does not guarantee passing all hidden tests.
"""

import importlib.util
from pathlib import Path
import numpy as np
import torch


def load_student_module():
    path = Path("ts_cnn.py")
    if not path.exists():
        raise FileNotFoundError(
            "Could not find ts_cnn.py in the current directory.")
    spec = importlib.util.spec_from_file_location("student_ts_cnn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    student = load_student_module()

    # Part A
    assert student.conv1d_output_length(128, 5, 1, 2) == 128
    assert student.maxpool1d_output_length(128, 2, 2) == 64

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    k = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    y = student.manual_convolution_1d(x, k, stride=1, padding=1)
    expected = np.array([-2.0, -2.0, -2.0, 3.0], dtype=np.float32)
    assert np.allclose(
        y, expected), f"manual_convolution_1d failed: got {y}, expected {expected}"

    a = student.prepare_input_batch(np.zeros((128,), dtype=np.float32))
    b = student.prepare_input_batch(np.zeros((5, 128), dtype=np.float32))
    assert tuple(a.shape) == (1, 1, 128)
    assert tuple(b.shape) == (5, 1, 128)
    assert a.dtype == torch.float32

    # Parts B and C
    shallow = student.ShallowCNN1D()
    better = student.BetterCNN1D()

    out1 = shallow(np.zeros((4, 128), dtype=np.float32))
    out2 = better(np.zeros((4, 128), dtype=np.float32))
    assert tuple(out1.shape) == (4, 4)
    assert tuple(out2.shape) == (4, 4)

    pred = student.predict(better, np.zeros(
        (9, 128), dtype=np.float32), batch_size=4)
    assert tuple(pred.shape) == (9,)
    assert pred.dtype == torch.long

    # Part D
    train_x = np.load("data/train_signals.npy")
    train_y = np.load("data/train_labels.npy")
    val_x = np.load("data/val_signals.npy")
    val_y = np.load("data/val_labels.npy")

    model, history = student.train_better_model(
        train_x, train_y, val_x, val_y,
        epochs=10, batch_size=64, lr=1e-3, device="cpu"
    )

    assert isinstance(history, dict)
    assert set(history.keys()) == {"train_loss", "train_acc", "val_acc"}
    assert len(history["train_loss"]) == 10
    assert len(history["train_acc"]) == 10
    assert len(history["val_acc"]) == 10

    final_val_acc = history["val_acc"][-1]
    assert final_val_acc >= 0.80, f"Public smoke-test val acc too low: {final_val_acc:.4f}"

    print("Final validation accuracy after 10 epochs:", final_val_acc)
    print("Public checker passed.")


if __name__ == "__main__":
    main()
