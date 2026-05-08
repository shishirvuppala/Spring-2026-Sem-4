import time
import numpy as np

from targets import target_function
from student_train import StudentClass

TIMEOUT = 15.0
TOLERANCE = 1e-2


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def main():
    start = time.time()
    model = StudentClass().fit_model(seed=123)
    elapsed = time.time() - start

    rng = np.random.default_rng(999)
    x_test = rng.uniform(0.0, 1.0, size=(2048, 1))
    y_test = target_function(x_test)
    y_pred = model.predict(x_test)
    err = mse(y_pred, y_test)

    print(f"Test MSE: {err:.8f}")
    print(f"Train time: {elapsed:.3f} sec")
    if elapsed > TIMEOUT:
        print("Time limit exceeded!")
    elif err > TOLERANCE:
        print("MSE too high!")
    else:
        print("Success!")


if __name__ == "__main__":
    main()
