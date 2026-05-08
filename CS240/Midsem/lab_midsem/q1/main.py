import numpy as np
import os
import sys

from model import LogisticModel, train, accuracy, give_optimal_hyperparameter

# ======================================================
# Decide which tests to run
# ======================================================

TESTS_ROOT = "tests"

if len(sys.argv) > 1:
    # Run specific test
    test_names = [sys.argv[1]]
else:
    # Run all tests
    test_names = sorted(os.listdir(TESTS_ROOT))

# ======================================================
# Run tests
# ======================================================

for TEST_NAME in test_names:

    TEST_DIR = os.path.join(TESTS_ROOT, TEST_NAME)

    if not os.path.isdir(TEST_DIR):
        continue

    print("\n====================================")
    print("Running:", TEST_NAME)
    print("====================================")

    # ======================================================
    # Load dataset
    # ======================================================

    Xtr = np.load(os.path.join(TEST_DIR, "X_train.npy"))
    ytr = np.load(os.path.join(TEST_DIR, "y_train.npy"))

    Xval = np.load(os.path.join(TEST_DIR, "X_val.npy"))
    yval = np.load(os.path.join(TEST_DIR, "y_val.npy"))

    Xte = np.load(os.path.join(TEST_DIR, "X_test.npy"))
    yte = np.load(os.path.join(TEST_DIR, "y_test.npy"))

    # ======================================================
    # Find optimal λ (model handles standardization)
    # ======================================================

    lam_opt = give_optimal_hyperparameter(Xtr, ytr, Xval, yval)

    print("Optimal λ:", lam_opt)

    # ======================================================
    # Train final model
    # ======================================================

    model = LogisticModel(Xtr.shape[1], lam=lam_opt)
    train(model, Xtr, ytr)

    # ======================================================
    # Metrics
    # ======================================================

    train_acc = accuracy(ytr, model.predict(Xtr))
    val_acc   = accuracy(yval, model.predict(Xval))
    test_acc  = accuracy(yte, model.predict(Xte))

    print("\nPerformance")
    print("-----------")
    print("Train accuracy :", train_acc)
    print("Val accuracy   :", val_acc)
    print("Test accuracy  :", test_acc)