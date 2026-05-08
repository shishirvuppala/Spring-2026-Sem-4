from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV, RidgeCV 
# read logistic regression doc to see how to regularize and set random state
# DO NOT IMPORT ANY OTHER FUNCTIONS FROM SKLEARN

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# DO NOT ADD ANY MORE IMPORTS.
random.seed(42)
np.random.seed(42)

# Whenever you initialize models for please set the random_seed for better replicability. Otherwise your values may differ slightly when we run it on our machines.
# model = LogisticRegression(random_state=20)

class Model:
    def __init__(self):
        # TODO: Initialise any variables/ models you want to use here. You can also initialize them in fit if you wish
        pass


    # HELPER FUNCTIONS... FILL THEM OUT AS YOU WISH
    # Transform features for X in dataframe1
    def transform_features1(self, X: pd.DataFrame):
        return X


    # Transform features fo X in dataframe2
    def transform_features2(self, X: pd.DataFrame):
        return X


    def fit_task1(self, X1_train: pd.DataFrame, y1_train: pd.Series):
        """
            X1_train: Data for Task 1 the classification (pd.DataFrame)
            y1_train: Labels for Task 1 the classification task (pd.Series)
            X2_train: Data for Task 2 the Regression (pd.DataFrame)
            y2_train: Values for Task 2 (pd.Series)
        """

        X1_train_with_features = self.transform_features1(X1_train)
        
        # Train model(s) here

        # Save model(s) as instance variables for use in predict
        pass
            
    def fit_task2(self, X2_train: pd.DataFrame, y2_train: pd.Series):
        """
            X2_train: Data for Task 2 the Regression (pd.DataFrame)
            y2_train: Values for Task 2 (pd.Series)
        """
        X2_train_with_features = self.transform_features2(X2_train)
        
        # Train model(s) here

        # Save model(s) as instance variables for use in predict
        pass

    def predict_task1(self, X1_test: pd.DataFrame):
        """
            X1_test: Test Data for task 1
            returns:
                pred1: A pandas Series which has the predictions for X1_test.
                (As long as main.py properly prints the accuracies, it is fine if you don't return these as pandas Series (they can be numpy series for e.g.))
        """
        X1_test_with_features = self.transform_features1(X1_test)
        
        # Make predictions here

        # Return predictions as a pandas Series

        # remove this when you have your actual predictions
        naive_prediction_chocolate = ["Gopstopper"] * len(X1_test)
        pred1 = pd.Series(naive_prediction_chocolate)
        return pred1


    def predict_task2(self, X2_test: pd.DataFrame):
        """
            X2_test: Test Data for task 2
            returns:
                pred2: A pandas Series which has the predictions for X2_test.
                (As long as main.py properly prints the R2 scores, it is fine if you don't return these as pandas Series (they can be numpy series for e.g.))
        """
        X2_test_with_features = self.transform_features2(X2_test)
        
        # Make predictions here

        # Return predictions as a pandas Series

        # remove this when you have your actual predictions
        naive_prediction_energy = [0] * len(X2_test)
        pred2 = pd.Series(naive_prediction_energy)
        return pred2
    

# helper function to analyze a dataframe. saves / displays a plot of your choice.
# This is a failry simple function for more sophisticated plots try building them yourself.

def analyze_dataframe(df: pd.DataFrame):
    # Select numerical columns
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not num_cols:
        print("No numerical columns found in dataframe.")
        return

    print("\nNumerical columns:")
    for col in num_cols:
        print(f"- {col}")

    print("\nWhat plot do you want to make?")
    print("1. Histogram")
    print("2. Scatter plot")

    choice = input("Enter choice (1/2): ").strip()

    name = input("Enter base name for saved figure: ").strip()

    if not name:
        print("Invalid name.")
        return

    if choice == "1":
        col = input("Enter column name for histogram: ").strip()

        if col not in num_cols:
            print("Invalid column.")
            return

        filename = f"{name}_{col}_hist.png"

        plt.figure()
        plt.hist(df[col].dropna(), bins=30)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {col}")
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

        print(f"Saved figure as: {filename}")
        return

    elif choice == "2":
        col1 = input("Enter first column: ").strip()
        col2 = input("Enter second column: ").strip()

        if col1 not in num_cols or col2 not in num_cols:
            print("Invalid column(s).")
            return

        filename = f"{name}_{col1}_{col2}_scatter.png"

        plt.figure()
        plt.scatter(df[col1], df[col2])
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f"Scatter Plot: {col1} vs {col2}")
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

        print(f"Saved figure as: {filename}")
        return

    else:
        print("Invalid choice.")
        return
