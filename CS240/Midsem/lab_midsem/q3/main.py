# DO NOT MODIFY THIS FILE. YOU WILL BE EVALUTED USING THIS ORIGINAL MAIN FILE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import Model
from sklearn.metrics import accuracy_score, r2_score

df1 = pd.read_csv("q3_1.csv")
df2 = pd.read_csv("q3_2.csv")

X1 = df1.drop(columns="ChocolateType")
y1 = df1["ChocolateType"]

X2 = df2.drop(columns="EnergyConsumption")
y2 = df2["EnergyConsumption"] 

N1_train = int(0.8 * len(X1))
N2_train = int(0.8 * len(X2))

X1_train = X1[:N1_train]
y1_train = y1[:N1_train]

X1_test = X1[N1_train:]
y1_test = y1[N1_train:]

X2_train = X2[:N2_train]
y2_train = y2[:N2_train]

X2_test = X2[N2_train:]
y2_test = y2[N2_train:]


model = Model()

model.fit_task1(X1_train, y1_train)
pred1 = model.predict_task1(X1_test)

model.fit_task2(X2_train, y2_train)
pred2 = model.predict_task2(X2_test)

acc1 = accuracy_score(y_true=y1_test, y_pred=pred1)
print(f"Accuracy of Classifier: {acc1}")

r2score2 = r2_score(y_true=y2_test, y_pred=pred2)
print(f"R2 Score of Regressor: {r2score2}")