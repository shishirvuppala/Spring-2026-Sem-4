# Helper file... you can save a dataframe in your q3_solve.py using
# df.to_csv('filename')
# Then load it here and analyze it

from model import analyze_dataframe
import pandas as pd
import matplotlib.pyplot as plt


# Also helpful for other plots you may want to see...


# FILL OUT FILENAME
filename = "q3_1.csv"

df = pd.read_csv(filename)

analyze_dataframe(df)
