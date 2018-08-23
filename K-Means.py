import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




pd.set_option('display.max_columns', 1000)


df = pd.read_csv('College_Data', index_col=1)

print(df.head())
print(df.info())
print(df.describe())