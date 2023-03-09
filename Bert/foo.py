import pandas as pd
import numpy as np

data = pd.read_csv('LIAR-PLUS\dataset\\train2.tsv', sep="\t")
data = data.fillna(0)
print(data.isnull().sum())