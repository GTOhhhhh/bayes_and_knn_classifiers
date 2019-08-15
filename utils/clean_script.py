import pandas as pd
from pprint import pprint

df = pd.read_csv('pima.csv')
df = df.drop(df.columns[-1], axis = 1)
df.to_csv('pima_test.csv', index=False)