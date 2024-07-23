import pandas as pd

df = pd.read_csv("resources/sample_data.csv")

df['confident_level'] = 1

df.to_csv("resources/sample_data2.csv")