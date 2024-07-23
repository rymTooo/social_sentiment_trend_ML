import pandas as pd

df = pd.read_csv("./resources/thai_setiment_dataset.csv", delimiter="\t", names=["text","label"])
label_mapping = {"pos":"positive","neg":"negative"}
df['label'] = df['label'].map(label_mapping)
print(df)
df.to_csv("resources/thai_dataset.csv")