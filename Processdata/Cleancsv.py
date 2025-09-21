import pandas as pd

df = pd.read_csv('Processdata/combined_df_mapped_ver3.csv')


df_cleaned = df.dropna()

df.cleaned.to_csv('Dataset.csv',index=False)