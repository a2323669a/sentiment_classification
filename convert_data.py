#%%
import pandas as pd
import numpy as np

df = pd.read_csv("./data/test.csv")
#%%
df_split = df['feature'].str.split(' ',expand = True)
#%%
df.drop('feature',axis=1,inplace=True)
#%%
df_conv = df.join(df_split)
#%%
df_conv.to_csv('./data/test_conv.csv',index=False)