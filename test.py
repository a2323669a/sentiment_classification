#%%

import pandas as pd
df = pd.read_csv("./data/test_conv.csv")
#%%
id_list = df.iloc[:,0].values.reshape(-1,)
df.drop(labels='id',axis=1,inplace=True)
test_x = df.values.reshape((-1,48,48,1))

#%%

import keras

model :keras.Sequential = keras.models.load_model('./h5/early_best.h5')
pred = model.predict(test_x,verbose=2)

#%%
import numpy as np
pred = np.argmax(pred,axis=1)
#%%

pred = pred.tolist()
id_list = id_list.tolist()
df_result = pd.DataFrame({'id':id_list,'label':pred})


#%%
df_result.to_csv('./result/pred.csv',index=False)
