
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import  MLPClassifier
import pandas as pd
from sklearn.decomposition import PCA


lr=0.1    #learning rate
num_hidden=3
sizes=[100,100,100]
batch_size=20
epochs=50

train="train.csv"
val="valid.csv"


DF = pd.read_csv(train)
# DF = DF.sample(frac=1).reset_index(drop=True) #shuffle the data
train_X =  DF.iloc[:10000,1:785].to_numpy()

train_Y = DF.iloc[:10000,785].to_numpy()


DF_val = pd.read_csv(val)
val_X = DF_val.iloc[:300,1:785].to_numpy()
val_Y = DF_val.iloc[:300,785].to_numpy()



pca = PCA(n_components=100,random_state=1234)
pca.fit(train_X)
train_X = pca.transform(train_X)

print(train_X.shape)
print(train_Y.shape)


model = MLPClassifier(sizes,"logistic",'sgd',batch_size=batch_size,learning_rate="constant",learning_rate_init=lr,
                      max_iter=epochs,momentum=0.9,nesterovs_momentum=True,random_state=1234,verbose=True)
# pca.fit(val_X)
val_X = pca.transform(val_X)


model.fit(train_X,train_Y)
print(model.predict(train_X))
print(model.score(train_X,train_Y))
print(model.score(val_X,val_Y))





