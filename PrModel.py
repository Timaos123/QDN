import tensorflow as tf
from SimEnvironment import SimEnvironment
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import random
import time
import numpy as np
import tqdm
import copy
from sklearn.metrics import f1_score
import os


class PrModel:

    def __init__(self, T, dim=5, decisionType=[-1, 0, 1], summary=False, reloadModel=False):
        self.T = T
        self.dim = dim
        self.decisionType = decisionType
        self.buildModel(summary=summary, reloadModel=reloadModel)
        self.oldF1 = 0

    def buildModel(self, summary=True, reloadModel=False):
        
        inputLayer=tf.keras.layers.Input(shape=(self.T,self.dim,))
        lstmLayer=tf.keras.layers.LSTM(64,activation="tanh")(inputLayer)
        outputLayer=tf.keras.layers.Dense(2,activation="softmax")(lstmLayer)

        self.model = tf.keras.Model(inputs=inputLayer, outputs=outputLayer)
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(
            lr=0.001), loss="sparse_categorical_crossentropy")
        
        if reloadModel == True:
            self.model.load_weights("model/rlModel/RLModel")
        if summary == True:
            self.model.summary()

    def fit(self, x, y, testSize=0.2, epochs=15):
        
        trainX,testX,trainY,testY = train_test_split(x,y,test_size=testSize)
        
        es = tf.keras.callbacks.EarlyStopping(
            mode="auto", monitor="loss", patience=5)
        self.model.fit(trainX, trainY, epochs=epochs, callbacks=[es])
        preY=self.model.predict(testX)
        
        f1=f1_score(testY,np.argmax(preY,axis=1))
        print("f1:",f1)

    def predict(self, testX):
        preY = self.model.predict(testX)
        return preY[0]

if __name__=="__main__":

    T = 5
    dim = 5
    xIndex=[0,1,2,3]
    yIndex=4

    mySimEnv = SimEnvironment(dim=dim)
    pArr = np.array([mySimEnv.giveRandomVal(i) for i in range(1, 150)])
    
    pxArr = np.array([row[xIndex] for row in pArr])
    pyArr = np.array([row[yIndex] for row in pArr])
    
    pXArr = np.array([(pxArr[pI+1:pI+T]-pxArr[pI:pI+T-1]) /
                        pxArr[pI:pI+T-1] for pI in range(pxArr.shape[0]-T)])
    
    pYArr = np.array([(pyArr[pI]-pyArr[pI-1])/pyArr[pI-1]
                      for pI in range(T, pyArr.shape[0])])
    
    myPrModel = PrModel(T-1,dim=dim-1)
    myPrModel.fit(pXArr, (pYArr > 0).astype(int),epochs=50)
    
    preY=myPrModel.predict(np.array([pXArr[0]]))
    
    print(preY)
    print(123)
