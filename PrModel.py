import tensorflow as tf
from SimUtil import SimEnvironment,SimDecision
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import random
import time
import numpy as np
import tqdm
import copy
from sklearn.metrics import f1_score
import os
from functools import reduce


class PrModel:

    def __init__(self, T, dim=5, decisionType=[-1, 0, 1], summary=False, reloadModel=False):
        self.T = T
        self.dim = dim
        self.decisionType = decisionType
        self.oldF1 = 0
        self.actionNum = len(decisionType)
        self.buildCriticModel(summary=summary, reloadModel=reloadModel)
        self.buildActorModel(reloadModel=reloadModel, summary=summary)

    def buildCriticModel(self, summary=True, reloadModel=False):
        
        inputLayer=tf.keras.layers.Input(shape=(self.T,self.dim,))
        lstmLayer=tf.keras.layers.LSTM(64,activation="tanh")(inputLayer)
        outputLayer=tf.keras.layers.Dense(2,activation="softmax")(lstmLayer)

        self.criticModel = tf.keras.Model(
            inputs=inputLayer, outputs=outputLayer)
        self.criticModel.compile(optimizer=tf.keras.optimizers.RMSprop(
            lr=0.001), loss="sparse_categorical_crossentropy")
        
        if reloadModel == True:
            self.criticModel.load_weights("model/criticModel/RLModel")
            
        if summary == True:
            self.criticModel.summary()

    def fitCritic(self, x, y, testSize=0.2, epochs=15):
        
        self.fluctuation=y.mean()
        trainX,testX,trainY,testY = train_test_split(x,y,test_size=testSize)
        
        es = tf.keras.callbacks.EarlyStopping(
            mode="auto", monitor="loss", patience=5)
        self.criticModel.fit(trainX, (trainY>0).astype(int), epochs=epochs, callbacks=[es])
        preY = self.criticModel.predict(testX)
        
        f1=f1_score((testY>0).astype(int),np.argmax(preY,axis=1))
        print("f1:",f1)

    def predictCritic(self, testX):
        preY = self.criticModel.predict(testX)
        return preY[0]

    def realCriticResult(self,realYArr,actionArr):
        return realYArr*actionArr

    def actorLoss(self,realY,preActArr):
        pass

    def buildActorModel(self,reloadModel=True,summary=True):
        seqInputLayer = tf.keras.layers.Input(shape=(self.T, self.dim,)) # 状态维度
        actInputLayer = tf.keras.layers.Input(shape=(self.T, self.actionNum,))  # 行为
        profitInputLayer = tf.keras.layers.Input(shape=(self.T, 1,))  # 总收益增减
        
        inputLayer = tf.keras.layers.Concatenate(
            axis=-1)([seqInputLayer, actInputLayer, profitInputLayer])

        criticLstmLayer = tf.keras.layers.LSTM(64, activation="tanh",name="criticLstmLayer")(inputLayer)
        criticOutputLayer = tf.keras.layers.Dense(
            2, activation="softmax", name="criticOutputLayer")(criticLstmLayer)
        
        actorLstmLayer1 = tf.keras.layers.LSTM(64, activation="tanh")(inputLayer)
        actorOutputLayer1 = tf.keras.layers.Dense(3, activation="softmax",name="act1Output")(actorLstmLayer1)
        
        actorLstmLayer2 = tf.keras.layers.LSTM(64, activation="tanh")(inputLayer)
        actorOutputLayer2 = tf.keras.layers.Dense(3, activation="softmax", name="act2Output")(actorLstmLayer2)
        
        actorOutputLayer1_a = tf.keras.layers.Multiply()(
            [actorOutputLayer1, tf.constant([[-1.0, 0.0, 1.0]])])
        actorOutputLayer2_a = tf.keras.layers.Multiply()(
            [actorOutputLayer2, tf.constant([[-1.0, 0.0, 1.0]])])
        a1a2DotLayer = tf.keras.layers.Dot(
            axes=-1)([actorOutputLayer1_a, actorOutputLayer2_a])
        a1a2SqLayer = tf.keras.layers.Multiply()([a1a2DotLayer, a1a2DotLayer])
        a1a2SqLayer = -(tf.keras.layers.BatchNormalization()(a1a2SqLayer)+1.0)/2.0
        
        finalAddLayer1 = tf.keras.layers.Add()(
            [tf.constant([1.0]), profitInputLayer[:,-1]])
        
        finalMultiplyLayer = tf.keras.layers.Multiply(name="multiply1"
        )([actorOutputLayer1, actorOutputLayer2, criticOutputLayer[:,-1]])
        finalDotLayer = tf.keras.layers.Dot(name="dot",axes=-1)(
            [finalMultiplyLayer, tf.constant([[-1.0,0.0, 1.0]])])
        finalAddLayer2 = tf.keras.layers.Add(name="add1")(
            [tf.constant([[1.0]]), finalDotLayer])
        
        finalOutputLayer2 = tf.keras.layers.Multiply(name="multiply2")(
            [finalAddLayer1, finalAddLayer2])
        finalOutputLayer2 = tf.keras.layers.Add(name="add2")(
            [finalOutputLayer2, a1a2SqLayer])

        self.actorModel = tf.keras.Model(
            inputs=[seqInputLayer, actInputLayer, profitInputLayer], outputs=finalOutputLayer2)
        
        self.actorModel.compile(optimizer=tf.keras.optimizers.RMSprop(
            lr=0.1), loss="mse")

        if reloadModel == True:
            self.actorModel.load_weights("model/actorModel/RLModel")
            
        if summary == True:
            self.actorModel.summary()

    def expandAct(self,actArr,actSpace=[-1,0,1]):
        actList = actArr.tolist()
        
        if len(actArr.shape) == 1:
            newActRow = []
            for actColI in range(len(actList)):
                newActItem = np.zeros(len(actSpace))
                newActItem[actSpace.index(actList[actColI])] = 1
                newActRow.append(newActItem)
            return np.array(newActRow)
        
        newActList=[]
        for actRowI in range(len(actList)):
            newActRow=[]
            for actColI in range(len(actList[0])):
                newActItem=np.zeros(len(actSpace))
                newActItem[actSpace.index(actList[actRowI][actColI])]=1
                newActRow.append(newActItem)
            newActList.append(newActRow)
        newActArr = np.array(newActList)
        return newActArr
                
    def fitActor(self, x, y, seqIndex=0, testSize=0.2, epochs=15):
        strategySpace = [-1, 0, 1]
        myDecider = SimDecision(
            strategySpace=strategySpace, T=self.T+2)  # 新增两个决策空间
        aArr = np.array(myDecider.developDecisions(x.shape[0]))
        aX=aArr[:,:-2]
        aY=aArr[:,-2]
        
        totalRewardList=[]
        totalRewardIndex=0
        for rowI in range(aX.shape[0]):
            for colI in range(aX.shape[1]):
                if rowI==0 and colI ==0:
                    totalRewardItem = aX[rowI][colI]*x[rowI][colI][seqIndex]
                else:
                    totalRewardItem = (totalRewardList[totalRewardIndex-1]+1)*(
                        1+aX[rowI][colI]*x[rowI][colI][seqIndex])-1
                totalRewardList.append(totalRewardItem)
                totalRewardIndex+=1
        totalRewardArr = np.array(totalRewardList).reshape([-1,T-1,1])
        
        aX = self.expandAct(aX)
        aX = aX.reshape([aX.shape[0], aX.shape[1], len(strategySpace)])
        
        aY = np.ones([aX.shape[0],1])
        
        # totalX=np.concatenate([x,aX,totalRewardArr],axis=-1)
        
        self.actorModel.fit([x, aX, totalRewardArr], aY,epochs=epochs)
        
        self.act1Model = tf.keras.Model(inputs=self.actorModel.input,
                                        outputs=self.actorModel.get_layer('act1Output').output)
        self.act2Model = tf.keras.Model(inputs=self.actorModel.input,
                                        outputs=self.actorModel.get_layer('act2Output').output)

    def predictActor(self,x,aArr,seqIndex=0):
        
        aX = aArr

        totalRewardList = []
        totalRewardIndex = 0
        for rowI in range(aX.shape[0]):
            for colI in range(aX.shape[1]):
                if rowI == 0 and colI == 0:
                    totalRewardItem = aX[rowI][colI]*x[rowI][colI][seqIndex]
                else:
                    totalRewardItem = (totalRewardList[totalRewardIndex-1]+1)*(
                        1+aX[rowI][colI]*x[rowI][colI][seqIndex])-1
                totalRewardList.append(totalRewardItem)
                totalRewardIndex += 1
        totalRewardArr = np.array(totalRewardList).reshape([-1, T-1, 1])

        aX = self.expandAct(aX)
        preY1 = self.act1Model.predict([x, aX, totalRewardArr])
        preY2 = self.act2Model.predict([x, aX, totalRewardArr])
        
        return preY1,preY2
    
    def predictActorResult(self, x, aArr, seqIndex=0):

        aX = aArr

        totalRewardList = []
        totalRewardIndex = 0
        for rowI in range(aX.shape[0]):
            for colI in range(aX.shape[1]):
                if rowI == 0 and colI == 0:
                    totalRewardItem = aX[rowI][colI]*x[rowI][colI][seqIndex]
                else:
                    totalRewardItem = (totalRewardList[totalRewardIndex-1]+1)*(
                        1+aX[rowI][colI]*x[rowI][colI][seqIndex])-1
                totalRewardList.append(totalRewardItem)
                totalRewardIndex += 1
        totalRewardArr = np.array(totalRewardList).reshape([-1, T-1, 1])

        aX = self.expandAct(aX)
        preY = self.actorModel.predict([x, aX, totalRewardArr])
        
        return preY

if __name__=="__main__":

    T = 5
    dim = 5
    xIndex=[0,1,2,3]
    yIndex=4
    decisionSpace=[-1,0,1]

    mySimEnv = SimEnvironment(dim=dim)
    pArr = np.array([mySimEnv.giveRandomVal(i) for i in range(1, 150)])
    
    pxArr = np.array([row[xIndex] for row in pArr])
    pyArr = np.array([row[yIndex] for row in pArr])
    
    pXArr = np.array([(pxArr[pI+1:pI+T]-pxArr[pI:pI+T-1]) /
                        pxArr[pI:pI+T-1] for pI in range(pxArr.shape[0]-T)])
    
    pYArr = np.array([(pyArr[pI]-pyArr[pI-1])/pyArr[pI-1] for pI in range(T, pyArr.shape[0])])
    
    myPrModel = PrModel(T-1,dim=dim-1,summary=True)
    
    myPrModel.fitCritic(pXArr,pYArr,epochs=50)
    preY=myPrModel.predictCritic(np.array([pXArr[0]]))
    
    myPrModel.fitActor(pXArr, pYArr, epochs=150)
    
    # np.array([[0,0,0,0]])
    preX = pXArr[:5]
    initA = np.array([[int(random.random() > 0.5)
                        for rowItem in range(T-1)]])
    actList=[]
    preY1List=[]
    preY2List=[]
    preYList=[]
    for preXI in range(preX.shape[0]):
        preY1, preY2 = myPrModel.predictActor(
            np.array([pXArr[preXI]]), aArr=initA)
        preY = myPrModel.predictActorResult(
            np.array([pXArr[preXI]]), aArr=initA)
        
        preY1List.append(decisionSpace[np.argmax(preY1)])
        preY2List.append(decisionSpace[np.argmax(preY2)])
        preYList.append(preY)
        
        initA[0] = np.array(initA[0, 1:].tolist()+[decisionSpace[np.argmax(preY1)]])
        actList.append(initA[0].tolist())
        
    print(preY1List)
    print(preY2List)
    print(np.array(preYList))
    print(pyArr[:10])
    print(123)
 
