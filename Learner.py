import tensorflow as tf
from SimEnvironment import SimEnvironment
from sklearn.neural_network import MLPClassifier
import random
import time
import numpy as np
import tqdm
import copy
from sklearn.metrics import f1_score
import os
class prModel:

    def __init__(self,T,decisionType=[-1,0,1],summary=False,reloadModel=False):
        self.T=T
        self.decisionType=decisionType
        self.buildModel(summary=summary,reloadModel=reloadModel)
        self.oldF1=0

    def buildModel(self,summary=True,reloadModel=True):
        priceInputLayer=tf.keras.layers.Input(shape=[self.T,],name="price_input")
        priceReshapeLayer=tf.keras.layers.Reshape([self.T,1,])(priceInputLayer)
        batchNomLayer = tf.keras.layers.BatchNormalization()(priceReshapeLayer)
        priceOutputLayer = tf.keras.layers.GRU(
            64, activation="tanh", return_sequences=True)(batchNomLayer)
        priceOutputLayer = tf.keras.layers.GRU(
            32, activation="linear", return_sequences=True)(priceOutputLayer)
        priceOutputLayer = tf.keras.layers.GRU(
            1, activation="tanh", return_sequences=True, name="price_output_lstm")(priceOutputLayer)

        decisionInputLayer=tf.keras.layers.Input(shape=[self.T,],name="decision_input")
        decisionReshapeLayer=tf.keras.layers.Reshape([self.T,1,])(decisionInputLayer)

        statusInputLayer=tf.keras.layers.Input(shape=[self.T,],name="status_input")
        statusReshapeLayer = tf.keras.layers.Reshape(
            [self.T, 1, ])(statusInputLayer)
        batchNomLayer = tf.keras.layers.BatchNormalization()(statusReshapeLayer)

        concatLayer = tf.keras.layers.Concatenate(
            name="price_decision_concat", axis=-1)([decisionReshapeLayer, priceOutputLayer, batchNomLayer])
        statusOutputLayer = tf.keras.layers.GRU(
            32, activation="relu", return_sequences=True)(concatLayer)
        statusOutputLayer = tf.keras.layers.GRU(
            64, activation="relu", return_sequences=True)(statusOutputLayer)
        statusOutputLayer = tf.keras.layers.GRU(
            1, activation="softmax", name="status_output")(statusOutputLayer)

        self.model=tf.keras.Model(inputs=[priceInputLayer,decisionInputLayer,statusInputLayer],outputs=[priceOutputLayer,statusOutputLayer])
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
                             loss={'price_output_lstm':'mse',
                                   'status_output': 'binary_crossentropy'})
        if reloadModel==True:
            self.model.load_weights("model/rlModel/RLModel")
        if summary==True:
            self.model.summary()

    def fit(self,x,y,testSize=0.2,epochs=15):
        
        priceX=np.array([x[0][i:i+self.T] for i in range(len(x[0])-self.T)])
        decisionX=np.array([x[1][i:i+self.T] for i in range(len(x[1])-self.T)])
        statusX=np.array([x[2][i:i+self.T] for i in range(len(x[1])-self.T)])

        priceY=np.array([y[0][i:i+self.T] for i in range(len(y[0])-self.T)])
        statusY=y[1][self.T:]
        statusY=(np.array(statusY)>0).astype(int)

        trainIndexList = random.sample(
            list(range(len(priceX.tolist()))), int(priceX.shape[0]*(1-testSize)))
        testIndexList = random.sample(
            list(range(len(priceX.tolist()))), int(priceX.shape[0]*testSize))

        trainPriceX=priceX[trainIndexList]
        trainDecisionX=decisionX[trainIndexList]
        trainStatusX=statusX[trainIndexList]
        trainPriceY=priceY[trainIndexList]
        trainStatusY=statusY[trainIndexList]
        
        testPriceX = priceX[testIndexList]
        testDecisionX=decisionX[testIndexList]
        testStatusX=statusX[testIndexList]
        testPriceY=priceY[testIndexList]
        testStatusY=statusY[testIndexList]

        es=tf.keras.callbacks.EarlyStopping(mode="auto",monitor="loss",patience=5)
        self.model.fit([np.array(trainPriceX), np.array(trainDecisionX), np.array(trainStatusX)], [
                       np.array(trainPriceY), np.array(trainStatusY)], epochs=epochs, callbacks=[es])
        prePriceY,preStatusY=self.model.predict([np.array(testPriceX),np.array(testDecisionX),np.array(testStatusX)])
        preStatusY=(preStatusY>0.5).astype(int).reshape([1,-1])[0]

        f1=f1_score(testStatusY,preStatusY,average="macro")
        print("f1-score",f1)
        if f1>self.oldF1:
            if "rlModel" not in os.listdir("model"):
                os.mkdir("model/rlModel")
            self.model.save_weights("model/rlModel/RLModel")
            self.oldF1=f1

        return max(f1-0.1,0.1)


    def predict(self,x):

        priceX=[x[0]]
        decisionX=[x[1]]
        statusX=[x[2]]
        
        statusList=[]
        for decisionTypeItem in self.decisionType:
            tmpDecisionX=copy.deepcopy(decisionX)
            tmpDecisionX[0].append(decisionTypeItem)
            newPrice, newStatus = self.model.predict(
                [np.array(priceX), np.array(tmpDecisionX), (np.array(statusX) > 0).astype(int)])
            statusList.append(newStatus[0][0])
        
        # statusList=[random.sample([1,2,3],3)]

        return statusList

class Learner:

    def __init__(self,T,summary=True):
        
        self.inputT=T
        
        self.QTable={}
        self.rlModel=prModel(T,summary=summary)
        self.tmpHistoricalPriceList = []
        self.historicalPriceList = []
        self.historicalDecisionList = []
        self.historicalStatusList = []
        self.status={
            "initPrice":0,
            "direction":0,
            "capital":1
        }
        self.oldCapital=1
        self.strategyList=[self.goUp,self.closeUp,self.observe,self.goDown,self.closeDown]
        self.freeDegree=1

    def fitModel(self):
        self.historicalDecisionList.append(self.status["direction"])
        self.historicalStatusList.append(self.status["direction"])
        self.meanPrice = np.mean(self.historicalPriceList)
        self.stdPrice = np.std(self.historicalPriceList)
        regPriceList = (np.array(self.historicalPriceList)-self.meanPrice)/self.stdPrice
        self.freeDegree=1-self.rlModel.fit(
            [regPriceList[:-1], self.historicalDecisionList[:-1],
                self.historicalStatusList[:-1]],
            [regPriceList[1:], self.historicalStatusList[1:]],epochs=200)
        self.historicalDecisionList=self.historicalDecisionList[:-1]
        self.historicalStatusList=self.historicalStatusList[:-1]

    def status2strategy(self,oldStatus,newStatus):
        if oldStatus == newStatus:
            strategyI = 2
        elif oldStatus == 0 and newStatus == 1:
            strategyI = 0
        elif oldStatus == 1 and newStatus == 0:
            strategyI = 1
        elif oldStatus == 0 and newStatus == -1:
            strategyI = 3
        elif oldStatus == -1 and newStatus == 0:
            strategyI = 4
        else:
            return False
        return strategyI

    def doStrategy(self,newPriceList):
        if "meanPrice" not in dir(self):
            self.meanPrice = np.array(newPriceList).mean()
            self.stdPrice = np.array(newPriceList).std()
        regPriceList = (np.array(newPriceList) -
                        self.meanPrice)/self.stdPrice
        x = [regPriceList[-self.inputT:], self.historicalDecisionList[-self.inputT+1:],
             self.historicalStatusList[-self.inputT:]]
        if random.random()<self.freeDegree:
            statusPList=random.sample([1,2,3],3)
        else:
            statusPList = self.rlModel.predict(x)
        preStatusI = statusPList.index(max(statusPList))
        statusNameList=[-1,0,1]

        strategyI=self.status2strategy(
            self.status["direction"], statusNameList[preStatusI])

        strategyNameList=["buy-up","close-up","observe","buy-down","close-down"]
        strategyMethod=self.strategyList[strategyI]
        tfStatus=strategyMethod(newPriceList)
        if tfStatus==False:
            while strategyI==False:
                statusPList[preStatusI] = 0
                preStatusI = statusPList.index(max(statusPList))
                strategyI = self.status2strategy(
                    self.status["direction"], statusNameList[preStatusI])
                strategyMethod=self.strategyList[strategyI]
                tfStatus=strategyMethod(newPriceList)
                if tfStatus==True:
                    break
        return strategyNameList[strategyI]

    def observe(self,newPriceList):
        self.calQ(newPriceList)
        return True

    def goUp(self,newPriceList):
        if self.status["direction"]!=0:
            return False
        else:
            self.status["initPrice"]=newPriceList[-1]
            self.status["direction"]=1
            self.calQ(newPriceList)
        return True
    
    def closeUp(self,newPriceList):
        if self.status["direction"]==1:
            self.calQ(newPriceList)
            self.status["direction"]=0
        else:
            return False
        return True

    def goDown(self,newPriceList):
        if self.status["direction"]!=0:
            return False
        else:
            self.status["initPrice"] = newPriceList[-1]
            self.status["direction"]=-1
            self.calQ(newPriceList)
        return True

    def closeDown(self,newPriceList):
        if self.status["direction"]==-1:
            self.calQ(newPriceList)
            self.status["direction"]=0
        else:
            return False
        return True

    def calQ(self,newPriceList):
        if self.status["direction"]!=0:
            profitRate = self.status["direction"]*(newPriceList[-1]-self.status["initPrice"])/(self.status["initPrice"]+0.0001)
        else:
            profitRate=0
        self.status["capital"] *= (1+profitRate)
        self.recordHistory(newPriceList)

    def recordHistory(self,newPriceList):
        if len(self.historicalPriceList)==0:
            self.historicalPriceList=newPriceList.copy()
            self.historicalDecisionList = [
                0 for priceI in range(len(self.historicalPriceList)-1)]
            self.historicalStatusList = [
                0 for priceI in range(len(self.historicalPriceList)-1)]
        self.historicalPriceList.append(newPriceList[-1])
        self.historicalDecisionList.append(self.status["direction"])# 永远是上一轮的决策
        self.historicalStatusList.append((self.status["capital"]-self.oldCapital)/self.oldCapital)# 永远是上一轮的收益
        self.oldCapital = self.status["capital"]
            
    def getInfo(self):
        return self.status
 
if __name__=="__main__":
    
    T=10
    myLearner=Learner(T,summary=True)
    mySE=SimEnvironment()
    waitToFit=False

    totalStatusList=[]
    pList=[]
    newPriceList=[]
    i=0
    while True:
        newPriceList.append(mySE.giveRandomVal(i))
        if len(newPriceList) >=T:
            strategyStr=myLearner.doStrategy(newPriceList)
            if myLearner.status["direction"] == 0:
                pList.append(str(newPriceList[-1]))
                totalStatusList.append(pList)
                seqStr = "->".join(pList)
                print("price sequence:", seqStr, ";strategy:",
                      strategyStr, ";capital:", myLearner.status)
                pList=[]
            else:
                pList.append(str(newPriceList[-1]))
                seqStr="->".join(pList)
                print("price sequence:",seqStr,";strategy:",strategyStr,";capital:",myLearner.status)
            newPriceList = newPriceList[1:]
        else:
            myLearner.observe(newPriceList)
        if i>0 and i%200==0:
            waitToFit=True
        if waitToFit==True and (myLearner.status["direction"]==0 or random.random()>0.8):
            myLearner.fitModel()
            waitToFit=False
        time.sleep(0.1)
        i+=1
    print(totalStatusList)
