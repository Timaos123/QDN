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
from PrModel import PrModel

class Learner:

    def __init__(self,T,dim=5,summary=True):
        
        self.inputT=T
        
        self.QTable={}
        self.rlModel = PrModel(T, dim=dim, summary=summary)
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
        regPriceList = (np.array(newPriceList) -self.meanPrice)/self.stdPrice
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
    
    T=5
    dim=5
    myLearner = Learner(T, summary=True, dim=dim)
    mySE = SimEnvironment(dim=dim)
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
