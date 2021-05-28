import tensorflow as tf
from SimEnvironment import SimEnvironment
from sklearn.neural_network import MLPClassifier
import random
import time

class prModel:

    def __init__(self):
        pass

    def fit(self,x):
        pass

    def predict(self,x):
        return [random.sample(list(range(3)),3)]

class Learner:

    def __init__(self,T):
        
        self.inputT=T
        
        self.QTable={}
        self.model=prModel()
        self.tmpHistoricalPriceList = []
        self.historicalPriceList = []
        self.historicalDecisionList = []
        self.historicalStatusList = []
        self.state={
            "initPrice":0,
            "direction":0,
            "capital":1
        }
        self.strategyList=[self.goUp,self.closeUp,self.observe,self.goDown,self.closeDown]

    def fitModel(self):
        self.historicalDecisionList.append(self.state["direction"])
        self.historicalStateList.append(self.state["direction"])
        self.model.fit(
            [self.historicalPriceList, self.historicalDecisionList], [self.historicalPriceList,self.historicalStateList])

    def status2strategy(self,oldStatus,newStatus):
        if oldStatus == newStatus:
            strategyI = 2
        elif oldStatus == 0 and newStatus == 1:
            strategyI = 0
        elif oldStatus == 1 and newStatus == 0:
            strategyI = 1
        elif oldStatus == 0 and newStatus == -1:
            strategyI = 2
        elif oldStatus == -1 and newStatus == 0:
            strategyI = 3
        return strategyI

    def doStrategy(self,newPriceListList):

        x=[newPriceListList]
        statusPList = self.model.predict(x)[0]
        preStatusI = statusPList.index(max(statusPList))
        statusNameList=[-1,0,1]

        strategyI=self.status2strategy(
            self.state["direction"], statusNameList[preStatusI])
        
        strategyNameList=["buy-up","close-up","observe","buy-down","close-down"]
        while self.strategyList[strategyI](newPriceList)==False:
            statusPList[preStatusI] = 0
            preStatusI = statusPList.index(max(statusPList))
            strategyI = self.status2strategy(
                self.state["direction"], statusNameList[preStatusI])
        return strategyNameList[strategyI]

    def observe(self,newPriceList):
        self.calQ(newPriceList)
        return True

    def goUp(self,newPriceList):
        if self.state["direction"]!=0:
            return False
        else:
            self.state["initPrice"]=newPriceList[-1]
            self.state["direction"]=1
            self.calQ(newPriceList)
        return True
    
    def closeUp(self,newPriceList):
        if self.state["direction"]==1:
            self.calQ(newPriceList)
            self.state["direction"]=0
        else:
            return False
        return True

    def goDown(self,newPriceList):
        if self.state["direction"]!=0:
            return False
        else:
            self.state["initPrice"] = newPriceList[-1]
            self.state["direction"]=-1
            self.calQ(newPriceList)
        return True

    def closeDown(self,newPriceList):
        if self.state["direction"]==-1:
            self.calQ(newPriceList)
            self.state["direction"]=0
        else:
            return False
        return True

    def calQ(self,newPriceList):
        if self.state["direction"]!=0:
            profit = self.state["direction"]*(newPriceList[-1]-self.state["initPrice"])
            self.state["capital"] *= (1+profit/self.state["initPrice"])
            self.recordHistory(newPriceList)

    def recordHistory(self,newPriceList):
        if len(self.historicalPriceList)==0:
            self.historicalPriceList=newPriceList.copy()
            self.historicalDecisionList = [
                0 for priceI in range(len(self.historicalPriceList)-1)]
            self.historicalStatusList = [
                1 for priceI in range(len(self.historicalPriceList)-1)]
        self.historicalPriceList.append(newPriceList[-1])
        self.historicalDecisionList.append(self.state["direction"])# 永远是上一轮的决策
        self.historicalStatusList.append(self.state["capital"])
        
            
    def getInfo(self):
        return self.state
 
if __name__=="__main__":
    
    T=5
    myLearner=Learner(T)
    mySE=SimEnvironment()

    totalStateList=[]
    pList=[]
    newPriceList=[]
    for i in range(150):
        newPriceList.append(mySE.giveRandomVal(i))
        if len(newPriceList) >=T:
            strategyStr=myLearner.doStrategy(newPriceList)
            if myLearner.state["direction"] == 0:
                pList.append(str(newPriceList[-1]))
                totalStateList.append(pList)
                seqStr = "->".join(pList)
                print("price sequence:", seqStr, ";strategy:",
                      strategyStr, ";capital:", myLearner.state)
                pList=[]
            else:
                pList.append(str(newPriceList[-1]))
                seqStr="->".join(pList)
                print("price sequence:",seqStr,";strategy:",strategyStr,";capital:",myLearner.state)
            newPriceList = newPriceList[1:]
        # if i%50==0:
        #     myLearner.fitModel()
        # time.sleep(0.5)
    print(totalStateList)
