import tensorflow as tf
from SimEnvironment import SimEnvironment
import random
import time

class prModel:

    def __init__(self):
        pass

    def fit(self,x):
        pass

    def predict(self,x):
        return [random.sample(list(range(5)),5)]

class Learner:

    def __init__(self):
        self.QTable={}
        self.model=prModel()
        self.state={
            "initPrice":0,
            "direction":0,
            "capital":1
        }
        self.strategyList=[self.goUp,self.closeUp,self.observe,self.goDown,self.closeDown]

    def fitModel(self):
        self.model.fit(self.history)

    def doStrategy(self,newPrice):

        x=[newPrice]
        strategyPList=self.model.predict(x)[0]
        strategyI=strategyPList.index(max(strategyPList))

        strategyNameList=["buy-up","close-up","observe","buy-down","close-down"]
        while self.strategyList[strategyI](newPrice)==False:
            strategyPList[strategyI]=0
            strategyI=strategyPList.index(max(strategyPList))
        return strategyNameList[strategyI]

    def observe(self,newPrice):
        self.calQ(newPrice)
        return True

    def goUp(self,newPrice):
        if self.state["direction"]!=0:
            return False
        else:
            self.state["initPrice"]=newPrice
            self.state["direction"]=1
        self.calQ(newPrice)
        return True
    
    def closeUp(self,newPrice):
        if self.state["direction"]==1:
            self.state["initPrice"]=newPrice
            self.state["direction"]=0
        else:
            return False
        self.calQ(newPrice)
        return True

    def goDown(self,newPrice):
        if self.state["direction"]!=0:
            return False
        else:
            self.state["initPrice"]=newPrice
            self.state["direction"]=-1
        self.calQ(newPrice)
        return True

    def closeDown(self,newPrice):
        if self.state["direction"]==-1:
            self.state["initPrice"]=newPrice
            self.state["direction"]=0
        else:
            return False
        self.calQ(newPrice)
        return True

    def calQ(self,newPrice):
        if self.state["direction"]!=0:
            self.state["capital"]*=(1+(newPrice-self.state["initPrice"])/self.state["initPrice"])

    def getInfo(self):
        return self.state
 
if __name__=="__main__":
    
    myLearner=Learner()
    mySE=SimEnvironment()

    totalStateList=[]
    pList=[]
    for i in range(150):
        newPrice=mySE.giveRandomVal(i)
        strategyStr=myLearner.doStrategy(newPrice)
        if myLearner.state["direction"]==0:
            totalStateList.append(pList)
            seqStr="->".join(pList)
            pList=[]
        else:
            pList.append(str(newPrice))
            seqStr="->".join(pList)
        print("price sequence:",seqStr,";strategy:",strategyStr,";capital:",myLearner.state)
        # time.sleep(0.5)
    print(totalStateList)