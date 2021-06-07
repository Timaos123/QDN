import random
import copy

class SimEnvironment:

    def __init__(self, dim=5):
        self.dim = dim
        pass

    def giveRandomVal(self,iter):
        # randomV=round(random.random()*(-1/5*(iter%10)**2+2*(iter%10)),2)
        randomV=[round(-1/5*(iter%10)**2+2*(iter%10)+10,2) for i in range(self.dim)]
        return randomV


class SimDecision:
    
    def __init__(self,strategySpace=[-1,0,1],T=5):
        self.strategySpace=strategySpace
        self.T=T
        
    def developDecisions(self,sampleNum):
        sampleDecisionList=[]
        for sampleI in range(sampleNum):
            sampleTList=[]
            for TI in range(self.T):
                if TI>0:
                    if sampleTList[-1]==1:
                        sampleTList.append(random.sample(
                            self.strategySpace[1:], 1)[0])
                    elif sampleTList[-1] == -1:
                        sampleTList.append(random.sample(
                            self.strategySpace[:-1], 1)[0])
                    else:
                        sampleTList.append(random.sample(
                            self.strategySpace, 1)[0])
                else:
                    sampleTList.append(random.sample(
                        self.strategySpace, 1)[0])
            sampleDecisionList.append(copy.deepcopy(sampleTList))
        return sampleDecisionList
        

if __name__=="__main__":
    
    T=5
    dim=5
    
    mySimEnv=SimEnvironment(dim=dim)
    pList=[mySimEnv.giveRandomVal(i) for i in range(1, 150)]
    pRowList = [pList[pI:pI+T] for pI in range(len(pList)-T)]

    print(123)
