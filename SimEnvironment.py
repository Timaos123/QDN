import random

class SimEnvironment:

    def __init__(self):
        pass

    def giveRandomVal(self,iter):
        # randomV=round(random.random()*(-1/5*(iter%10)**2+2*(iter%10)),2)
        randomV=round(-1/5*(iter%10)**2+2*(iter%10)+10,2)
        return randomV

if __name__=="__main__":
    mySimEnv=SimEnvironment()
    for i in range(1,150):
        print(i,":")
        print(mySimEnv.giveRandomVal(i))
