import random

class SimEnvironment:

    def __init__(self):
        pass

    def giveRandomVal(self,iter):
        return random.random()*(-2*(iter%10)**2+20*(iter%10)+random.random())

if __name__=="__main__":
    mySimEnv=SimEnvironment()
    for i in range(1,150):
        print(i,":")
        print(mySimEnv.giveRandomVal(i))
