import tensorflow as tf
import numpy as np

class Tester:

    def __init__(self,T):
        self.T=T
        self.buildModel()

    def buildModel(self):

        actorOutputLayer1=tf.keras.layers.Input(shape=[3,])
        actorOutputLayer2=tf.keras.layers.Input(shape=[3,])
        profitInputLayer=tf.keras.layers.Input(shape=[self.T,])
        criticOutputLayer=tf.keras.layers.Input(shape=[2,])

        finalAddLayer1 = tf.keras.layers.Add()(
            [tf.constant([1.0]), profitInputLayer[:,-1]])
        
        finalMultiplyLayer = tf.keras.layers.Multiply(name="multiply1")([actorOutputLayer1, actorOutputLayer2, criticOutputLayer[:,-1]])
        finalDotLayer = tf.keras.layers.Dot(name="dot",axes=-1)([finalMultiplyLayer, tf.constant([[-1.0,0.0, 1.0]])])
        finalAddLayer2 = tf.keras.layers.Add(name="add1")([tf.constant([[1.0]]), finalDotLayer])
        
        finalOutputLayer2 = tf.keras.layers.Multiply(name="multiply2")([finalAddLayer1, finalAddLayer2])
        finalOutputLayer2 = tf.keras.layers.Add(name="add2")([finalOutputLayer2, tf.constant([[-1.0]])])

        self.testModel=tf.keras.Model(inputs=[actorOutputLayer1,actorOutputLayer2,profitInputLayer,criticOutputLayer],outputs=finalOutputLayer2)

    def predict(self,x):
        preY=self.testModel.predict(x)
        return preY

if __name__=="__main__":
    
    actorOutputX1=np.array([[0,0,1]])
    actorOutputX2=np.array([[0,0,1]])
    profitOutputX=np.array([[0,1,0,-1,1]])
    criticOutputX=np.array([[0,1]])

    myTester=Tester(5)
    preY=myTester.testModel.predict([actorOutputX1,actorOutputX2,profitOutputX,criticOutputX])

    print(preY)