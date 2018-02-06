from sklearn.linear_model import LinearRegression
import numpy as np

class PressTimerCalculator(object):
    K = 2.24
    def __init__(self, **kwargs):
        self.X = []
        self.Y = []
        self.batchSize = [10,30,60,100]
        self.m = []

    def add(self, dist,residual):
        self.X.append([dist])
        self.Y.append([residual*self.K])
        print('add %r %r'%(dist,residual))
        if len(self.m)<len(self.batchSize) and len(self.Y)>=self.batchSize[len(self.m)]:
            m = LinearRegression()
            m.fit(np.array(self.X),np.array( self.Y ))
            self.X = []
            self.Y = []
            self.m.append(m)

    def predict(self,dist):
        tm = self.K * dist;
        for m in self.m:
            tm = tm - m.predict(np.array( [[dist]] ))[0]
        return tm
