from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

class PressTimerCalculator(object):
    K = 2.23
    def __init__(self, **kwargs):
        self.X = []
        self.Y = []
        self.batchSize = [80,50]
        self.m = []
    def canAdd(self):
        return len(self.m)<len(self.batchSize)

    def __saveToFile(self):
        df = pd.DataFrame()
        df['dist'] = np.array(self.X).reshape((-1,))
        df['residual'] = np.array(self.Y).reshape((-1,))
        df.to_csv('linear%d.csv'%len(self.m))

    def add(self, dist,residual,tm):
        self.X.append([dist])
        self.Y.append([residual*self.K])
        print('add %r %r'%(dist,residual))
        if len(self.m)<len(self.batchSize) and len(self.Y)>=self.batchSize[len(self.m)]:
            m = LinearRegression()
            m.fit(np.array(self.X),np.array( self.Y ))
            print('<<<<<<<<<<<<  fitted one linear residual model  >>>>>>>>>>>')
            self.__saveToFile();
            self.X = []
            self.Y = []
            self.m.append(m)

    def predict(self,dist):
        tm = self.K * dist;
        for m in self.m:
            tm = tm - m.predict(np.array( [[dist]] ))[0]
        return tm
