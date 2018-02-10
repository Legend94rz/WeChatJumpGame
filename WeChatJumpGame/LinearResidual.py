from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import pickle as pkl
import pandas as pd
import numpy as np

class PressTimerCalculator(object):
    K = 2.24
    p=1
    def __init__(self):
        self.X = []
        self.Y = []
        self.batchSize = [30,30,30]
        self.m = []
        self.lm = pkl.load(open('linearModel.pkl','rb'))['m']

    def canAdd(self):
        return len(self.m)<len(self.batchSize)

    def __saveToFile(self):
        df = pd.DataFrame()
        df['dist'] = np.array(self.X).reshape((-1,))
        df['time'] = np.array(self.Y).reshape((-1,))
        df.to_csv('linear%d.csv'%(PressTimerCalculator.p))
        PressTimerCalculator.p = PressTimerCalculator.p+1

    def add(self, dist,residual,tm):
        if abs(residual)>70:
            return
        self.X.append([dist+residual])
        self.Y.append([tm])
        print('add %r %r'%(dist,residual))
        if len(self.m)<len(self.batchSize) and len(self.Y)>=self.batchSize[len(self.m)]:
            m = LinearRegression()
            m.fit(np.array(self.X),np.array( self.Y ))
            print('<<<<<<<<<<<<  fitted one linear residual model  >>>>>>>>>>>')
            self.__saveToFile()
            self.X = []
            self.Y = []
            self.m.append(m)

    def predict(self,dist):
        #tm = self.K * dist;
        #for m in self.m:
        #    p = m.predict(np.array( [[dist]] ))[0]
        #    #if abs(p)<=20:
        #    print('rectify : %d'%(p/self.K))
        #    tm = tm - p
        #return tm
        return self.lm.predict(np.array( [[dist*dist,dist]] ))[0]



if __name__=="__main__":
    m = LinearRegression()
    f = pd.DataFrame()
    for i in range(13):
        f = f.append(pd.read_csv('linear%d.csv'%i),ignore_index=True)
    from sklearn.model_selection import KFold
    from sklearn.metrics  import mean_squared_error
    print(len(f))
    f['sqr']=np.power(f['dist'],2)
    #m.fit(f['dist'].values.reshape((-1,1)),f['time'])
    #print(mean_squared_error(f['time'],m.predict(f['dist'].values.reshape((-1,1)))))
    m.fit(f[['sqr','dist']],f['time'])
    print(mean_squared_error(f['time'],m.predict(f[['sqr','dist']])))

    print('coef: %r,  intercept: %r'%( m.coef_,m.intercept_ ))
    pkl.dump({'m':m},open('linearModel.pkl','wb'))