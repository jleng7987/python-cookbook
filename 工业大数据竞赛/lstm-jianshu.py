import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#读取数据

df = pd.read_csv("D:\PycharmProjects\python-cookbook\工业大数据竞赛\data\\Qi777.csv")
data = df.values
print(data.shape)

seq_len = 7
X_train = np.array([data[i : i + seq_len, :] for i in range(data.shape[0] - seq_len)])
y_train = np.array([data[i + seq_len, -1] for i in range(data.shape[0]- seq_len)])
# X_test = np.array([data_test[i : i + seq_len, :] for i in range(data_test.shape[0]- seq_len)])
# y_test = np.array([data_test[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])
print(X_train)
print("lalalaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(y_train)

def seq2seqModel(X,step):
    '''
    序列到序列堆叠式LSTM模型
    '''
    model=Sequential()
    model.add(LSTM(256, activation='relu', return_sequences=True,input_shape=(step,X.shape[2])))
    model.add(LSTM(256, activation='relu'))
    model.add(Dense(X.shape[2]))
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__=='__main__':
    #数据集加载
    with open('dataset.txt') as f:
        data_list=[one.strip().split(',') for one in f.readlines()[1:] if one]
    dataset=[]
    for i in range(len(data_list)):
        dataset.append([float(O) for O in data_list[i][1:]])
    dataset=np.array(dataset)
    step=7
    X_train,X_test,y_train,y_test=dataSplit(dataset,step)
    model=seq2seqModel(X_train,step)
    model.fit(X_train,y_train,epochs=50,verbose=0)


