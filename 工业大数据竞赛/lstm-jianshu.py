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
#data = pd.read_csv("./data/tempall.csv")[['w']]
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

#
# dataf = data.values[0:-288]
# def create_dataset(dataset, timesteps=36,predict_size=6):#构造数据集
#     datax=[]#构造x
#     datay=[]#构造y
#     for each in range(len(dataset)-timesteps - predict_steps):
#         x = dataset[each:each+timesteps,0]
#         y = dataset[each+timesteps:each+timesteps+predict_steps,0]
#         datax.append(x)
#         datay.append(y)
#     return datax, datay#np.array(datax),np.array(datay)
# #构造train and predict
# scaler = MinMaxScaler(feature_range=(0,1))
# dataf = scaler.fit_transform(dataf)
# train = dataf.copy()
# timesteps = 72#构造x，为72个数据,表示每次用前72个数据作为一段
# predict_steps = 12#构造y，为12个数据，表示用后12个数据作为一段
# length = 288#预测多步，预测288个数据，每次预测12个，想想要怎么构造预测才能满足288？
# trainx, trainy = create_dataset(train, timesteps, predict_steps)
# trainx = np.array(trainx)
# trainy = np.array(trainy)
#
# #变换
# trainx = np.reshape(trainx,(trainx.shape[0],timesteps,1))#变换shape,以满足keras
# #lstm training
# model = Sequential()
# model.add(LSTM(128,input_shape=(timesteps,1),return_sequences= True))
# model.add(Dropout(0.5))
# model.add(LSTM(128,return_sequences=True))
# #model.add(Dropout(0.3))
# model.add(LSTM(64,return_sequences=False))
# #model.add(Dropout(0.2))
# model.add(Dense(predict_steps))
# model.compile(loss="mean_squared_error",optimizer="adam")
# model.fit(trainx,trainy, epochs= 50, batch_size=200)
# #predict
# #因为每次只能预测12个数据，但是我要预测288个数据，所以采用的就是循环预测的思路。每次预测的12个数据，添加到数据集中充当预测x，然后在预测新的12个y，再添加到预测x列表中，如此往复。最终预测出288个点。
# predict_xlist = []#添加预测x列表
# predict_y = []#添加预测y列表
# predict_xlist.extend(dataf[dataf.shape[0]-timesteps:dataf.shape[0],0].tolist())#已经存在的最后timesteps个数据添加进列表，预测新值(比如已经有的数据从1,2,3到288。现在要预测后面的数据，所以将216到288的72个数据添加到列表中，预测新的值即288以后的数据）
# while len(predict_y) < length:
#     predictx = np.array(predict_xlist[-timesteps:])#从最新的predict_xlist取出timesteps个数据，预测新的predict_steps个数据（因为每次预测的y会添加到predict_xlist列表中，为了预测将来的值，所以每次构造的x要取这个列表中最后的timesteps个数据词啊性）
#     predictx = np.reshape(predictx,(1,timesteps,1))#变换格式，适应LSTM模型
#     #print("predictx"),print(predictx),print(predictx.shape)
#     #预测新值
#     lstm_predict = model.predict(predictx)
#     #predict_list.append(train_predict)#新值y添加进列表，做x
#     #滚动预测
#     #print("lstm_predict"),print(lstm_predict[0])
#     predict_xlist.extend(lstm_predict[0])#将新预测出来的predict_steps个数据，加入predict_xlist列表，用于下次预测
#     # invert
#     lstm_predict = scaler.inverse_transform(lstm_predict)
#     predict_y.extend(lstm_predict[0])#预测的结果y，每次预测的12个数据，添加进去，直到预测288个为止
#     #print("xlist", predict_xlist, len(predict_xlist))
#     #print(lstm_predict, len(lstm_predict))
#     #print(predict_y, len(predict_y))
# #error
#
# y_ture = np.array(data.values[-288:])
# train_score = np.sqrt(mean_squared_error(y_ture,predict_y))
# print("train score RMSE: %.2f"% train_score)
# y_predict = pd.DataFrame(predict_y,columns=["predict"])
# y_predict.to_csv("y_predict_LSTM.csv",index=False)
# #plot
# #plt.plot(y_ture,c="g")
# #plt.plot(predict_y, c="r")
# #plt.show()