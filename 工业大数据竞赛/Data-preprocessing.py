import pandas as pd
import numpy as np

# 变量命名规则：
# 入库流量：Flow
# 环境：Envs
# 降雨：RainF
# 遥测：Tel


Path_excel = "D:\PycharmProjects\python-cookbook\工业大数据竞赛\data\\"


def getdata(path):
    df = pd.read_excel(path)
    return df


def dataexpansion(df, n):
    datamore = pd.DataFrame()
    for i in range(len(df)):
        a = df.loc[i]
        d = pd.DataFrame(a).T
        datamore = datamore.append([d] * n)  # 每行复制n倍
    return datamore


Data_Tel = getdata(Path_excel + "入库流量数据.xlsx")
# print(Data_Tel)
Data_Envs = getdata(Path_excel + "环境表.xlsx")
Data_RainF = getdata(Path_excel + "降雨预报数据.xlsx")

Data_Tel_3 = dataexpansion(Data_Tel, 3)
Data_Tel_3['TimeStample'] = pd.to_datetime(Data_Tel_3['TimeStample'])
# print(Data_Envs_24.dtypes)

data_temp = pd.merge(Data_Envs_24, Data_RainF, how='left', on='TimeStample')
# print(data_temp)
Data_Tel_3.to_csv(Path_excel+'Qi.csv')

data_temp_all = pd.concat([data_temp,Data_Tel],axis=1)
# print(data_temp_all)
data_temp_all.to_csv(Path_excel+'tempall.csv')