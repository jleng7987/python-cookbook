import pandas as pd

Path_excel = "D:\PycharmProjects\python-cookbook\工业大数据竞赛\data\\"
df = pd.read_csv("D:\PycharmProjects\python-cookbook\工业大数据竞赛\data\Qi.csv")
df['TimeStample1'] = df.TimeStample.apply(lambda x: x.split(' ')[0])
print(df)
df2 = df[['TimeStample1','Qi']]
# df[['TimeStample1','Qi']].to_csv(Path_excel+'Qi1.csv')
dfdll = pd.read_csv("D:\PycharmProjects\python-cookbook\工业大数据竞赛\data\\tempall.csv")

a = set(dfdll.TimeStample1)^set(df2.TimeStample1)
print(a)
dfdll1= dfdll.drop(df[(dfdll.TimeStample1).isin(a)].index)
dfdll1.to_csv(Path_excel+'Qi777.csv')
print(dfdll1)
# data_temp = pd.merge(df2, dfdll, how='inner', on='TimeStample1')
data_temp = pd.concat([df2, dfdll1], axis=1)
data_temp.to_csv(Path_excel+'Qi666.csv')