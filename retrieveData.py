import pandas as pd


cols = ['unit number', 'time in cycles'] + ["operational setting" + str(i) for i in range(1, 4)] + ["sensor measurement" + str(i) for i in range(1, 26)]
pd.read_csv('CMAPSSData/train_FD001.txt', sep=' ', header=None, names=cols, index_col=False).to_csv('CMAPSSData/train_FD001.csv', index=False)