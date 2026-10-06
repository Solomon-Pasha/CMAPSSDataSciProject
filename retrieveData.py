import pandas as pd


cols = ['unit number', 'time in cycles'] + ["operational setting" + str(i) for i in range(1, 4)] + ["sensor measurement" + str(i) for i in range(1, 22)]
# Defines the column names for the dataset.

trainData = pd.read_csv('CMAPSSData/train_FD001.txt', sep=r"\s+", header=None, names=cols, index_col=False)
testData = pd.read_csv('CMAPSSData/test_FD001.txt', sep=r"\s+", header=None, names=cols, index_col=False)
rulData = pd.read_csv('CMAPSSData/RUL_FD001.txt', sep=r"\s+", header=None, names=['RUL'], index_col=False)
# Reads datapoints, splitting columns on any length of whitespace.

trainData.to_csv('CMAPSSData/train_FD001.csv', index=False)