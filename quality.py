import pandas as pd
import os

def loadCMAPSS(dataset : str):
    cols = ['unit number', 'time in cycles'] + ["operational setting" + str(i) for i in range(1, 4)] + ["sensor measurement" + str(i) for i in range(1, 22)]
    # Defines the column names for the dataset.

    trainData = pd.read_csv('CMAPSSData/train_' + dataset + '.txt', sep=r"\s+", header=None, names=cols, index_col=False)
    testData = pd.read_csv('CMAPSSData/test_' + dataset + '.txt', sep=r"\s+", header=None, names=cols, index_col=False)
    rulData = pd.read_csv('CMAPSSData/RUL_' + dataset + '.txt', sep=r"\s+", header=None, names=['RUL'], index_col=False)
    # Reads datapoints, splitting columns on any length of whitespace.
    return [trainData, testData, rulData]

def checkIsMissing(data : pd.DataFrame):
    return data.isna().sum()

def checkDuplicates(df : pd.DataFrame):
    return df.duplicated(["unit number","time in cycles"]).sum()

def checkCycleContinuity(df : pd.DataFrame):
    data = df.groupby("unit number")["time in cycles"]
    steps = data.diff()                                   # one value per row
    increase = (steps.eq(1) | steps.isna()).groupby(df["unit number"]).all()
    start = data.min().eq(1)
    complete = ~(increase & start)
    return complete[complete].index

def checkConsistency(train: pd.DataFrame, test: pd.DataFrame, rul: pd.DataFrame):
    return {
        "rulTestMatch" : set(test["unit number"]) == set(range(1, len(rul) + 1)),
        "noMissingRul" : checkIsMissing(rul) == 0,
        "rulPositive" : bool((rul["RUL"] > 0).all())
        }

def engineLifetimes(train: pd.DataFrame):
    return train.groupby("unit number")["time in cycles"].max()

def summariseColumns(train: pd.DataFrame):
    cols = train.columns[2:]
    summary = train[cols].describe().T[["mean", "std", "min", "max"]]
    summary["unique"] = train[cols].nunique()
    summary["relStd"] = summary["std"] / summary["mean"].abs()
    return summary.round(4)

def driftVsNoise(data: pd.DataFrame, n=10):
    cols = data.columns[2:]
    g = data.groupby("unit number")
    drift = (g[cols].apply(lambda d: d.tail(n).mean() - d.head(n).mean())).mean()
    # A measure of how much the average of each column changes between the initial and final values.
    noise = g[cols].apply(lambda d: d.head(n).std()).mean()
    # A measure of how much the values at the beginning of each column vary on average.
    return (drift.abs() / (noise + 1e-9)).sort_values()


def getDataSetInfo(dataset: str):
    trainData, testData, rulData = loadCMAPSS(dataset)
    consistencyInfo = checkConsistency(trainData, testData, rulData)
    return {
        "trainDataIsMissing" : checkIsMissing(trainData),
        "testDataIsMissing" : checkIsMissing(testData),
        "duplicateTrainData" : checkDuplicates(trainData),
        "duplicateTestData" : checkDuplicates(testData),
        "trainEngineLifetimes" : engineLifetimes(trainData),
        "testEngineLifetimes" : engineLifetimes(testData),
        "trainIsContinuous" : checkCycleContinuity(trainData),
        "testIsContinuous" : checkCycleContinuity(testData),
        "rulTestMatch" : consistencyInfo.get("rulTestMatch"),
        "noMissingRul" : consistencyInfo.get("noMissingRul"),
        "rulPositive" : consistencyInfo.get("rulPositive"),
        "driftVsNoiseTrain": driftVsNoise(trainData)
        
    }