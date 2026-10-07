import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
import sklearn.base as skb
import sklearn.linear_model as skl
import sklearn.ensemble as ske
import sklearn.preprocessing as skpr
import sklearn.decomposition as skd
import sklearn.model_selection as skm
relevantColumns = ["sensor measurement3", "sensor measurement17", "sensor measurement2", "sensor measurement21",
"sensor measurement20", "sensor measurement15", "sensor measurement8", "sensor measurement13", "sensor measurement7", "sensor measurement4",
"sensor measurement12", "sensor measurement11"]

def prepareData(df: pd.DataFrame, sensors, cap=125):
    return df[sensors], df["RUL"].clip(upper=cap), df["unit number"]

def defineModels():
    return {
        "pureLinear" : make_pipeline(skpr.StandardScaler(), skl.LinearRegression()),
        # Regular linear regression, values are normalised to mean 0, standard deviation 1.
        "ridgeRegression" : make_pipeline(skpr.StandardScaler(), skl.RidgeCV(alphas=np.logspace(-3, 3, 13))),
        # Ridge regression, large coefficient values are penalised by the cost function based on alpha, which multiple values can be given for.
        "linearRegressionPCA" : make_pipeline(skpr.StandardScaler(), skd.PCA(n_components=2), skl.LinearRegression()),
        # Regular linear regression, but the dimensions of the input are reduced to the number of components in the form of two new variables with maximal variation captured
        "randomForest" : ske.RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=0, n_jobs=-1)
        # Produces a set of 100 decision trees based on different subsets of the data and considering the columns as conditions, then takes the average of which decision is made on each record.
    }
    
def addRollingFeatures(df, sensors, window=10):
    # Adds a smoothed value to each row which is the average of the previous 9 columns and itself.
    out = df.copy()
    rolled = out.groupby("unit number")[sensors].transform(
        lambda s: s.rolling(window, min_periods=1).mean())
    new_cols = [f"{c}_roll{window}" for c in sensors]
    out[new_cols] = rolled.values
    return out, new_cols
    
def crossValidateModels(trainData: pd.DataFrame, sensors=relevantColumns):
    smoothedTest, newCols = addRollingFeatures(trainData, sensors)
    X, y, groups = prepareData(smoothedTest, newCols)
    
    splitter = skm.GroupKFold(n_splits=5)
    models = defineModels()
    rows = []
    for name, model in models.items():
        errors = []
        
        for trainIndex, testIndex in splitter.split(X, y, groups):
            m = skb.clone(model).fit(X.iloc[trainIndex], y.iloc[trainIndex])
            errors.append(np.sqrt(((m.predict(X.iloc[testIndex]) - y.iloc[testIndex]) ** 2).mean()))
        
        rows.append({"model": name, "mean": np.mean(errors), "std": np.std(errors), **{f"fold{i + 1}": e for i, e in enumerate(errors)}})

    return pd.DataFrame(rows)

def evaluateOnTest(model:str, trainData:pd.DataFrame, testData:pd.DataFrame, rulData: pd.DataFrame, sensors=relevantColumns):
    models = defineModels()
    used = models.get(model)
    
    smoothedTrain, newCols = addRollingFeatures(trainData.copy(), sensors)
    X, y, groups = prepareData(smoothedTrain, newCols)
    used.fit(X, y)
    # Trains on dataset
    
    smoothedTest, newCols = addRollingFeatures(testData.copy(), sensors)
    
    rulCopy = rulData.copy()
    max_cycle = smoothedTest.groupby("unit number")["time in cycles"].transform("max") 
    rul_by_engine = pd.Series(rulCopy["RUL"].values, index=range(1, len(rulCopy) + 1))
    smoothedTest["RUL_true"] = smoothedTest["unit number"].map(rul_by_engine) + max_cycle - smoothedTest["time in cycles"]  # engine number -> RUL file value
    smoothedTest["RUL_pred"] = used.predict(smoothedTest[newCols])
    
    last = smoothedTest.groupby("unit number").tail(1)
    assert (last["RUL_true"].values == rulCopy["RUL"].values).all(), "RUL mismatch between test engines and RUL file"
    true_capped = last["RUL_true"].clip(upper=125)
    rmse = np.sqrt(((last["RUL_pred"] - true_capped) ** 2).mean())
    return smoothedTest, rmse
    