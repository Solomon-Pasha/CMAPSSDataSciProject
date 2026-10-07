import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
import sklearn.base as skb
import sklearn.linear_model as skl
import sklearn.ensemble as ske
import sklearn.preprocessing as skpr
import sklearn.decomposition as skd
def prepare_data(df: pd.DataFrame, sensors, cap=125):
    return [df[sensors],df["RUL"].clip(upper=cap),df["unit number"]]

def defineModels():
    return {
        "pureLinear" : make_pipeline(skpr.StandardScaler(), skl.LinearRegression()),
        # Regular linear regression, values are normalised to the +1 to -1 range
        "ridgeRegression" : make_pipeline(skpr.StandardScaler(), skl.RidgeCV(alphas=np.logspace(-3, 3, 13))),
        # Ridge regression, large coefficient values are penalised/rewarded by the cost function based on alpha, which multiple values can be given for.
        "linearRegressionPCA" : make_pipeline(skpr.StandardScaler(), skd.PCA(n_components=2), skl.LinearRegression()),
        # Regular linear regression, but the dimensions of the input are reduced to the number of components to avoid over-correlation
        "randomForest" : ske.RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=0, n_jobs=-1)
        # Produces a set of 100 decision trees based on different subsets of the data and considering different columns as conditions, then takes the average which decision is made on each record.
    }