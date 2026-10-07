import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import quality as qual
import process as proc
import regression as reg


                
trainData, testData, rulData = qual.loadCMAPSS("FD001")



# Verifies the data matches what is expected.
if trainData.shape != (20631, 26):
    raise ValueError(f"Train data shape is {trainData.shape}, expected (20631, 26)")
if testData.shape != (13096, 26):
    raise ValueError(f"Test data shape is {testData.shape}, expected (13096, 26)")

# metadata = qual.getDataSetInfo("FD001")
# print(proc.remainingLife(trainData).groupby("unit number")["RUL"].max())
# print(metadata.get("trainEngineLifetimes"))
# # check that the highest cycle count difference in the data is exactly the number of data points - 1.
# print(proc.remainingLife(trainData).groupby("unit number")["RUL"].min())

# z = proc.displayRemainingVsVars(proc.remainingLife(trainData), engines=[1,2,3,4,5])
# x = proc.getCorrelationHeatmap(trainData)
# plt.show()
# print(proc.getStrongestCorrelations(trainData))
# print(reg.crossValidateModels(proc.remainingLife(trainData)))
# print(reg.crossValidateModels(proc.remainingLife(trainData), ["sensor measurement3", "sensor measurement17", "sensor measurement2", "sensor measurement21",
# "sensor measurement20", "sensor measurement15", "sensor measurement8", "sensor measurement13", "sensor measurement7", "sensor measurement4",
# "sensor measurement12", "sensor measurement11", "sensor measurement14", "sensor measurement9"]))

modelResults = {}

for name, model in reg.defineModels().items():
    print(name)
    smoothedTestResult, rmseResult = reg.evaluateOnTest(name,proc.remainingLife(trainData),testData,rulData)
    if name == "pureLinear":
        err_lin = (smoothedTestResult["RUL_pred"] - smoothedTestResult["RUL_true"].clip(upper=125)).to_numpy()
    elif name == "randomForest":
        err_rf = (smoothedTestResult["RUL_pred"] - smoothedTestResult["RUL_true"].clip(upper=125)).to_numpy()
        
    modelResults[name] = smoothedTestResult

rng = np.random.default_rng(0)
rmse = lambda e: np.sqrt((e ** 2).mean())
diffs = []
for _ in range(2000):
    idx = rng.integers(0, len(err_rf), len(err_rf))    # 100 engine positions, with repeats
    diffs.append(rmse(err_lin[idx]) - rmse(err_rf[idx]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"Linear minus RF RMSE: 95% interval {lo:.2f} to {hi:.2f}")

proc.plotLastCyclePredictions(modelResults)
plt.show()