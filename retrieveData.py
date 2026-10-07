import pandas as pd
import matplotlib.pyplot as plt
import quality as qual
import process as proc


                
trainData, testData, rulData = qual.loadCMAPSS("FD001")



# Verifies the data matches what is expected.
if trainData.shape != (20631, 26):
    raise ValueError(f"Train data shape is {trainData.shape}, expected (20631, 26)")
if testData.shape != (13096, 26):
    raise ValueError(f"Test data shape is {testData.shape}, expected (13096, 26)")

metadata = qual.getDataSetInfo("FD001")
print(proc.remainingLife(trainData).groupby("unit number")["RUL"].max())
print(metadata.get("trainEngineLifetimes"))
# check that the highest cycle count difference in the data is exactly the number of data points - 1.
print(proc.remainingLife(trainData).groupby("unit number")["RUL"].min())

# z = proc.displayRemainingVsVars(proc.remainingLife(trainData), engines=[1,2,3,4,5])
# x = proc.getCorrelationHeatmap(trainData)
# plt.show()
print(proc.getStrongestCorrelations(trainData))