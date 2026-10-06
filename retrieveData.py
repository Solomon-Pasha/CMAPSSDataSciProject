import pandas as pd
import quality as qual


                
trainData, testData, rulData = qual.loadCMAPSS("FD001")



# Verifies the data matches what is expected.
if trainData.shape != (20631, 26):
    raise ValueError(f"Train data shape is {trainData.shape}, expected (20631, 26)")
if testData.shape != (13096, 26):
    raise ValueError(f"Test data shape is {testData.shape}, expected (13096, 26)")

print(qual.checkConsistency(trainData, testData, rulData))
print(qual.engineLifetimes(trainData))
trainData.to_csv('outputData/train_FD001.csv', index=False)
testData.to_csv('outputData/test_FD001.csv', index=False)
rulData.to_csv('outputData/RUL_FD001.csv', index=False)
qual.summariseColumns(trainData).to_csv('outputData/trainSummary_FD001.csv')