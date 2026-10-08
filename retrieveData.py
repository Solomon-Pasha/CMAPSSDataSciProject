import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import quality as qual
import process as proc
import regression as reg
from pathlib import Path
relevantColumnsWith9And14 = ["sensor measurement3", "sensor measurement17", "sensor measurement2", "sensor measurement21",
"sensor measurement20", "sensor measurement15", "sensor measurement8", "sensor measurement13", "sensor measurement7", "sensor measurement4",
"sensor measurement12", "sensor measurement11", "sensor measurement14", "sensor measurement9"]
relevantColumnsWithout9And14 = ["sensor measurement3", "sensor measurement17", "sensor measurement2", "sensor measurement21",
"sensor measurement20", "sensor measurement15", "sensor measurement8", "sensor measurement13", "sensor measurement7", "sensor measurement4",
"sensor measurement12", "sensor measurement11"]

def getGraphsIndependent(dataset:str):
    trainData, testData, rulData = qual.loadCMAPSS(dataset)
    trainDataWithRul = proc.remainingLife(trainData)
    return {
        "sensorTrends": proc.displayRemainingVsVars(trainDataWithRul, engines=[1,2,3,4,5], sensor_cols=relevantColumnsWith9And14),
        "correlationHeatmap": proc.getCorrelationHeatmap(trainData)
    }


def getGraphsForSensorSet(dataset:str, sensors:list[str]):
    trainData, testData, rulData = qual.loadCMAPSS(dataset)
    trainDataWithRul = proc.remainingLife(trainData)

    modelResults = {}
    for name, model in reg.defineModels().items():
        smoothedTestResult, rmseResult = reg.evaluateOnTest(name,trainDataWithRul,testData,rulData,sensors)
        modelResults[name] = smoothedTestResult
    return {
        "lastCyclePredictions": proc.plotLastCyclePredictions(modelResults)
    }

def getInfoIndependent(dataset:str):
    trainData, testData, rulData = qual.loadCMAPSS(dataset)
    trainDataWithRul = proc.remainingLife(trainData)
        
    metadata = qual.getDataSetInfo(dataset) 
    columnDataResults = qual.summariseColumns(trainData).to_markdown()
    return {
        "driftVsNoise": metadata.get("driftVsNoiseTrain").to_markdown(), 
        "summary": columnDataResults
    }


def getInfoForSensorSet(dataset:str, sensors:list[str]):
    trainData, testData, rulData = qual.loadCMAPSS(dataset)
    trainDataWithRul = proc.remainingLife(trainData)
    
    metadata = qual.getDataSetInfo(dataset) 
    columnDataResults = qual.summariseColumns(trainData).to_markdown(index=False)
    crossValResults = reg.crossValidateModels(trainDataWithRul, sensors).to_markdown(index=False)
    
    modelResults = {} 
    for name, model in reg.defineModels().items():
        smoothedTestResult, rmseResult = reg.evaluateOnTest(name,trainDataWithRul,testData,rulData,sensors)
        last = proc.lastCycle(smoothedTestResult)
        err = last["error"].to_numpy()
        modelResults[name] = [smoothedTestResult, rmseResult, err]
        

    rng = np.random.default_rng(0)
    rmse = lambda e: np.sqrt((e ** 2).mean())
    diffs = []
    err_rf = modelResults.get("randomForest")[2]
    err_lin = modelResults.get("pureLinear")[2]
    for _ in range(2000):
        idx = rng.integers(0, len(err_rf), len(err_rf))    # 100 engine positions, with repeats
        diffs.append(rmse(err_lin[idx]) - rmse(err_rf[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    testResults = pd.DataFrame({"model": list(modelResults), "test_rmse": [v[1] for v in modelResults.values()]}).to_markdown(index=False)
    return {"crossValidation": crossValResults, 
            "testResults": testResults,
            "bootstrap": f"Linear minus RF RMSE, 95% interval: {lo:.2f} to {hi:.2f}"}


def main():
    Path("results").mkdir(exist_ok=True)
    Path("results/with_9_14").mkdir(exist_ok=True)
    Path("results/without_9_14").mkdir(exist_ok=True)
    
    independentInfo = getInfoIndependent("FD001")
    for name, item in independentInfo.items():
             Path("results/"+name+".txt").write_text(item)
    
    allIncluded = getInfoForSensorSet("FD001", relevantColumnsWith9And14)
    for name, item in allIncluded.items():
         Path("results/with_9_14/"+name+".txt").write_text(item)
    
    outliersRemoved = getInfoForSensorSet("FD001", relevantColumnsWithout9And14)
    for name, item in outliersRemoved.items():
        Path("results/without_9_14/"+name+".txt").write_text(item)
        
    for name, fig in getGraphsIndependent("FD001").items():
        fig.savefig("results/"+name+".png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    for name, fig in getGraphsForSensorSet("FD001", relevantColumnsWith9And14).items():
        fig.savefig("results/with_9_14/"+name+".png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    for name, fig in getGraphsForSensorSet("FD001", relevantColumnsWithout9And14).items():
        fig.savefig("results/without_9_14/"+name+".png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    
if __name__ == "__main__":
    main()