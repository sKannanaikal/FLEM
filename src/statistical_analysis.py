import json
import matplotlib.pyplot as plt
from scipy import stats

GLOBAL_KS_RESULTS_FILE = '/home/stk5106/raw_byte_classifier/results/ks_functions_only_maliciousFunctions.json'
GLOBAL_KS_RESULTS_DATA = {}
GLOBAL_LIME_RESULTS_FILE ='/home/stk5106/raw_byte_classifier/results/lime_functions_only_maliciousFunctions.json'
GLOBAL_LIME_RESULTS_DATA = {}
GLOBAL_FUNCTION_MAPPING_FILE = '/home/stk5106/raw_byte_classifier/results/neofunction_offsets.json'
GLOBAL_FUNCTION_MAPPING_DATA = {}

def transformData(functionlist, sample):
    updatedData = functionlist
    currentMaliciousIndex = 0
    currentFunctionCount = 0
    functions = GLOBAL_FUNCTION_MAPPING_DATA[sample]
    dissassembledFunctions = list(functions.keys())
    for function in functionlist:
        for functionID in range(len(dissassembledFunctions)):
            if function == dissassembledFunctions[functionID]:
                updatedData[currentMaliciousIndex] = functionID
                currentMaliciousIndex += 1
    return updatedData

def plotData(data):
    plt.hist(data, bins=20, edgecolor='black')
    plt.xlabel('P-values')
    plt.ylabel('Count')
    plt.title('Distrubution of P-values between Lime and Kernel Shap Malicious Function Ranking')
    plt.axvline(x=0.05, color='red', linestyle='--', label='significance threshold')
    plt.legend()
    plt.show()

def calculateAverage(data):
    totalData = 0
    for item in data:
        totalData += item
    return totalData / len(data)

def main():
    global GLOBAL_KS_RESULTS_DATA
    global GLOBAL_LIME_RESULTS_DATA
    global GLOBAL_FUNCTION_MAPPING_DATA
    
    ks = []
    lime = []
    collectedPValues = []
    tauCoeficient = []
    
    with open(GLOBAL_KS_RESULTS_FILE, 'r') as ksResultsFile:
        GLOBAL_KS_RESULTS_DATA = json.load(ksResultsFile)
    
    print('[+] Obtained Kernel Shap Results')
    
    with open(GLOBAL_LIME_RESULTS_FILE, 'r') as limeResultsFile:
        GLOBAL_LIME_RESULTS_DATA = json.load(limeResultsFile)
    
    print('[+] Obtained Lime Results')
    
    with open(GLOBAL_FUNCTION_MAPPING_FILE, 'r') as functionMapping:
        GLOBAL_FUNCTION_MAPPING_DATA = json.load(functionMapping)
    
    print('[+] Obtained Global Function Mapping')
    
    for sample in GLOBAL_KS_RESULTS_DATA:
        if sample in GLOBAL_LIME_RESULTS_DATA:
            ks = GLOBAL_KS_RESULTS_DATA[sample]
            lime = GLOBAL_LIME_RESULTS_DATA[sample]
            ksNumerical = transformData(ks, sample)
            limeNumerical = transformData(lime, sample)
            results = stats.kendalltau(ksNumerical, limeNumerical)
            print(f'[+] {sample} - {results.pvalue}')
            collectedPValues.append(results.pvalue)
            tauCoeficient.append(results.statistic)
    
    print(f'On Average there was a tau value of {calculateAverage(tauCoeficient)}')
    plotData(collectedPValues)

if __name__ == '__main__':
    main()