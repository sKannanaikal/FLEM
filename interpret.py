import torch
import json
import numpy as np
import os
import warnings

from captum.attr import KernelShap
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
from train import MalConvForSequenceClassification
from train import MalConvConfig

GLOBAL_JSON_DATA = {}
GLOBAL_MAXIMUM_EXECUTABLE_SIZE = 100000
GLOBAL_MINIMUM_EXECUTABLE_SIZE = 512
GLOBAL_FLEM_MODEL = '/home/stk5106/raw_byte_classifier/models/flem_partialLukeExecutableSections'
GLOBAL_EXE_MAP = '/home/stk5106/raw_byte_classifier/function_offsets.json'
GLOBAL_MALWARE_DIR = '/home/stk5106/raw_byte_classifier/dataset/malware/'
GLOBAL_DISSASSEMBLED_DIR = '/home/stk5106/raw_byte_classifier/dataset/disassembled/'
GLOBAL_JSON_FILE_LOCATION = '/home/stk5106/raw_byte_classifier/maliciousFunctions.json'

def updateJSONData(filename, maliciousFunctions):
    global GLOBAL_JSON_DATA
    
    GLOBAL_JSON_DATA[filename] = maliciousFunctions

def findBinary(assemblyFileName):
    hashedMalwareName = assemblyFileName.split('.')[0]
    malwareSamples = os.listdir(GLOBAL_MALWARE_DIR)
    samplePath = ''
    
    for sample in malwareSamples:
        if sample.split('.')[0] == hashedMalwareName:
            samplePath = f'{GLOBAL_MALWARE_DIR}{sample}'
            return samplePath
    
    return ''

def extractFunctionsFromBinary(filepath, functionMapping):
    binaryFunctionBytes = b''
    totalBytesExtracted = 0
    fucntionCount = 0
    fileSize = os.path.getsize(filepath)
    with open(filepath, 'rb') as binaryFile:
        for function in functionMapping:
            try:
                startingAddress = int(functionMapping[function][0], 16)
                endingAddress = int(functionMapping[function][1], 16)
            except TypeError:
                return None
            inclusivelength = endingAddress - startingAddress
            if inclusivelength <= 0:
                return None
            totalBytesExtracted += inclusivelength
            binaryFile.seek(startingAddress)
            if binaryFile.tell() > fileSize:
                return None
            functionBytes = bytearray(binaryFile.read(inclusivelength))
            binaryFunctionBytes += functionBytes
            fucntionCount += 1
        
        if fucntionCount < 5:
            return None
    
    vectorizedBinary = torch.frombuffer(binaryFunctionBytes, dtype=torch.uint8)
    vectorizedBinary = vectorizedBinary.to(torch.int64)
    return vectorizedBinary

def generateFeatureMask(vectorizedBinary, functionMapping):
    featureMask = torch.zeros_like(vectorizedBinary)
    currentStartOffset = 0
    functionCount = 0
    totalBytesFeatured = 0
    for function in functionMapping:
        startingAddress = int(functionMapping[function][0], 16)
        endingAddress = int(functionMapping[function][1], 16)
        length = endingAddress - startingAddress
        for byteOffset in range(currentStartOffset, currentStartOffset + length - 1):
            featureMask[0, byteOffset] = functionCount
            totalBytesFeatured += 1
        functionCount += 1
        currentStartOffset = currentStartOffset + length
    
    return featureMask

def findMaliciousFunctions(attributions, functionMapping):
    maliciousFunctions = []
    
    attributionsNP = attributions.detach().numpy()
    
    meanAttributions = np.mean(np.abs(attributionsNP), axis=0)

    sortedAttributions = np.argsort(meanAttributions)[::-1]
    
    for i in range(5):
        maliciousFunctionIndex = sortedAttributions[i]
        functionName = list(functionMapping.keys())[maliciousFunctionIndex]
        maliciousFunctions.append(functionName)
    
    return maliciousFunctions

def modelInterpretation(vectorizedBinary, flem, functionMapping, ks):
    if vectorizedBinary.dim() == 1:
        vectorizedBinary = vectorizedBinary.unsqueeze(0)
    
    with torch.no_grad():
        logits = flem(vectorizedBinary)
    
    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)
    if(predictions == 1):
        print('\t Model predicted correctly')
    
    featureMask = generateFeatureMask(vectorizedBinary,  functionMapping)
    try:
        attr = ks.attribute(vectorizedBinary, target=0, feature_mask=featureMask, return_input_shape=False)
    except (ValueError):
        return None

    return attr

def shap_forward_wrapper(model):
    def forward_func(perturbed_input):
        result = model(perturbed_input)
        if torch.isnan(result).any():
            print(f"NaN encountered in SHAP perturbation with input: {perturbed_input}")
        return result
    return forward_func

def main():
    global GLOBAL_FLEM_MODEL
    
    warnings.filterwarnings("ignore")

    configuration = MalConvConfig(num_labels=2, pad_token_id=257)
    flem = MalConvForSequenceClassification(configuration)
    flem.load_state_dict(torch.load(GLOBAL_FLEM_MODEL, weights_only=True))
    flem.eval()
    print('[+] Loaded in FLEM - raw byte malware classifier')
    
    ks = KernelShap(shap_forward_wrapper(flem))
    print('[+] Created Kernel Shap Instance')
    
    with open(GLOBAL_EXE_MAP, 'r') as json_data:
        exe_map = json.load(json_data)
    
    print('[+] Loaded in Executable Mapping of all Dissasembled Files')
    
    for dissassembledFile in exe_map:
        print(f'[+] Processing {dissassembledFile}')
        
        functionMapping = exe_map[dissassembledFile]
        print('\tObtained Mapping of Functions')
        
        malwareLocation = findBinary(dissassembledFile)
        if malwareLocation == '':
            continue
        
        print(f'\tSample Located at: {malwareLocation}')
        
        vectorizedBinary = extractFunctionsFromBinary(malwareLocation, functionMapping)
        if vectorizedBinary == None:
            print('\tUnable to obtain vectorized representation of binary')
            continue
        
        if vectorizedBinary.size(0) < GLOBAL_MINIMUM_EXECUTABLE_SIZE:
            print(f'\tThis sample is far too small ignoring it.')
            continue
        
        print(f'\tObtained vectorized representaiton of binary {vectorizedBinary.shape}\n{vectorizedBinary}')
        
        attr = modelInterpretation(vectorizedBinary, flem, functionMapping, ks)
        if attr == None:
            print('\tProblematic input binary skipping')
            continue
        print('\tFinished model interpretation with malware sample')
        
        maliciousFunctions = findMaliciousFunctions(attr, functionMapping)
        print(f'\tMost malicious functions: {maliciousFunctions}')
        
        updateJSONData(malwareLocation.split('/')[-1], maliciousFunctions)

    with open(GLOBAL_JSON_FILE_LOCATION, 'w') as mapFile:
        json.dump(GLOBAL_JSON_DATA, mapFile, indent=4, sort_keys=True)
    
    print('[+] Interpretation Session Complete')
    
if __name__ == '__main__':
    main()
