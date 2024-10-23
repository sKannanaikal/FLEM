import re
import json
import os

GLOBAL_JSON_DATA = {}
GLOBAL_DATASET_PATH = '/home/stk5106/raw_byte_classifier/dataset/disassembled'
GLOBAL_JSON_FILE = '/home/stk5106/raw_byte_classifier/function_offsets.json'

def analyzeDissassembled(file):
    global GLOBAL_JSON_DATA

    fileMap = {}
    
    with open(f'{GLOBAL_DATASET_PATH}/{file}', 'r') as dissassembly:
        code = dissassembly.read()
    
    functions = code.split('\n\n')
    addressesPattern = r"(?<=\s)[A-Fa-f0-9]{8}(?=\s)"
    functionNamePattern = r"\b[_a-zA-Z][a-zA-Z0-9_@<>=]*(?=\s*\()"
    
    for function in functions:
        addresses = re.findall(addressesPattern, function)
        
        if len(addresses) >= 2:
            startingAddress = addresses[0]
            endingAddress = addresses[-2]
        else:
            continue
        
        match = re.search(functionNamePattern, function)
        if match is not None:
            functionName = match.group()
            fileMap[functionName] = (startingAddress, endingAddress)
        else:
            continue
    
    GLOBAL_JSON_DATA[file] = fileMap  
    
def main():
    test = '017799f79d8c33e6ae8fb28a56555a866ac0092e2dd93235c46c2c8cfbde4f01.asm'
    testFunction = 'Catch_All@004015bf'
    dissassemblyFiles = os.listdir(GLOBAL_DATASET_PATH)
    
    for file in dissassemblyFiles:
        print(f"[+] Analyzing {file}")
        analyzeDissassembled(file)
    
    with open(GLOBAL_JSON_FILE, 'w') as jsonFile:
        json.dump(GLOBAL_JSON_DATA, jsonFile, indent=2)

if __name__ == '__main__':
    main()
