'''
@brief Implementation of MalConv

@author Luke Kurlandski (lk3591)
@author Sean Kannanaikal (stk5106)
'''

import math
import torch
import os
from torch import nn, Tensor
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

import pe_sections



from datetime import datetime

GLOBAL_BENIGN_EXECUTABLES_LOCATION = "/home/stk5106/raw_byte_classifier/dataset/benign/" #folder containing benign PEs
GLOBAL_MALICIOUS_EXECUTABLES_LOCATION = "/home/stk5106/raw_byte_classifier/dataset/malware" #folder containing malicious PEs
GLOBAL_MAXIMUM_EXECUTABLE_SIZE = 100000
GLOBAL_LOG_FILE = '/home/stk5106/raw_byte_classifier/outputLog'

class MalConvConfig:

    """
    Configuration used by original authors:

        >>> MalConvConfig(
                vocab_size=257,
                embedding_size=8,
                channels=128,
                stride=500,
                kernel_size=500,
                mlp_hidden_size=128,
                pad_token_id=0,
            )
    """

    def __init__(
        self,
        vocab_size: int = 264,
        embedding_size: int = 256,
        channels: int = 128,
        stride: int = 512,
        kernel_size: int = 512,
        mlp_hidden_size: int = 128,
        pad_token_id: int = 0,
        num_labels: int = -1,
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.channels = channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.mlp_hidden_size = mlp_hidden_size
        self.pad_token_id = pad_token_id
        self.num_labels = num_labels


class MalConv(nn.Module):

    def __init__(self, config: MalConvConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            padding_idx=config.pad_token_id,
        )
        
        self.conv_1 = nn.Conv1d(
            in_channels=config.embedding_size,
            out_channels=config.channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
        )
        self.conv_2 = nn.Conv1d(
            in_channels=config.embedding_size,
            out_channels=config.channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, input_ids: Tensor) -> Tensor:

        # B: batch size
        # L: sequence length
        # E: embedding size
        # C: channels
        # S: stride

        B = input_ids.shape[0]
        L = input_ids.shape[1]
        E = self.config.embedding_size
        C = self.config.channels
        S = math.floor((L - self.config.kernel_size) / self.config.stride + 1)

        input_ids: Tensor                                                 # [B, L]
        assert tuple(input_ids.shape) == (B, L), f"{input_ids.shape} != {(B, L)}"

        input_embeddings: Tensor = self.embed(input_ids).transpose(1, 2)  # [B, E, L]
        assert tuple(input_embeddings.shape) == (B, E, L), f"{input_embeddings.shape} != {(B, E, L)}"

        cnn_1_value: Tensor = self.conv_1(input_embeddings)               # [B, C, S]
        assert tuple(cnn_1_value.shape) == (B, C, S), f"{cnn_1_value.shape} != {(B, C, S)}"

        cnn_2_value: Tensor = self.conv_2(input_embeddings)               # [B, C, S]
        assert tuple(cnn_2_value.shape) == (B, C, S), f"{cnn_2_value.shape} != {(B, C, S)}"

        gating_value: Tensor = cnn_1_value * F.sigmoid(cnn_2_value)       # [B, C, S]
        assert tuple(gating_value.shape) == (B, C, S), f"{gating_value.shape} != {(B, C, S)}"

        pooled_value: Tensor = self.pooling(gating_value)                 # [B, C, 1]
        assert tuple(pooled_value.shape) == (B, C, 1), f"{pooled_value.shape} != {(B, C, 1)}"

        hidden_states: Tensor = pooled_value.squeeze(-1)                  # [B, C]
        assert tuple(hidden_states.shape) == (B, C), f"{hidden_states.shape} != {(B, C)}"

        return hidden_states


class MalConvForSequenceClassification(nn.Module):

    def __init__(self, config: MalConvConfig):
        super().__init__()
        self.malconv = MalConv(config)
        self.clf_head = nn.Sequential(
            nn.Linear(config.channels, config.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(config.mlp_hidden_size, config.num_labels),
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden_states = self.malconv.forward(input_ids)
        logits = self.clf_head.forward(hidden_states)
        return logits

class ExecutableDataset(Dataset):

    def __init__(self, annotations_file, datadirectory, transform=None):
        self.dataframe = pd.read_csv(annotations_file, sep=',', header=None)
        self.datadirectory = datadirectory

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, index):
        global GLOBAL_BENIGN_EXECUTABLES_LOCATION
        global GLOBAL_MALICIOUS_EXECUTABLES_LOCATION

        filename = ''
        benignSamples = os.listdir(GLOBAL_BENIGN_EXECUTABLES_LOCATION)
        maliciousSamples = os.listdir(GLOBAL_MALICIOUS_EXECUTABLES_LOCATION)
        #update these range numbers
        if(index >= 0 and index <= len(benignSamples) - 1):
            
            filename = f'/benign/{benignSamples[index]}'

        elif(index >= len(benignSamples) and index <= len(benignSamples) + len(maliciousSamples)):
            filename = f'/malware/{maliciousSamples[index - len(benignSamples)]}'

        else:
            print('[-] Invalid Index Requested')
            exit(0)
        
        filepath = f'/home/stk5106/raw_byte_classifier/dataset{filename}'
        binaryDataVectorized, fake = readBinary(filepath)
        
        if fake == 1:
            return binaryDataVectorized, 0

        label = self.dataframe.iloc[index, 1]
        return binaryDataVectorized, label

def custom_collate(batch):
    binaryVectors = [executable[0] for executable in batch]
    maliciousLabels = torch.tensor([label[1] for label in batch])
    paddedBinaryVectors = pad_sequence(binaryVectors, batch_first=True, padding_value=257)
    return paddedBinaryVectors, maliciousLabels

def readBinary(path):
    global GLOBAL_MAXIMUM_EXECUTABLE_SIZE
    binaryDataRaw = b''
    fake = 0
    with open(path, "rb") as executable:
        
        extractor = pe_sections.GetExecutableSectionBounds(path)
        bounds, error = extractor('lief')
        
        for sectionByteRange in bounds:
            length = sectionByteRange[1] - sectionByteRange[0]
            if length > 0 and sectionByteRange[0] >= 0:
                executable.seek(sectionByteRange[0])
                binaryDataRaw += bytearray(executable.read(length))

        #binaryDataRaw = bytearray(executable.read(GLOBAL_MAXIMUM_EXECUTABLE_SIZE))
        
        if(binaryDataRaw == b''):
            binaryDataRaw = b'\x00'
            fake = 1
            
        binaryDataVector = torch.frombuffer(binaryDataRaw[:GLOBAL_MAXIMUM_EXECUTABLE_SIZE], dtype=torch.uint8)
        binaryDataVector = binaryDataVector.to(torch.int64)
        return binaryDataVector, fake
        
    return b''
def validate(model, validationLoader, lossFunction, device, datasetSize):
    model.eval()
    runningLoss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in validationLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = lossFunction(outputs, labels)
            runningLoss += loss.item() * labels.size(0)
            for i in range(len(outputs)):
                correct += checkPrediction(outputs[i], labels[i])
            
    accuracy = 100 * correct / len(validationLoader.dataset) 
    print(f'Model Validation Accuracy: {accuracy}%')

    validationLoss = runningLoss / len(validationLoader.dataset)
    return validationLoss

def checkPrediction(logit, label):
    predicted_labels = torch.argmax(logit, dim=0)
    
    if predicted_labels.item() == label.item():
        return 1
    
    return 0
    

def train(model, trainingLoader, lossFunction, optimizer, device, datasetSize):
    model.train()
    runningLoss = 0.0
    correct = 0
    for images, labels in trainingLoader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = lossFunction(outputs, labels)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item() * labels.size(0)
        for i in range(len(outputs)):
            correct += checkPrediction(outputs[i], labels[i])
    
    accuracy = 100 * correct / len(trainingLoader.dataset) 
    
    global GLOBAL_LOG_FILE
    with open(GLOBAL_LOG_FILE) as log:
        log.write(f'Model Training Accuracy: {accuracy}%')
        
    trainingLoss = runningLoss / len(trainingLoader.dataset)
    return trainingLoss

def main():
    dataDirectory = '/home/stk5106/raw_byte_classifier/dataset/'
    labels = '/home/stk5106/raw_byte_classifier/dataset/labels.csv'
    dataset = ExecutableDataset(labels, dataDirectory, None)

    datasetLength = dataset.__len__()

    trainingSetSize = int(0.8 * datasetLength)
    validationSetSize = datasetLength - trainingSetSize

    trainingDataset, validationDataset = torch.utils.data.random_split(dataset, [trainingSetSize, validationSetSize])
    
    print('[+] Split Dataset into training and validation set!')

    trainingLoader = DataLoader(trainingDataset, batch_size=20, shuffle=True, collate_fn=custom_collate)
    validationLoader = DataLoader(validationDataset, batch_size=20, shuffle=True, collate_fn=custom_collate)
    
    print('[+] Created loaders for each dataset')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
    configuration = MalConvConfig(num_labels=2, pad_token_id=257)
    model = MalConvForSequenceClassification(configuration)
    model.to(device)
    
    print('[+] Moved Model Over to GPU')

    #TODO: look more into the loss and optimizer function
    lossFunction = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochCount = 5

    trainingLosses, validationLosses = [], []

    for epoch in range(epochCount):
        print(f'[+] Starting Training Sequence {epoch}')
        trainingLoss = train(model, trainingLoader, lossFunction, optimizer, device, trainingSetSize)
        trainingLosses.append(trainingLoss)

        print(f'[+] Starting Validation Sequence {epoch}')
        validationLoss = validate(model, validationLoader, lossFunction, device, validationSetSize)
        validationLosses.append(validationLoss)

        global GLOBAL_LOG_FILE
        with open(GLOBAL_LOG_FILE) as log:
            log.write(f"Epoch {epoch + 1}/{epochCount} - Training Loss: {trainingLoss}, Validation Loss: {validationLoss}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_path = f"/home/stk5106/raw_byte_classifier/models/modelCodeOnlySunday_{timestamp}_{epoch}"
        
        torch.save(model.state_dict(), model_path)
        
        print('[+] Model Saved')

if __name__ == "__main__":
    main()
