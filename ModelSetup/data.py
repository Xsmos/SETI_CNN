import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from random import shuffle
import sys, os, time, h5py

def read_data(fin, seed, mode, normalize):
    trainfile = h5py.File(fin)
    shape1 = trainfile['figure'].shape[0]
    shape2 = (trainfile['figure'].shape[1])*(trainfile['figure'].shape[2])
    shape3 = trainfile['figure'].shape[-1]
    flattenedtrain = np.array(trainfile['figure']).reshape(shape1, shape2, shape3)
    labels = np.array(trainfile['target'][:], dtype = 'int')
    trainfile.close()


    if normalize:
        normalized = []
        for each in flattenedtrain:
            imNorm = (each - np.min(each))/(np.max(each) - np.min(each))
            normalized.append(imNorm)
        finalTrain = np.array(normalized, dtype = 'float32')
    else:
        finalTrain = np.array(flattenedtrain, dtype = 'float32')
    
    elements = finalTrain.shape[0]

    np.random.seed(seed)
    indices = np.arange(elements)
    np.random.shuffle(indices)

    if   mode=='train':   size, offset = int(elements*0.75), int(elements*0.00)
    elif mode=='valid':   size, offset = int(elements*0.10), int(elements*0.75)
    elif mode=='test':    size, offset = int(elements*0.15), int(elements*0.85)
    elif mode=='all':     size, offset = int(elements*1.00), int(elements*0.00)
    else:                 raise Exception('Wrong name!')

    indices = indices[offset:offset+size]
    finalTrain, labels = finalTrain[indices], labels[indices]
    finalTrain = finalTrain.reshape(finalTrain.shape[0], 1, finalTrain.shape[1], finalTrain.shape[2])
    print(finalTrain.shape)
    return finalTrain, labels

class make_dataset():
    def __init__(self, mode, seed, fin):
        inp, out = read_data(fin, seed, mode, normalize=True)
        self.size = inp.shape[0]
        self.input = torch.tensor(inp, dtype=torch.float32)
        self.output = torch.tensor(out, dtype=torch.float32)

        print(f'Feature size: {np.shape(self.input)}\nLabel size: {np.shape(self.output)}')

    def __len__(self):
        return self.size
    
    def __getitem__(self,idx):
        return self.input[idx], self.output[idx]
    
def create_dataset(mode, seed, fin, batch_size, shuffle=False):
    data = make_dataset(mode,seed,fin)
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)
    return data_loader
