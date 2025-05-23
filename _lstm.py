# _lstm.py contains functions that support the management of training and testing data as well as the training and testing process for the LSTM model. 


## PRELIMINARIES ##
# general 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
import random as rn
import csv
from tqdm import tqdm

# for lstm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable 

## DEFINE PARAMETERS ##
num_epochs = 2500         # number of training repetitions
learning_rate = 0.00001   # learning rate
input_size = 14           # number of features in input
batch_size = 1            # batch size
output_size = 1           # number of output features (just SWE) 
hidden_size = 256         # number of features in hidden state
num_layers = 2            # number of stacked lstm layers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## DEFINE LSTM ##
# adapted from https://cnvrg.io/pytorch-lstm/ 
class LSTM(nn.Module):
    def __init__(self, input_size, batch_size, output_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size           # input size
        self.batch_size = batch_size           # batch size
        self.output_size = output_size         # output_size 
        self.num_layers = num_layers           # number of recurrent layers of LSTM
        self.hidden_size = hidden_size         # number of features in hidden state

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=0.4)
        
        self.fc = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
    
    def forward(self,x):
        # flatten parameters
        self.lstm.flatten_parameters()
        
        # establish initial layers, send to device
        h_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size)  # hidden state
        c_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size)  # internal state
        h_0 = h_0.to(DEVICE) 
        c_0 = c_0.to(DEVICE)
        
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        out = self.relu(output) # experimental relu layer
        out = self.fc(out) # linear layer to translate output
        return out
        #return out.view(-1)

# ## TRAIN MODEL ##
# # returns loss and validation records
# def train_lstm(lstm, train_swe_tensors, train_non_swe_tensors, test_swe_tensors, test_non_swe_tensors):
#     # define loss function and optimizers
#     loss_fn = torch.nn.SmoothL1Loss()                                 # loss function
#     optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate) # optimizer and solver: Adam

#     # train model - doesn't work for 1 yr of data
#     loss_record = np.zeros(num_epochs)
#     val_record = np.zeros(num_epochs)
#     for epoch in tqdm(range(num_epochs)):
#         # training
#         for i in range(len(train_swe_tensors)):
#             # reshape to be in format (sequence_length, batch_size, input_size)
#             x_train = torch.reshape(train_non_swe_tensors[i], (train_non_swe_tensors[i].shape[0], 1, train_non_swe_tensors[i].shape[1]))
#             y_train = train_swe_tensors[i]
#             #torch.reshape(train_swe_tensors[i], (1, train_swe_tensors[i].shape[0], train_swe_tensors[i].shape[1]))
    
#             outputs = lstm.forward(x_train) #forward pass
#             optimizer.zero_grad() #caluclate the gradient, manually setting to 0
    
#             # obtain the loss function
#             loss = loss_fn(outputs.reshape(outputs.shape[0],1), y_train)
    
#             loss.backward() #calculate the loss of the loss function
    
#             optimizer.step() #improve from loss, i.e backprop
    
#         # validation
#         with torch.no_grad():
#             for i in range(len(test_swe_tensors)):
#                 # reshape to be in format (sequence_length, batch_size, input_size)
#                 x_test = torch.reshape(test_non_swe_tensors[i], (test_non_swe_tensors[i].shape[0], 1, test_non_swe_tensors[i].shape[1]))
#                 y_test = test_swe_tensors[i]
        
#                 output_val = lstm.forward(x_test) #forward pass
        
#                 # obtain the loss function
#                 loss_val = loss_fn(outputs.reshape(output_val.shape[0],1), y_test)
       
#         loss_record[epoch] = loss.item()
#         val_record[epoch] = loss_val.item()
    
#         if epoch % 500 == 0:
#             print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
#             print("Epoch: %d, loss: %1.5f" % (epoch, loss_val.item())) 

#     return loss_record, val_record

## TRAIN MODEL ##
# returns loss records
def train_lstm(lstm, train_swe_tensors, train_non_swe_tensors, test_swe_tensors, test_non_swe_tensors):
    # define loss function and optimizers
    loss_fn = torch.nn.SmoothL1Loss()                                 # loss function
    optimizer = torch.optim.AdamW(lstm.parameters(), lr=learning_rate) # optimizer and solver: Adam

    # train model - doesn't work for 1 yr of data
    loss_record = np.zeros(num_epochs)
    for epoch in tqdm(range(num_epochs)):
        # training
        for i in range(len(train_swe_tensors)):
            # reshape to be in format (sequence_length, batch_size, input_size)
            x_train = torch.reshape(train_non_swe_tensors[i], (train_non_swe_tensors[i].shape[0], 1, train_non_swe_tensors[i].shape[1]))
            y_train = train_swe_tensors[i]
            #torch.reshape(train_swe_tensors[i], (1, train_swe_tensors[i].shape[0], train_swe_tensors[i].shape[1]))
    
            outputs = lstm.forward(x_train) #forward pass
            optimizer.zero_grad() #caluclate the gradient, manually setting to 0
    
            # obtain the loss function
            loss = loss_fn(outputs.reshape(outputs.shape[0],1), y_train)
    
            loss.backward() #calculate the loss of the loss function
    
            optimizer.step() #improve from loss, i.e backprop
            
        loss_record[epoch] = loss.item()
    
        if epoch % 500 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))  

    return loss_record
