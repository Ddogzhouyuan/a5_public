#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, f, embSizeChar, maxLengthOfWords, k=5):
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=embSizeChar, out_channels=f, kernel_size=k)
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool1d(maxLengthOfWords - k + 1)

    def forward(self, X_reshaped):
        X_conv = self.conv1d(X_reshaped)
        X_conv_out = self.maxPool(self.relu(X_conv))
        X_conv_out = X_conv_out.squeeze()
        return X_conv_out

    ### END YOUR CODE

