#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, embSize):
        super(Highway, self).__init__()
        self.linearProj = nn.Linear(embSize, embSize)
        self.gateProj = nn.Linear(embSize, embSize)
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()


    def forward(self, X_conv_out):
        X_proj = self.relu(self.linearProj(X_conv_out))
        X_gate = self.sigmod(self.gateProj(X_conv_out))
        X_highway = X_gate * X_proj + (1 - X_gate) * X_conv_out
        return X_highway





    ### END YOUR CODE

