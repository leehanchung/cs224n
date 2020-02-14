#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
tests.py: sanity checks for assignment 5
Usage:
    test_modules.py
"""
import unittest
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils

from highway import Highway
from cnn import CNN


#----------
# CONSTANTS
#----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 4
DROPOUT_RATE = 0.0
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

def initialize_layers(model: nn.Module):
    """ 
    Reference: https://pytorch.org/docs/stable/nn.html#torch.nn.Module.apply

    @param model: initialize layer weights for test cases.
    """
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.5)
            if m.bias is not None:
                m.bias.data.fill_(0.5)
        elif type(m) == nn.Conv1d:
            m.weight.data.fill_(0.5)
            if m.bias is not None:
                m.bias.data.fill_(0.5)
        elif type(m) == nn.Embedding:
            m.weight.data.fill_(0.15)
        elif type(m) == nn.Dropout:
            nn.Dropout(DROPOUT_RATE)

    #with torch.no_grad():
    model.apply(init_weights)
    return model

class TestHighway(unittest.TestCase):
    """ 
    Sanity check for Highway module
    """
    def setUp(self):
        self.word_emb_size = EMBED_SIZE
        self.model = Highway(word_emb_size=EMBED_SIZE)

    def test_Highway_shape(self):
        self.model = initialize_layers(self.model)
        
        input = torch.zeros(BATCH_SIZE, EMBED_SIZE)
        output = self.model.forward(input)
        
        self.assertEqual(input.shape, output.shape, "incorrect output shape")
        self.assertEqual(self.model.proj.in_features, EMBED_SIZE, "incorrect proj layer input size")
        self.assertEqual(self.model.proj.out_features, EMBED_SIZE, "incorrect proj layer output size")
        self.assertEqual(self.model.gate.in_features, EMBED_SIZE, "incorrect gate layer input size")
        self.assertEqual(self.model.gate.out_features, EMBED_SIZE, "incorrect gate layer output size")
    
    def test_Highway_forward(self):
        
        def generate_data():
            """
            generate test data
            """
            X_conv_input = np.random.rand(BATCH_SIZE, EMBED_SIZE)
            W_proj = np.ones((EMBED_SIZE, EMBED_SIZE)) * 0.5
            b_proj = np.ones(EMBED_SIZE) * 0.5

            W_gate = np.ones((EMBED_SIZE, EMBED_SIZE)) * 0.5
            b_gate = np.ones(EMBED_SIZE) * 0.5

            def relu(input):
                return np.maximum(input, 0)

            def sigmoid(input):
                return 1. / (1 + np.exp(-input))

            X_proj = relu(X_conv_input.dot(W_proj) + b_proj)
            X_gate = sigmoid(X_conv_input.dot(W_gate) + b_gate)
            X_highway = X_gate * X_proj + (1 - X_gate) * X_conv_input

            return X_conv_input, X_highway

        self.model = initialize_layers(self.model)

        input, output = generate_data()
        input = torch.from_numpy(input.astype(np.float32))
        output = torch.from_numpy(output.astype(np.float32))

        with torch.no_grad():
            test_output = self.model(input)

        self.assertEqual(test_output.shape, output.shape, "incorrect output shape")
        self.assertTrue(np.allclose(test_output.numpy(), output.numpy()), "output values not close to equal")


if __name__ == '__main__':
    unittest.main(verbosity=2)
