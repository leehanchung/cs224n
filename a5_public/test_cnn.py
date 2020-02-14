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

class TestCNN(unittest.TestCase):
    """ 
    Sanity check for CNN module
    """
    def setUp(self):
        self.char_emb_size = EMBED_SIZE-1
        self.word_emb_size = EMBED_SIZE
        self.m_word = 10
        self.model = CNN(char_embed_size=self.char_emb_size,
                         word_embed_size=self.word_emb_size,
                         max_len=self.m_word)

    def test_CNN_shape(self):
        self.model = initialize_layers(self.model)
        
        input = torch.ones((BATCH_SIZE, self.char_emb_size, self.m_word))
        with torch.no_grad():  
            output = self.model.forward(input)
        
        self.assertEqual(output.shape, (BATCH_SIZE, self.word_emb_size), "incorrect output shape")
        self.assertEqual(self.model.conv1d.in_channels, self.char_emb_size, "incorrect input channel size")
        self.assertEqual(self.model.conv1d.out_channels, self.word_emb_size, "incorrect output channel size")
        
        
if __name__ == '__main__':
    unittest.main(verbosity=2)
