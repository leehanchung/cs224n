#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    """
    Highway Layer for combining outputs of Conv1D and skip connection
    """
    def __init__(self, emb_size: int):
        """
        Initialize two linear layers with bias.

        @param emb_size (int): word embedding size. e_{word} in project assignment.
        """
        super(Highway, self).__init__()
        self.emb_size = emb_size
        self.proj = nn.Linear(in_features=self.emb_size,
                              out_features=self.emb_size,
                              bias=True)
        self.gate = nn.Linear(in_features=self.emb_size,
                              out_features=self.emb_size,
                              bias=True)

    def forward(self, X_conv_out: torch.Tensor) -> torch.Tensor:
        """
        Highway connection, formula (8), (9), (10)

        @param x_conv_out (torch.Tensor): output of conv1D layer, shape (batch_size, emb_size)
        @outupt x_highway (torch.Tensor): output of highway layer, shape (batch_size, emb_size)
        """
        X_proj = F.relu(self.proj(X_conv_out))
        X_gate = torch.sigmoid(self.gate(X_conv_out))

        X_highway = X_proj * X_gate + (1 - X_gate) * X_conv_out

        return X_highway

    ### END YOUR CODE

if __name__ == "__main__":
    pass