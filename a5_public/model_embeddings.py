#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h

        self.num_embedding = len(vocab.char2id)
        self.char_embed_size = 50
        self.word_embed_size = word_embed_size
        self.max_len = 21
        self.dropout_rate = 0.3
        self.padding_idx = vocab.char2id['<pad>']

        self.embedding = nn.Embedding(num_embeddings=self.num_embedding,
                                      embedding_dim=self.char_embed_size,
                                      padding_idx=self.padding_idx)
        self.cnn = CNN(char_embed_size=self.char_embed_size,
                       word_embed_size=self.word_embed_size,
                       max_len=self.max_len)
        self.highway = Highway(word_emb_size=self.word_embed_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h

        # input: (sentence_length, batch_size, max_word_length)
        # flatten sentence_length and batch_size together to fit the network
        sentence_length, batch_size, max_len = input.shape
        input_reshape = input.contiguous().view(sentence_length*batch_size, max_len)
        
        # (sentence_length*batch_size, max_word_length)
        X_embed = self.embedding(input_reshape)

        # (sentence_length*batch_size, max_word_length, char_embed_size)
        # Necessary to permute because the PyTorch Conv1D function performs the convolution 
        # only on thelastdimension of the input
        X_reshaped = X_embed.permute(0, 2, 1)
        
        # (sentence_length*batch_size, char_embed_size, max_word_length)
        X_conv_out = self.cnn(X_reshaped)

        # (sentence_length*batch_size, word_embed_size)
        X_highway = self.highway(X_conv_out)

        # (sentence_length*batch_size, word_embed_size)
        X_word_embed = self.dropout(X_highway)

        # (sentence_length*batch_size, word_embed_size)
        output = X_word_embed.contiguous().view(sentence_length, batch_size, self.word_embed_size)

        # (sentence_length, batch_size, word_embed_size)
        return output

        ### END YOUR CODE

