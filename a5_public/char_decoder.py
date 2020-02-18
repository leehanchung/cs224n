#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.

        # (length, batch_size)
        X_embed = self.decoderCharEmb(input)
        # (length, batch_size, char_embed_size)
        h_t, dec_hidden = self.charDecoder(X_embed, dec_hidden)
        # h_t, (length, batch_size, hidden_size)
        scores = self.char_output_projection(h_t)
        # scores, (length, batch_size, target_vocab size)
        return scores, dec_hidden

        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss

        self.padding_idx = self.target_vocab.char2id['<pad>']

        # batch sequence train input x_1...n with target x_2...n+1
        input = char_sequence[:-1]
        target = char_sequence[1:]

        # input, (length, batch_size)
        scores, dec_hidden = self.forward(input, dec_hidden)
        # (14), (15). skip padding and sum the loss instead of average. 
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx,
                                                      reduction='sum')
        
        # nn.CrossEntropyLoss takes (batch_size, number of classes),
        # thus we have to premute both scores and target batch size to the first dimension

        # scores, (length, batch_size, target_vocab size)
        # target, (length, batch_size)
        scores = scores.permute(1, 2, 0)
        target = target.permute(1, 0)
        # scores, (batch_size, target_vocab size, length)
        # target, (batch_size, length)
        loss_char_dec = self.cross_entropy_loss(scores, target)

        return loss_char_dec
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        batch_size = initialStates[0].shape[1]
        current_char = torch.tensor([[self.target_vocab.start_of_word] * batch_size],
                                    device=device)
        # current_char, (length, batch_size)
        dec_hidden = initialStates
        # initialize decodedWords as a list of empty string
        decodedWords = [''] * batch_size
        # initialize a flag for each decoded words since in test time, the model might
        # predict none end_of_word characters after we got an end_of_word and we don't
        # want to write those down.
        decodedWords_flag = [False] * batch_size
        
        for _ in range(max_length):
            scores, dec_hidden = self.forward(input=current_char,
                                              dec_hidden=dec_hidden)

            # scores, (length, batch_size, target_vocab_size)
            # argmax over the the vocab_size
            current_char = torch.argmax(scores, dim=-1)
        
            # current_char, (length, batch_size)
            # convert current_char tensor to a list of idx
            current_idx = current_char.tolist()[0]
        
            # convert the idx to char and add to each decodedWords if not start, end, or pad.
            for i, char_idx in enumerate(current_idx):#range(len(current_idx)):
                char = self.target_vocab.id2char[char_idx]
                if char is not '}' and not decodedWords_flag[i]:
                    # don't write to word when its start, end, pad, or unk token
                    if char not in ['{', '}', '<pad>', '<unk>']:
                        decodedWords[i] += char
                else:

                    # end_of_word seen. flag it to prevent further writing
                    decodedWords_flag[i] = True

        return decodedWords
        ### END YOUR CODE

