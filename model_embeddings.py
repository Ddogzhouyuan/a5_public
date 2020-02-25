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
import torch
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
        self.word_embed_size = word_embed_size
        self.char_embed_size = 50
        self.max_word_len = 21
        self.dropout_rate = 0.3
        self.vocab = vocab

        self.char_embedding = nn.Embedding(num_embeddings=len(self.vocab.char2id), embedding_dim=self.char_embed_size, padding_idx=vocab.char2id['<pad>'])
        self.CNN = CNN(k=5, f=self.word_embed_size, embSizeChar=self.char_embed_size, maxLengthOfWords=self.max_word_len)
        self.Highway = Highway(embSize=self.word_embed_size)
        self.dropout = nn.Dropout(self.dropout_rate)

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
        # char_embeddings = self.char_embedding(input)
        # sent_len, batch_size, max_word, _ = char_embeddings.shape
        # char_embeddings = char_embeddings.view((sent_len * batch_size, max_word, self.char_embed_size)).transpose(2, 1)
        # X_conv = self.CNN(char_embeddings)
        # X_highway = self.Highway(X_conv)
        # output = self.dropout(X_highway)
        # output = output.view((sent_len, batch_size, self.word_embed_size))
        # return output
        X_word_emb_list = []
        # divide input into sentence_length batchs
        for X_padded in input:
            X_emb = self.char_embedding(X_padded)
            X_reshaped = torch.transpose(X_emb, dim0=-1, dim1=-2)
            # conv1d can only take 3-dim mat as input
            # so it needs to concat/stack all the embeddings of word
            # after going through the network
            X_conv_out = self.CNN(X_reshaped)
            X_highway = self.Highway(X_conv_out)
            X_word_emb = self.dropout(X_highway)
            X_word_emb_list.append(X_word_emb)

        X_word_emb = torch.stack(X_word_emb_list)
        return X_word_emb
        ### END YOUR CODE

