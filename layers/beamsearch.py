"""Beam search implementation in PyTorch."""
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

# Code borrowed from PyTorch OpenNMT example
# https://github.com/pytorch/examples/blob/master/OpenNMT/onmt/Beam.py

import torch

# try:
#     from layers.sentencedecoder import SentenceDecoder
# except:
#     from sentencedecoder import SentenceDecoder

import torch
import torch.nn as nn
from torch.autograd import Variable

import layers.utils as utils
from layers.wordspretrained import PretrainedEmbeddings

class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, vocab_bosindex, vocab_eosindex, alpha, cuda=False,
                 min_length=5):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = vocab_eosindex
        self.bos = vocab_bosindex
        self.eos = vocab_eosindex
        self.alpha = alpha
        self.min_length = min_length
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

        # The attentions (matrix) for each time.
        self.attn = []

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk`: Compute and update
    # the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #
    # Returns: True if beam search is complete.

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores and normalized by length
        if len(self.prevKs) > 0:
            sentence_length = len(self.nextYs)
            beam_lk = workd_lk/sentence_length**self.alpha \
                      + self.scores.unsqueeze(1).expand_as(workd_lk) \
                      * (sentence_length-1)**self.alpha \
                      / sentence_length**self.alpha
            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i].data[0] == self.eos:
                    beam_lk[i] = -1e20
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0].data[0] == self.eos:
            self.done = True

        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[0], ids[0]

    # Get the score of the best in the beam.
    def get_best_k(self):
        k = self.size
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[:k], ids[:k]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]

if __name__=="__main__":

    pretrainedEmbeddings = PretrainedEmbeddings({"word_embeddings" : torch.randn(10,3),
                                                "pretrained_embdim" : 3,
                                                 "vocabulary_size":10,
                                                 "embeddings_requires_grad": False})

    dict_args = {
    				'input_dim' : 3, #pretrainedEmbeddings.pretrained_embdim
    				'rnn_hdim' : 3,
    				'rnn_type' : 'LSTM',
    				'vocabulary_size' : pretrainedEmbeddings.vocabulary_size,
    				'tie_weights' : True,
    				'word_embeddings' : pretrainedEmbeddings.embeddings.weight,
                    'vocabulary_bosindex': 1,
                    'vocabulary_eosindex': 0,
                    'pretrained_words_layer': pretrainedEmbeddings
    			}

    sentenceDecoder = SentenceDecoder(dict_args)
    osequence = sentenceDecoder(Variable(torch.randn(2,3,3)), Variable(torch.randn(2,3)), Variable(torch.LongTensor([[1,1,1],[1,0,0]])))

    class Vocab(object):
        def __init__(self):
            self.word2index = {'<eos>': 0, '<bos>': 1, 'this': 2, 'is': 3, 'a': 4, 'test': 5,
            		 'of': 6, 'the': 7, 'beam': 8, 'search': 9}
            self.index2word = {0: '<eos>', 1: '<bos>', 2:'this', 3: 'is', 4: 'a', 5: 'test',
            6: 'of', 7: 'the', 8: 'beam', 9: 'search'}

    vocab = Vocab()
    beam = Beam(3, vocab_eosindex = 0, vocab_bosindex=1, alpha=0.7, cuda=False)

    import random
    i = random.randint(0,1)
    beam.advance(osequence[i])
    beam.get_hyp(0)
