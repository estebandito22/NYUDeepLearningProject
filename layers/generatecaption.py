from layers.beamsearch import Beam
import torch.nn.functional as functional
import torch


class GenerateCaption(object):

    def __init__(self, decoder_rnn, h_t, c_t, isequence, osequence,
                 word_embedding_layer, decoder_embedding_layer, rnn_type,
                 beam_width, vocab_bosindex, vocab_eosindex, cuda=False,
                 max_words=100, alpha=0.7):
        self.decoder_rnn = decoder_rnn
        self.h_t = h_t
        self.c_t = c_t
        self.isequence = isequence
        self.osequence = osequence
        self.word_embedding_layer = word_embedding_layer
        self.decoder_embedding_layer = decoder_embedding_layer
        self.rnn_type = rnn_type
        self.beam_width = beam_width
        self.vocab_bosindex = vocab_bosindex
        self.vocab_eosindex = vocab_eosindex
        self.cuda = cuda
        self.max_words = max_words
        self.alpha = alpha

        self.beam = Beam(self.beam_width, self.vocab_bosindex,
                    self.vocab_eosindex, alpha=self.alpha, cuda=self.cuda)

    def _init_step(self):

        # Takes the initial step and duplicates the decoder rnn by beam_width
        self.step = 0
        input = self.isequence[self.step]
        if self.rnn_type == 'LSTM':
            h_t, c_t = self.decoder_rnn(input, (self.h_t, self.c_t)) #h_t: batch_size*hidden_dim
        elif self.rnn_type == 'GRU':
            h_t = self.decoder_rnn(input, self.h_t) #h_t: batch_size*hidden_dim
        elif self.rnn_type == 'RNN':
            pass

        # Generate log probabilities for the next step and duplicate by
        # beam_width
        self.osequence[self.step] = self.decoder_embedding_layer(h_t)
        cur_osequence_probs = functional.log_softmax(self.osequence[self.step],
                                                     dim=1)
        self.osequence_probs_beams = cur_osequence_probs.expand(self.beam_width,-1)

        # Advance the beam and duplicate inputsequence by beam_width
        self.done = self.beam.advance(self.osequence_probs_beams)
        self.hyp_vectors = [self.word_embedding_layer(self.beam.get_hyp(i)[-1]) for \
                            i in range(self.beam_width)]
        self.isequences = [torch.cat([self.isequence.clone(),
                                      hyp_vector.unsqueeze(0)]) for \
                           hyp_vector in self.hyp_vectors]

        # Duplicate osequence and hidden states by beam_width
        self.osequences = [self.osequence.clone() for _ in range(self.beam_width)]
        self.h_ts = [h_t.clone() for _ in range(self.beam_width)]
        self.c_ts = [c_t.clone() for _ in range(self.beam_width)]

        if not self.done:
            for i, osequence in enumerate(self.osequences):
                #extend osequence, will be overwritten next iteration
                self.osequences[i] = torch.cat([osequence,
                                                self.osequence[self.step].\
                                                unsqueeze(0).clone()])

    def _advance(self):
        beam_idx = 0
        self.step += 1
        # produce the log probabilities for each beam
        for isequence, osequence, h_t, c_t in zip(self.isequences, self.osequences,
                                                  self.h_ts, self.c_ts):

            input = isequence[self.step]
            if self.rnn_type == 'LSTM':
                h_t, c_t = self.decoder_rnn(input, (h_t, c_t)) #h_t: batch_size*hidden_dim
            elif self.rnn_type == 'GRU':
                h_t = self.decoder_rnn(input, h_t) #h_t: batch_size*hidden_dim
            elif self.rnn_type == 'RNN':
                pass

            osequence[self.step] = self.decoder_embedding_layer(h_t)
            cur_osequence_probs = functional.log_softmax(osequence[self.step], dim=1)
            self.osequence_probs_beams[beam_idx] = cur_osequence_probs
            beam_idx += 1

        # Advance the beam, get the new word embedding vectors
        self.done = self.beam.advance(self.osequence_probs_beams)
        self.hyp_vectors = [self.word_embedding_layer(self.beam.get_hyp(i)[-1]) for \
                            i in range(self.beam_width)]

        # Determine the origins for the hypotheses and keep only the
        # input sequences related to these origins
        self.hyp_origins = self.beam.get_current_origin()
        self.isequences = [self.isequences[origin.data[0]] for origin in self.hyp_origins]
        self.isequences = [torch.cat([self.isequences[i],
                                      hyp_vector.unsqueeze(0)]) for
                           i, hyp_vector in enumerate(self.hyp_vectors)]

        if not self.done:
            for i, osequence in enumerate(self.osequences):
                #extend osequence, will be overwritten next iteration
                self.osequences[i] = torch.cat([osequence,
                                       osequence[self.step].unsqueeze(0)])

    def generate_caption(self):
        self._init_step()
        for i in range(self.max_words-1):
            if self.done == True:
                break
            self._advance()

        return self.beam.get_hyp(0)
