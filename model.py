from fastai.text import *


class BiLSTM(nn.Module):
    def __init__(self, batch_size, hidden_dim, device):
        super(BiLSTM, self).__init__()
        self.embedding_dim = 1024
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.batch_size = batch_size
        self.tag_size = 2
        # self.tag_to_ix = tag_to_ix
        # self.tagset_size = len(tag_to_ix)

        # self.elmo = Elmo(device)
        self.lstm = nn.GRU(self.embedding_dim, hidden_dim // 2, num_layers=self.num_layers, bidirectional=True,
                           batch_first=True)

        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, self.tag_size),
        )

        # # Matrix of transition parameters.  Entry i,j is the score of
        # # transitioning *to* i *from* j.
        # self.transitions = nn.Parameter(
        #     torch.randn(self.tagset_size, self.tagset_size))
        #
        # # These two statements enforce the constraint that we never transfer
        # # to the start tag and we never transfer from the stop tag
        # self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        # self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self._dev = device
        self.hidden = self.init_hidden(self.batch_size)
        self.to(self._dev)

    def init_hidden(self, bs):
        return torch.randn(self.num_layers * 2, bs, self.hidden_dim // 2).to(self._dev)

    def forward(self, lens, embs):
        """
        :param embs: (batch_size, maxlen, embedding_dim)
        :param lens: original sequence lengths of each input (batch_size)
        You should return a tensor of (batch_size x 2) logits for each class per batch element.
        """
        bs, maxlen, embedding_dim = embs.shape # hint: use this for reshaping RNN hidden state
        assert embedding_dim == self.embedding_dim

        # 1. Grab the embeddings:
        # embeddings for the token sequence are (batch_size, len, embedding_dim)
        # embeds = self.embedding_lookup(tokens)

        # 2. Sort seq_lens and embeds in descending order of seq_lens. (check out torch.sort)
        #    This is expected by torch.nn.utils.pack_padded_sequence.
        sorted_seq_lens, perm_ix = torch.sort(lens, descending=True)
        sorted_tokens = embs[perm_ix]

        # 3. Obtain a PackedSequence object from pack_padded_sequence.
        #    Be sure to pass batch_first=True as the first dimension of our input is the batch dim.
        packed_seq = nn.utils.rnn.pack_padded_sequence(sorted_tokens, sorted_seq_lens, batch_first=True)

        # 4. Apply the RNN over the sequence of packed embeddings to obtain a sentence encoding.
        #    Reset the hidden state each time
        self.hidden = self.init_hidden(bs)
        # encoding is (batch_size, seq_len, hidden_sz)
        encoding, _ = self.lstm(packed_seq, self.hidden)
        # undo the packing operation
        encoding, _ = nn.utils.rnn.pad_packed_sequence(encoding, batch_first=True)
        out = self.linear(encoding)

        # 6. Remember to unsort the output from step 5. If you sorted seq_lens and obtained a permutation
        #    over its indices (perm_ix), then the sorted indices over perm_ix will "unsort".
        #    For example:
        #       _, unperm_ix = perm_ix.sort(0)
        #       output = x[unperm_ix]
        #       return output
        _, unperm_ix = torch.sort(perm_ix)
        out = out[unperm_ix]

        # restore the original dimensions of the tensor
        padding = maxlen - out.shape[1]
        if padding != 0:
            padding = torch.zeros(size=(bs, padding, self.tag_size))
            out = torch.cat((out, padding), dim=1)
        return out

    def evaluate(self, sentence):
        """
        Evaluate a sentence as a list of words, outputting list as a numpy array
        """
        out = self.forward(sentence)
        out = torch.argmax(out, dim=1)
        return out.detach().numpy()

    # def _get_lstm_features(self, sentence):
    #     self.hidden = self.init_hidden()
    #     embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
    #     lstm_out, self.hidden = self.lstm(embeds, self.hidden)
    #     lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
    #     lstm_feats = self.hidden2tag(lstm_out)
    #     return lstm_feats
    #
    # def _score_sentence(self, feats, tags):
    #     # Gives the score of a provided tag sequence
    #     score = torch.zeros(1)
    #     tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
    #     for i, feat in enumerate(feats):
    #         score = score + \
    #                 self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
    #     score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
    #     return score
    #
    # def _viterbi_decode(self, feats):
    #     backpointers = []
    #
    #     # Initialize the viterbi variables in log space
    #     init_vvars = torch.full((1, self.tagset_size), -10000.)
    #     init_vvars[0][self.tag_to_ix[START_TAG]] = 0
    #
    #     # forward_var at step i holds the viterbi variables for step i-1
    #     forward_var = init_vvars
    #     for feat in feats:
    #         bptrs_t = []  # holds the backpointers for this step
    #         viterbivars_t = []  # holds the viterbi variables for this step
    #
    #         for next_tag in range(self.tagset_size):
    #             # next_tag_var[i] holds the viterbi variable for tag i at the
    #             # previous step, plus the score of transitioning
    #             # from tag i to next_tag.
    #             # We don't include the emission scores here because the max
    #             # does not depend on them (we add them in below)
    #             next_tag_var = forward_var + self.transitions[next_tag]
    #             best_tag_id = argmax(next_tag_var)
    #             bptrs_t.append(best_tag_id)
    #             viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
    #         # Now add in the emission scores, and assign forward_var to the set
    #         # of viterbi variables we just computed
    #         forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
    #         backpointers.append(bptrs_t)
    #
    #     # Transition to STOP_TAG
    #     terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
    #     best_tag_id = argmax(terminal_var)
    #     path_score = terminal_var[0][best_tag_id]
    #
    #     # Follow the back pointers to decode the best path.
    #     best_path = [best_tag_id]
    #     for bptrs_t in reversed(backpointers):
    #         best_tag_id = bptrs_t[best_tag_id]
    #         best_path.append(best_tag_id)
    #     # Pop off the start tag (we dont want to return that to the caller)
    #     start = best_path.pop()
    #     assert start == self.tag_to_ix[START_TAG]  # Sanity check
    #     best_path.reverse()
    #     return path_score, best_path
    #
    # def neg_log_likelihood(self, sentence, tags):
    #     feats = self._get_lstm_features(sentence)
    #     forward_score = self._forward_alg(feats)
    #     gold_score = self._score_sentence(feats, tags)
    #     return forward_score - gold_score

    # def forward(self, sentence):  # dont confuse this with _forward_alg above.
    #     # Get the emission scores from the BiLSTM
    #     lstm_feats = self._get_lstm_features(sentence)
    #
    #     # Find the best path, given the features.
    #     score, tag_seq = self._viterbi_decode(lstm_feats)
    #     return score, tag_seq
