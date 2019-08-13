import torch
from torch import nn

from elmo import Elmo


class BiLSTM(nn.Module):
    def __init__(self, hidden_dim, device):
        super(BiLSTM, self).__init__()
        self.embedding_dim = 1024
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        # self.tag_to_ix = tag_to_ix
        # self.tagset_size = len(tag_to_ix)

        self.elmo = Elmo(device)
        self.lstm = nn.GRU(self.embedding_dim, hidden_dim // 2, num_layers=self.num_layers, bidirectional=True,
                           batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 2),
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
        self.hidden = self.init_hidden()
        self.to(self._dev)

    def init_hidden(self):
        return torch.randn(self.num_layers * 2, 1, self.hidden_dim // 2).to(self._dev)

    def forward(self, sentence):
        """
        :param sentence: single list of words
        :return: a list of words
        """
        self.hidden = self.init_hidden()
        # embeds = [1, sentence_len, 1024]
        embeds = self.elmo([sentence])
        # lstm_out = (1, seq_len, hidden_size * 2)
        lstm_out, hidden_out = self.lstm(embeds, self.hidden)
        predictions = self.linear(lstm_out)
        # squeeze the result to get rid of the batch for (seq_len, 2)
        predictions = predictions.squeeze(0)
        return predictions

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
