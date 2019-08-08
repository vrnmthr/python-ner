"""
Defines a module that generates ELMo embeddings for input sentences
"""

import numpy as np
import torch
from allennlp.modules.elmo import Elmo as allen_elmo
from allennlp.modules.elmo import batch_to_ids
from torch import nn

from utils import read_conll


class Elmo(nn.Module):
    def __init__(self, device='cpu'):
        """ Load the ELMo model. The first time you run this, it will download a pretrained model. """
        super(Elmo, self).__init__()
        options = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weights = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        # initialize instance of Elmo model
        self.elmo = allen_elmo(options, weights, 2, dropout=0)
        self._dev = torch.device(device)
        self.to(self._dev)

    def forward(self, batch):
        """
        :param batch: List of tokenized sentences of varying lengths
        :return: Embeddings of dimension (batch, seq_len, embedding_dim), where seq_len is the max-length
        """
        # embeddings['elmo_representations'] is length two list of tensors.
        # Each element contains one layer of ELMo representations with shape
        # (batch, seq_len, embedding_dim).
        char_ids = batch_to_ids(batch).to(self._dev)
        embeddings = self.elmo(char_ids)
        return embeddings['elmo_representations'][-1]


def get_elmo_vectors(sentences):
    """
    Save all elmo vectors in a corpus to a file
    :param sentences: list of list of tokens
    :return:
    """
    token_lens = [len(s) for s in sentences]
    np.save("lens_test.npy", token_lens)
    print("saved lengths")
    elmo = Elmo()
    embeddings = elmo(sentences)
    embeddings = embeddings.detach().numpy()
    np.save("elmo_test.npy", embeddings)
    print("finished")


if __name__ == '__main__':
    TRAIN = list(read_conll("WNUT17-train"))
    DEV = list(read_conll("WNUT17-dev"))
    TEST = list(read_conll("WNUT17-test"))

    # TRAIN = TRAIN[:10]
    tokens = [[w[0][0] for w in t] for t in TEST]
    get_elmo_vectors(tokens)
