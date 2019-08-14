"""
Defines a module that generates ELMo embeddings for input sentences
"""

import joblib
import numpy as np
import torch
from allennlp.modules.elmo import Elmo as allen_elmo
from allennlp.modules.elmo import batch_to_ids
from torch import nn

import utils
from entity_recognition_datasets.src import utils as data_utils


class Elmo(nn.Module):
    def __init__(self, device):
        """ Load the ELMo model. The first time you run this, it will download a pretrained model. """
        super(Elmo, self).__init__()
        options = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weights = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        # initialize instance of Elmo model
        self.elmo = allen_elmo(options, weights, 2, dropout=0)
        self._dev = device
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


# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_elmo_vectors(sentences, labels, batch_size=512):
    """
    Save all elmo vectors to a list of files
    :param sentences: list of list of tokens
    :return:
    """
    # prepare lens
    token_lens = np.asarray([len(s) for s in sentences])

    # prepare embeddings
    elmo = Elmo(torch.device("cpu"))
    embeddings = elmo(sentences)
    embeddings = embeddings.detach().numpy()

    # prepare labels by padding appropriately with -1
    maxlen = max(token_lens)
    PAD_TOKEN = -1
    for i, label in enumerate(labels):
        label.extend([PAD_TOKEN] * (maxlen - len(label)))
    labels = np.asarray(labels)

    output = {
        "lens": token_lens,
        "embeddings": embeddings,
        "tags": labels,
    }

    joblib.dump(output, "elmo/vecs.joblib")
    print("saved vectors")


if __name__ == '__main__':
    data = list(data_utils.read_conll("WNUT17"))[:10]
    words = []
    tags = []
    for sentence in data:
        w, t = utils.get_words_and_tags(sentence)
        words.append(w)
        tags.append(t)
    get_elmo_vectors(words, tags)
