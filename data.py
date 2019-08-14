import joblib
from fastai.text import *

from entity_recognition_datasets.src import utils


class NERDataset(Dataset):

    def __init__(self, paths):
        """
        :param paths: a list of paths to load joblib objects out of elmo/
        """
        # self.raw_data = []
        self.lens = []
        self.embs = []
        self.tags = []

        for p in paths:
            data = joblib.load("elmo/{}.joblib".format(p))
            self.lens.extend(data['lens'])
            self.embs.extend(data['embeddings'])
            self.tags.extend(data['tags'])

        # # add WNUT17
        # wnut17_path = "WNUT17-{}".format(split)
        # self.raw_data.extend(utils.read_conll(wnut17_path))
        #
        # # add BTC
        # if split != "dev":
        #     btc_path = "BTC-{}".format(split)
        #     self.raw_data.extend(utils.read_conll(btc_path))
        #
        # # preprocess all the data
        # for sentence in self.raw_data:
        #     words, tags = self.get_words_and_tags(sentence)
        #     self.data.append((words, tags))

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, key):
        """
        Returns a single sentence
        :param key:
        :return:
        """
        return (self.lens[key], self.embs[key]), self.tags[key]

    def print_dist(self):
        """
        Prints information about the distribution of labels in the sets.
        :return:
        """
        total = {
            0: 0,
            1: 0
        }

        corpora = ["WNUT17", "BTC"]
        for c in corpora:
            counts = utils.get_NER_tagcounts(c)
            for tag in counts:
                if tag == "O":
                    total[0] += counts[tag]
                else:
                    total[1] += counts[tag]
        print(total)
        return total

# def create_btc_split():
#     """
#     """
#     stratified_split.write_new_split("BTC", 1000,
#                                      filedir="entity_recognition_datasets/data/BTC/CONLL-format/data_generated",
#                                      filename="btc")
