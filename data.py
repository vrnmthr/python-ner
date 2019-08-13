from entity_recognition_datasets.src import stratified_split
from entity_recognition_datasets.src import utils


class DataSource:

    def __init__(self, split):
        """
        :param split: should either be "dev", "test", or "train"
        """
        self.data = []

        # add WNUT17
        wnut17_path = "WNUT17-{}".format(split)
        self.data.extend(utils.read_conll(wnut17_path))

        # no dev set for BTC
        if split != "dev":
            btc_path = "BTC-{}".format(split)
            self.data.extend(utils.read_conll(btc_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

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


def create_btc_split():
    """
    """
    stratified_split.write_new_split("BTC", 1000,
                                     filedir="entity_recognition_datasets/data/BTC/CONLL-format/data_generated",
                                     filename="btc")
