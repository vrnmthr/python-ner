import argparse
import time

import numpy as np
import plotly
import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from entity_recognition_datasets.src import utils
from model import BiLSTM

EPOCHS = 5
LOSS_FUNC = nn.CrossEntropyLoss()


def get_label(x):
    # 1 = anonymize
    # 0 = don't
    if x == "O":
        return 0
    else:
        return 1


def get_words_and_tags(sentence):
    # extracts tags and words from CONLL formatted entry
    tags = list(map(lambda word: get_label(word[1]), sentence))
    words = list(map(lambda word: word[0][0], sentence))
    return words, tags


def train():
    """
    :return: train_loss, dev_loss
    """

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    print("training...")

    train_losses = []
    dev_losses = []
    for epoch in range(EPOCHS):
        print("EPOCH {}/{}".format(epoch + 1, EPOCHS))
        start = time.time()
        # run a training epoch
        train_loss = 0
        for sentence in tqdm(train_data, desc="train-set"):
            # prepare tags and words
            words, tags = get_words_and_tags(sentence)
            tags = torch.LongTensor(tags).to(device)
            # calculate predictions
            model.zero_grad()
            out = model(words)
            # backprop
            loss = LOSS_FUNC(out, tags)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_data)
        train_losses.append(train_loss)

        # run a dev epoch
        dev_loss = 0
        with torch.no_grad():
            for sentence in tqdm(dev_data, desc="dev-set"):
                words, tags = get_words_and_tags(sentence)
                tags = torch.LongTensor(tags).to(device)
                # calculate predictions
                model.zero_grad()
                out = model(words)
                loss = LOSS_FUNC(out, tags)
                dev_loss += loss.item()
        dev_loss /= len(dev_data)
        dev_losses.append(dev_loss)

        print("train loss = {}".format(train_loss))
        print("dev loss = {}".format(dev_loss))
        duration = time.time() - start
        print("epoch completed in {:.3f}s, {:.3f}s per iteration".format(
            duration,
            duration / (len(train_data) + len(dev_data))
        ))

    losses = {
        "train": train_losses,
        "dev": dev_losses
    }

    return losses


def graph_losses(losses):
    plots = []
    for name in losses:
        plot = plotly.graph_objs.Scatter(x=np.arange(EPOCHS), y=losses[name], name=name)
        plots.append(plot)
    plotly.offline.plot(plots, filename="train_loss.html")


def evaluate():
    # evaluate on test data
    with torch.no_grad():
        confusion = np.zeros((2, 2))
        for sentence in tqdm(test_data, desc="train-set"):
            words, tags = get_words_and_tags(sentence)
            pred = model.evaluate(words)
            assert len(pred) == len(tags)
            for i in range(len(pred)):
                confusion[pred[i]][tags[i]] += 1

        confusion /= np.sum(confusion)
        return confusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="pass --device cuda to run on gpu (default 'cpu')", default="cpu")
    args = parser.parse_args()

    device = torch.device('cpu')
    if 'cuda' in args.device:
        if torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            print("cuda not available...")
    print("Using device {}".format(device))

    print("loading datasets...")
    train_data = list(utils.read_conll('WNUT17-train'))
    print("loaded train data")
    dev_data = list(utils.read_conll("WNUT17-dev"))
    print("loaded dev data")
    test_data = list(utils.read_conll("WNUT17-test"))
    print("loaded test data")

    model = BiLSTM(32, device)
    print("allocated model")

    losses = train()
    print("graphing")
    graph_losses(losses)

    confusion = evaluate()
    print(confusion)
    print("accuracy: {}".format(np.sum(np.diagonal(confusion))))
