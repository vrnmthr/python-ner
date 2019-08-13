import argparse
import time

import numpy as np
import plotly
import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from data import DataSource
from model import BiLSTM

EPOCHS = 4
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
    words = list(map(lambda word: word[0][0].strip().lower(), sentence))
    return words, tags


def train_single(sentence, optimizer, backprop):
    """
    Runs a single sentence through the model and returns the loss. If backprop is True, then backprops.
    :return:
    """
    try:
        # prepare tags and words
        words, tags = get_words_and_tags(sentence)
        tags = torch.LongTensor(tags).to(device)
        # calculate predictions
        model.zero_grad()
        out = model(words)
        # backprop
        loss = LOSS_FUNC(out, tags)
        if backprop:
            loss.backward()
            optimizer.step()
        loss = loss.item()
    except Exception as e:
        print(e)
        print("words: {}".format(words))
        print("tags: {}".format(tags))
        print("continuing...")
        loss = 0
    return loss


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

        # run train epoch, with randomized training order
        start = time.time()
        train_loss = 0
        ixs = np.arange(len(train_data))
        np.random.shuffle(ixs)
        for ix in tqdm(ixs, desc="train-set"):
            sentence = train_data[ix]
            loss = train_single(sentence, optimizer, backprop=True)
            train_loss += loss
        train_loss /= len(train_data)
        train_losses.append(train_loss)
        duration = time.time() - start
        print("train set completed in {:.3f}s, {:.3f}s per iteration".format(duration, duration / len(train_data)))

        # run a dev epoch
        start = time.time()
        dev_loss = 0
        with torch.no_grad():
            for sentence in tqdm(dev_data, desc="dev-set"):
                loss = train_single(sentence, optimizer, backprop=False)
                dev_loss += loss
        dev_loss /= len(dev_data)
        dev_losses.append(dev_loss)
        duration = time.time() - start
        print("dev set completed in {:.3f}s, {:.3f}s per iteration".format(duration, duration / len(dev_data)))

        print("train loss = {}".format(train_loss))
        print("dev loss = {}".format(dev_loss))

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
            try:
                words, tags = get_words_and_tags(sentence)
                pred = model.evaluate(words)
                assert len(pred) == len(tags)
                for i in range(len(pred)):
                    confusion[pred[i]][tags[i]] += 1
            except Exception as e:
                print(e)
                print("words: {}".format(words))
                print("tags: {}".format(tags))
                print("continuing...")

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
    train_data = DataSource("train")
    print("loaded {} train data".format(len(train_data)))
    dev_data = DataSource("dev")
    print("loaded {} dev data".format(len(dev_data)))
    test_data = DataSource("test")
    print("loaded {} test data".format(len(test_data)))

    model = BiLSTM(64, device)
    print("allocated model")

    losses = train()
    print("graphing")
    graph_losses(losses)

    confusion = evaluate()
    print(confusion)
    print("accuracy: {}".format(np.sum(np.diagonal(confusion))))
