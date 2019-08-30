import argparse

import torch

from model import BiLSTM

parser = argparse.ArgumentParser()
parser.add_argument("restore", help="path to restore model from")
args = parser.parse_args()

device = torch.device("cpu")
model = BiLSTM(128, device)
model.load_state_dict(torch.load(args.restore))
print("loaded model from {}".format(args.restore))

while True:
    sentence = input("> ")
    sentence = sentence.split(" ")
    results = model.evaluate(sentence)
    out = list(zip(sentence, results))
    print(out)
