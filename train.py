import torch
from entity_recognition_datasets.src import utils
from torch import optim
from torch import nn
from model import BiLSTM


EPOCHS = 10
LOSS_FUNC = nn.CrossEntropyLoss()

data = list(utils.read_conll('WNUT17-train'))[:2]

model = BiLSTM(32)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)


def get_label(x):
    # 1 = anonymize
    # 0 = don't
    if x == "O":
        return 0
    else:
        return 1


# Make sure prepare_sequence from earlier in the LSTM section is loaded
print("training...")
for epoch in range(EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence in data:
        # prepare tags and words
        tags = torch.Tensor(list(map(lambda word: get_label(word[1]), sentence))).long()
        words = list(map(lambda word: word[0][0], sentence))

        # calculate predictions
        model.zero_grad()
        out = model(words)

        # backprop
        loss = LOSS_FUNC(out, tags)
        loss.backward()
        optimizer.step()

