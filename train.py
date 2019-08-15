import argparse

from fastai.text import *

from data import NERDataset
from model import BiLSTM

EPOCHS = 4
batch_size = 4


def get_data_bunch():
    return DataBunch.create(
        train_ds=NERDataset(["WNUT17-train"]),
        valid_ds=NERDataset(["WNUT17-dev"]),
        test_ds=NERDataset(["WNUT17-test"]),
        bs=batch_size,
        device=device,
    )


def get_learner():
    lossfunc = CrossEntropyFlat(weight=torch.FloatTensor([0.1, 0.9]), ignore_index=-1)

    learner = Learner(
        data=get_data_bunch(),
        model=model,
        metrics=None,
        loss_func=lossfunc,
    )
    return learner


def Accuracy(ignore_index=None):
    def accuracy(input: Tensor, targs: Tensor) -> Rank0Tensor:
        "Computes accuracy with `targs` when `input` is bs * n_classes."
        n = targs.shape[0]
        input = input.argmax(dim=-1).view(n, -1)
        targs = targs.view(n, -1)

        if ignore_index is not None:
            # create a mask for everything in ignore_index and a valid mask for valid entries
            mask = (targs == ignore_index).bool()
            valid = ~mask
            # set all values in the mask to 0 in targs and 1 in input
            input[mask] = 1
            targs[mask] = 0
            # calculate the mean, multiply by the number of items, divide by the number of ignored indices
            return (input == targs).float().sum() / valid.sum()
        else:
            return (input == targs).float().mean()

    return accuracy


def train(learner):
    learner.metrics = [
        Accuracy(ignore_index=-1),
        # ConfusionMatrix(),
        # Precision(),
        # Recall(),
    ]
    fit_one_cycle(
        learn=learner,
        cyc_len=10,
        max_lr=1e-4,
        wd=0.1,
    )


def find_lr(learner):
    print("finding learning rate")
    lr_find(learner, num_it=10000, wd=0.1)
    plt = learner.recorder.plot(return_fig=True)
    plt.savefig("out/lr_find.jpg")
    print("saved plot to out/lr_find.jpg")


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

    model = BiLSTM(batch_size, 64, device)
    print("allocated model")

    learner = get_learner()
    train(learner)
