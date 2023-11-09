import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import argparse
def load_dataset(dataset_name, device):
    # Get the dataset
    if dataset_name == 'FashionMNIST':
        train_set = datasets.FashionMNIST(root='../datasets/', download=True, transform=ToTensor(), train=True)
        test_set = datasets.FashionMNIST(root='../datasets/', download=True, transform=ToTensor(), train=False)
    elif dataset_name == 'MNIST':
        train_set = datasets.MNIST(root='../datasets/', download=True, transform=ToTensor(), train=True)
        test_set = datasets.MNIST(root='../datasets/', download=True, transform=ToTensor(), train=False)
    else:
        raise argparse.ArgumentTypeError('Unknown dataset name')

    X_train = train_set.data.type(torch.float32)
    y_train = train_set.targets.type(torch.int8).to(device)

    X_test = test_set.data.type(torch.float32)
    y_test = test_set.targets.type(torch.int8).to(device)

    print('Dataset:', dataset_name)
    print('X_train shape:', X_train.shape, X_train.dtype)
    print('y_train shape:', y_train.shape, y_train.dtype)
    print('X_test shape:', X_test.shape, X_test.dtype)
    print('y_test shape:', y_test.shape, y_test.dtype, '\n')

    X_train = torch.reshape(X_train, (X_train.shape[0], -1)).to(device)
    X_test = torch.reshape(X_test, (X_test.shape[0], -1)).to(device)

    return train_set, test_set, X_train, y_train, X_test, y_test

def ACC(classifier, X_test, y_test, subset_idxs, test_subset_idxs=None):
    y_pred = classifier.predict(X_test, subset_idxs, train_classifier=False, k=3)
    y_pred_np = y_pred.cpu().numpy()

    if test_subset_idxs is None:
        num_correct = np.sum(y_test.cpu().numpy() == y_pred_np)
    else:
        num_correct = np.sum(y_test.cpu().numpy() == y_pred_np[test_subset_idxs])
    acc = num_correct / y_test.shape[0]
    return acc

def get_random_balanced_subset_indices(dataset, subset_size):
    idxs = []
    dataset_size = len(dataset)

    for class_num, _ in enumerate(dataset.classes):
        # Get all the indices corresponding to the class number
        class_idxs = torch.nonzero((dataset.targets == class_num))
        # Remove unnecessary dimensions and convert to numpy array
        class_idxs = torch.squeeze(class_idxs).numpy()

        # Set the number of examples to be selected from the given class depending on the portion that it occupies in
        # complete dataset
        class_subset_size = (int)((class_idxs.size / dataset_size) * subset_size)

        # Get a random sample of the indices
        class_idxs = np.random.choice(class_idxs, class_subset_size, replace=False)

        idxs += list(class_idxs)

    return idxs

def get_class_example_indices(dataset, class_num):
    class_idxs = torch.nonzero((dataset.targets == class_num))
    class_idxs = torch.squeeze(class_idxs).numpy().tolist()
    return class_idxs