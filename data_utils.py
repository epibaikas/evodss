import numpy as np
import torch

def ACC(classifier, X_test, y_test, subset_idxs):
    y_pred = classifier.predict(X_test, subset_idxs, train_classifier=False, k=3)
    y_pred_np = y_pred.cpu().numpy()
    num_correct = np.sum(y_test.cpu().numpy() == y_pred_np)
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