import numpy as np
import torch

class kNearestNeighbors(object):
    """ A k-Nearest-Neighbors classifier with L2 distance """

    def __init__(self, X, y, device):
        """
        Instantiate the classifier.

        :param X: A torch Tensor of shape [num_train, D] containing num_train training examples
        each of dimension D.
        :param y: A torch Tensor of shape [num_train] containing the training labels,
        where y[i] is the label for X[i].
        :param device: The device name used by torch for acceleration (cpu, cuda, mps etc.)
        """

        self.X_train = X
        self.y_train = y
        self.device = device

        # A 2D torch tensor with dimensions [num_test, num_train] that holds the computed distances between the test
        # and the train examples
        self.dists = torch.empty([])

        # A 2D numpy array that holds the indices which sort the rows of the distance matrix
        self.sorting_idxs = torch.empty([])

    def train(self, X):
        """
        Train the classifier. For a k-Nearest-Neighbors classifier, training involves just computing a distance matrix
        with dimensions [num_test, num_train], between the training and test examples.

        :param X: A torch Tensor of shape [num_test, D] containing num_test test examples
        each of dimension D.
        """

        self.dists = self.compute_distances(X)
        # Find the sorting indices for each row of the distance matrix
        self.sorting_idxs = torch.argsort(self.dists, dim=1).numpy()

    def predict(self, X, subset_idxs=[], train_classifier=True, k=1):
        """
        Predict labels for the test data using this classifier.

        :param X: A torch Tensor of shape [num_test, D] containing num_test test examples
        each of dimension D.
        :param subset_idxs: A list/numpy array containing the indices of the training examples used for classification. If the list
        is empty, all the examples from X_train are used.
        :param train_classifier: A boolean indicating whether the distance matrix should be recomputed.
        :param k: The number of k-nearest-neighbors that vote for the predicted labels.


        :return y: A torch Tensor of shape [num_test] containing predicted labels for the
        test data, where y[i] is the predicted label for the test point X[i].
        """

        # Recompute the distance matrix using the new test examples
        if train_classifier:
            self.train(X)

        # Check if the list of subset_idxs is empty
        if len(subset_idxs) == 0:
            # Keep only the indices of the closest k examples
            kNN_labels = self.y_train[self.sorting_idxs[:, 0:k]].to(torch.int8)

        else:
            # If the number of examples in the subset is less than 1000, isolate the columns from the distance matrix
            # corresponding to the training examples in the subset and sort. This approach is faster than searching
            # sorting_idxs for examples in subset_idxs.
            if len(subset_idxs) < 1000:
                sorting_subset_idxs = torch.argsort(self.dists[:, subset_idxs], dim=1).numpy()
                sorting_subset_idxs = sorting_subset_idxs[:, 0:k]
                y_train_subset = self.y_train[subset_idxs]
                kNN_labels = y_train_subset[sorting_subset_idxs].to(torch.int8)
            else:
                # For subset sizes > 1000, find the first k examples from sorting_idxs that belong to subset_idxs.
                # Turn the subset_idxs list to a set for faster search.
                subset_idxs = set(subset_idxs)
                sorting_subset_idxs = [self.sorting_idxs_in_subset(row, subset_idxs, k) for row in self.sorting_idxs]

                # Retrieve the labels of the closest k examples
                kNN_labels = self.y_train[sorting_subset_idxs].to(torch.int8)

        # Find the most common label (in case of a tie, select the smallest label)
        kNN_label_counts = [torch.bincount(row) for row in kNN_labels]
        y_pred = torch.stack([torch.argmax(row) for row in kNN_label_counts])

        return y_pred

    def compute_distances(self, X):
        """
        Compute the l2 distance matrix between each test point in X and each training point
        in self.X_train. The computation of the matrix has been implemented using only matrix operations
        without any loops.

        :param X: A torch Tensor of shape [num_test, D] containing num_test test examples
        each of dimension D.

        :return dists: A torch Tensor of shape [num_test, num_train] where dists[i, j] is the
        l2 distance between the ith test point and the jth training point.
        """

        # The distance of a single test vector from a training vector is given by (x_test - x_train)^2 (ignoring the
        # square root). But this is equal to x_test * x_test.T + x_train * x_train.T - 2 * x_test * x_train.T.
        # Because we are dealing with multiple test and train examples, we can create 2 matrices containing all the dot
        # products necessary for the calculation of the distances. The shape of these matrices is (num_test, num_train).
        # X_dot contains the dot products of test examples copied over num_train columns.
        # X_train_dot contains the dot products of training examples copied over num_test rows.

        # X_dot = (X * X).sum(axis=1).reshape((X.shape[0], 1)) * np.ones(shape=(1, self.X_train.shape[0]))
        # X_train_dot = (self.X_train * self.X_train).sum(axis=1) * np.ones(shape=(X.shape[0], 1))
        # dists = X_dot + X_train_dot - 2*X.dot(self.X_train.T)  # Ignore computing the square root for efficiency

        X_dot = torch.mul(X, X).sum(axis=1).reshape((X.shape[0], 1)) * torch.ones(size=(1, self.X_train.shape[0]), dtype=torch.float32, device=self.device)
        X_train_dot = torch.mul(self.X_train, self.X_train).sum(axis=1) * torch.ones(size=(X.shape[0], 1), dtype=torch.float32, device=self.device)
        dists = X_dot + X_train_dot - 2*torch.mm(X, self.X_train.T)  # Ignore computing the square root for efficiency

        return dists

    def sorting_idxs_in_subset(self, sorting_idxs, subset_idxs, k):
        """
        Find the first k indices from sorting_idxs that exist in subset_idxs.

        :param sorting_idxs: A torch Tensor of shape [num_train].
        :param subset_idxs: A set that contains the indices of training examples.
        :param k: The number of k-nearest-neighbors that vote for the predicted labels.


        :return sorting_subset_idxs: A list with the first k indices from sorting_idxs that exist in subset_idxs.
        """

        count = 0
        sorting_subset_idxs = []

        for idx in sorting_idxs:
            if idx in subset_idxs:
                sorting_subset_idxs.append(idx)
                count += 1
                if count == k:
                    break

        return sorting_subset_idxs