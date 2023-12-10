# Python packages
import torch

# User-defined modules
from classifiers.k_nearest_neighbors import kNearestNeighbors

device = 'cuda' if torch.cuda.is_available() else "cpu"
                                    # Self dot-products
X_train = torch.tensor([[0, 0, 0],  #0
                        [1, 1, 1],  #3
                        [2, 2, 2],  #12
                        [3, 3, 3],  #27
                        [4, 4, 4]]) #48

X_test = torch.tensor([[1, 1, 1],   #3
                      [2, 2, 2]])   #12

y_train = torch.tensor([0, 0, 1, 1, 1])
y_test = torch.tensor([0, 1])

D = X_train.shape[0]

expected_dists = torch.tensor([[3, 0, 3, 12, 27],
                               [12, 3, 0, 3, 12]])

expected_sorting_idxs = torch.tensor([[1, 0, 2, 3, 4],
                                  [2, 1, 3, 0, 4]])

classifier = kNearestNeighbors(X=X_train, y=y_train, device=device)
classifier.train(X_test)

def test_train():
    assert expected_dists.shape[0] == classifier.dists.shape[0]
    assert expected_dists.shape[1] == classifier.dists.shape[1]
    assert torch.equal(expected_dists, classifier.dists)
    assert torch.equal(expected_sorting_idxs, classifier.sorting_idxs)

def test_sorting_idxs_in_subsets():
    assert [1, 0, 3] == classifier.sorting_idxs_in_subset([1, 0, 2, 3, 4], subset_idxs=[1, 3, 4, 0], k=3)
    assert [1, 3, 0] == classifier.sorting_idxs_in_subset([2, 1, 3, 0, 4], subset_idxs=[1, 3, 4, 0], k=3)

    assert [1, 2, 3] == classifier.sorting_idxs_in_subset([1, 0, 2, 3, 4], subset_idxs=[1, 3, 2, 4], k=3)
    assert [2, 1, 3] == classifier.sorting_idxs_in_subset([2, 1, 3, 0, 4], subset_idxs=[1, 3, 2, 4], k=3)

    assert [2, 1, 3, 4] == classifier.sorting_idxs_in_subset([2, 1, 3, 0, 4], subset_idxs=[1, 3, 2, 4], k=5)


def test_predict():
    # Test with empty subset_idxs list (full train set)
    y_pred = classifier.predict(X_test, subset_idxs=[], train_classifier=False, k=3)
    assert torch.equal(torch.tensor([0, 1]), y_pred)

    # Test with different subsets indices
    y_pred = classifier.predict(X_test, subset_idxs=[1, 3, 4, 0], train_classifier=False, k=3)
    assert torch.equal(torch.tensor([0, 1]), y_pred)

    y_pred = classifier.predict(X_test, subset_idxs=[1, 3, 2, 4], train_classifier=False, k=3)
    assert torch.equal(torch.tensor([1, 1]), y_pred)



