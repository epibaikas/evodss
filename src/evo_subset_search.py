# Python packages
import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import argparse
import concurrent.futures
import multiprocessing as mp
import pickle

# User-defined modules
from classifiers.k_nearest_neighbors import kNearestNeighbors
from data_utils import ACC
from argparse_utils import positive_int, float_zero_to_one
from ga_utils import (MultiprocessGA, fitness_func_dummy, crossover_func, mutation_func, on_start, child_process_init)

if __name__ == '__main__':
    # Set the process creation method to 'spawn' instead of 'fork' ('fork' is the default method in linux systems)
    mp.set_start_method('spawn')

    # Create an argument parser to set the dataset, subset proportion and trial number
    parser = argparse.ArgumentParser(
        'Script for improving the accuracy of kNN classifier on a subset of FashionMNIST/MNIST training sets via genetic algorithms')

    parser.add_argument('dataset', type=str,
                        help='The name of the dataset to be used')

    parser.add_argument('subset_prop', type=float_zero_to_one,
                        help='The size of the random subsets relative to the full set')

    parser.add_argument('trial', type=positive_int,
                        help='The trial number of the GA instance for the given subset proportion and dataset')

    args = vars(parser.parse_args())

    # Get the arguments
    subset_prop = args['subset_prop']
    dataset_name = args['dataset']
    trial = args['trial']

    # Adjust the random seed based on the trial number to obtain different results at every trial of the
    # GA execution
    random_seed = 41 + trial
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Set the device
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

    # Initialise classifier
    classifier = kNearestNeighbors(X_train, y_train, device)

    # Load distance matrix and sorting indices if they have already been computed in the past
    artifacts_dir_path = '../artifacts/'
    path_1 = artifacts_dir_path + dataset_name + '_dists.pt'
    path_2 = artifacts_dir_path + dataset_name + '_sorting_idxs.npy'
    if os.path.isfile(path_1) and os.path.isfile(path_2):
        print('Distance matrix and sorting indices loaded from file.')
        classifier.dists = torch.load(path_1)
        classifier.sorting_idxs = np.load(path_2)
    else:
        if not os.path.exists(artifacts_dir_path):
            os.makedirs(artifacts_dir_path)
        print('Compute distance matrix and sorting indices.')
        classifier.train(X_test)
        torch.save(classifier.dists, path_1)
        np.save(path_2, classifier.sorting_idxs)

    # Compute absolute subset size (number of examples) using the given subset proportion
    subset_size = (int)(subset_prop * len(train_set))

    # Genetic algorithm parameters
    num_generations = 400
    num_parents_mating = 20

    sol_per_pop = 100
    num_genes = subset_size

    init_range_low = 0
    init_range_high = len(train_set) - 1
    gene_space = range(len(train_set))

    parent_selection_type = 'sss'
    mutation_percent_genes = 10

    keep_elitism = 1

    # Create a ProcessPoolExecutor instance responsible for multiprocessing
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=4, initializer=child_process_init, initargs=(classifier, X_test, y_test))
    start_time = 0
    evo_time_elapsed = np.zeros(num_generations+1)

    ga_instance = MultiprocessGA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func_dummy,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_type=int,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           gene_space=gene_space,
                           parent_selection_type=parent_selection_type,
                           keep_elitism=keep_elitism,
                           crossover_type=crossover_func,
                           mutation_type=mutation_func,
                           mutation_percent_genes=mutation_percent_genes,
                           allow_duplicate_genes=False,
                           random_seed=random_seed,
                           save_solutions=False,
                           save_best_solutions=True,
                           on_start=on_start,
                           exectutor=executor,
                           start_time=start_time,
                           evo_time_elapsed=evo_time_elapsed)

    ga_instance.summary()

    # Start genetic algorithm
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("\nParameters of the best solution : \n\n {solution} \n".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    del executor

    print('Saving GA execution results to file...')
    results_dir_path = '../results/'
    identifier = dataset_name + '_subset_prop=' + f'{subset_prop:.4f}' + '_trial=' + str(trial)

    best_solutions = ga_instance.best_solutions
    best_solutions_fitness = ga_instance.best_solutions_fitness

    with open(results_dir_path + dataset_name + '/' + 'GA_results_' + identifier + '.pkl', 'wb') as handle:
        pickle.dump((best_solutions, best_solutions_fitness, ga_instance.evo_time_elapsed), handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Results saved. All done!')