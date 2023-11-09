# Python packages
import numpy as np
import torch
import os
import argparse
import concurrent.futures
import multiprocessing as mp

# User-defined modules
from classifiers.k_nearest_neighbors import kNearestNeighbors
from data_utils import load_dataset, ACC
from argparse_utils import positive_int, positive_float
from ga_utils import (MultiprocessGA, fitness_func_dummy, crossover_func, mutation_func, mutation_func_modified, on_start, on_generation, child_process_init)

if __name__ == '__main__':
    # Set the process creation method to 'spawn' instead of 'fork' ('fork' is the default method in linux systems)
    mp.set_start_method('spawn')

    # Create an argument parser to set the dataset, non-volatile memory size (in MBs) and trial number
    parser = argparse.ArgumentParser(
        'Script for evaluating the accuracy of a kNN classifier in a class-incremental learning scenario with non-volatile memory restrictions. ')

    parser.add_argument('dataset', type=str,
                        help='The name of the dataset to be used')

    parser.add_argument('memory_size', type=positive_float,
                        help='The size of the non-volatile memory in MB')

    parser.add_argument('seq', type=str,
                        help='Enter \'high\' or  \'low\' for high\low accuracy sequence of classes respectively')

    parser.add_argument('trial', type=positive_int,
                        help='The experiment trial number')

    args = vars(parser.parse_args())

    # Get the arguments
    dataset_name = args['dataset']
    memory_size = args['memory_size']
    seq_type = args['seq']
    trial = args['trial']

    # Adjust the random seed based on the trial number to obtain different results at every trial
    random_seed = 41 + trial
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Check that the class sequence is valid
    if seq_type != 'high' and seq_type != 'low':
        raise argparse.ArgumentTypeError('Invalid sequence type')

    # Set the device
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the dataset
    train_set, test_set, X_train, y_train, X_test, y_test = load_dataset(dataset_name, device)

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

    # Compute the number of examples that can be stored in non-volatile memory
    example_size = X_train.shape[1] + 1     # Number of bytes required per example (add 1 byte for label)
    K = (int)((memory_size * pow(2, 20)) / example_size)
    print('Data example size:', example_size, 'bytes')
    print('K:', K)

    class_sequences = np.load(artifacts_dir_path + dataset_name + '_class_sequences.npy')

    if seq_type == 'high':
        Q = class_sequences[0]
    elif seq_type == 'low':
        Q = class_sequences[1]

    num_of_classes = len(Q)
    print('Using ' + seq_type + ' accuracy sequence.')
    print('Q:', Q, '\n')

    # Create numPy array to store accuracy values.
    # The first row of the array is for random subset accuracies and the second one for the best accuracy obtained
    # through EvoDSS.
    acc_array = np.zeros((2, num_of_classes-1))

    # Random balanced subset selection ---------------------------------------------------------------------------------
    print('Random balanced subset selection')
    first_class_idxs = torch.nonzero((train_set.targets == Q[0]))
    first_class_idxs = torch.squeeze(first_class_idxs).numpy()

    # M is a list of numpy 1D arrays which hold the examples indices for every class of the sequence
    M = []
    print(M)
    if K > first_class_idxs.shape[0]:
        M.append(first_class_idxs)
    else:
        M.append(np.random.choice(first_class_idxs, size=K, replace=False))

    # Get the test examples with labels Q[0]
    test_example_idxs = torch.nonzero(test_set.targets == Q[0])
    test_example_idxs = torch.squeeze(test_example_idxs).numpy().tolist()

    for t in range(1, num_of_classes):
        K_t = (int) (K / (t + 1))   # Number of examples per class
        remainder = K % (t+1)
        print('K_t', K_t, 'remainder', remainder)

        for i in range(t):
            # Get the indices of the examples in M with label Q[i]
            class_idxs = M[i]

            if K_t <= class_idxs.shape[0]:
                # Keep only K_t examples with label Q[i]
                M[i] = np.random.choice(M[i], size=K_t, replace=False)

            # print('Class', Q[i], 'size', M[i].shape[0])

        new_class_idxs = torch.nonzero((train_set.targets == Q[t]))
        new_class_idxs = torch.squeeze(new_class_idxs).numpy()

        if K_t + remainder > new_class_idxs.shape[0]:
            M.append(new_class_idxs)
        else:
            M.append(np.random.choice(new_class_idxs, size=K_t + remainder, replace=False))
        # print('Class', Q[t], 'size', M[t].shape[0])

        # Get the test examples from the new class
        new_test_example_idxs = torch.nonzero(test_set.targets == Q[t])
        new_test_example_idxs = torch.squeeze(new_test_example_idxs).numpy().tolist()

        # Add the new test examples to the existing list
        test_example_idxs += new_test_example_idxs

        # Change the list of numpy arrays into a list of lists
        list_of_class_idx_lists = [class_idxs_numpy_array.tolist() for class_idxs_numpy_array in M]
        # Flatten the list of lists to a single list
        train_subset_idxs = [idx for sub_list in list_of_class_idx_lists for idx in sub_list]

        acc = ACC(classifier, X_test[test_example_idxs], y_test[test_example_idxs] , subset_idxs=train_subset_idxs, test_subset_idxs=test_example_idxs)
        acc_array[0, t-1] = acc
        print('t =', t, ', top-1 acc = {acc:0.5f}'.format(acc=acc))


    # Evolutionary Data Subset Selection -------------------------------------------------------------------------------
    print('\nEvoDSS')
    if K > first_class_idxs.shape[0]:
        M = first_class_idxs.tolist()
    else:
        M = np.random.choice(first_class_idxs, size=K, replace=False).tolist()

    # Get the test examples with labels Q[0]
    test_example_idxs = torch.nonzero(test_set.targets == Q[0])
    test_example_idxs = torch.squeeze(test_example_idxs).numpy().tolist()

    for t in range(1, num_of_classes):
        print('t = {t}, classes present {classes}'.format(t=t, classes=Q[0:t+1]))

        # Get the test examples from the new class
        new_test_example_idxs = torch.nonzero(test_set.targets == Q[t])
        new_test_example_idxs = torch.squeeze(new_test_example_idxs).numpy().tolist()

        # Add the new test examples to the existing list
        test_example_idxs += new_test_example_idxs

        # Get the training examples from the new class
        new_class_idxs = torch.nonzero((train_set.targets == Q[t]))
        new_class_idxs = torch.squeeze(new_class_idxs).numpy()

        V_t = M + new_class_idxs.tolist()
        print('Number of examples in V_t:', len(V_t))

        if (K > len(V_t)):
            # All examples can fit in memory so no need to use subset selection
            acc_array[1, t-1] = ACC(classifier, X_test[test_example_idxs], y_test[test_example_idxs] , subset_idxs=V_t, test_subset_idxs=test_example_idxs)
            print('t = {t}, no need for subset selection, top-1 acc = {acc:0.5f} \n'.format(t=t, acc= acc_array[1, t-1]))
            # Move all elements of V_t to M
            M = V_t
        else:
            sets_of_idxs_per_class = []
            for i in range(t+1):
                class_Q_i_idxs = torch.nonzero((y_train[V_t] == Q[i]))
                class_Q_i_idxs = torch.squeeze(class_Q_i_idxs).tolist()
                class_Q_i_idxs = np.array(V_t)[class_Q_i_idxs].tolist()
                sets_of_idxs_per_class.append(set(class_Q_i_idxs))

            # Genetic algorithm parameters
            num_generations = 10
            num_parents_mating = 20

            sol_per_pop = 100
            num_genes = K

            gene_space = V_t

            parent_selection_type = 'sss'
            mutation_percent_genes = 10

            keep_elitism = 1

            # Create a ProcessPoolExecutor instance responsible for multiprocessing
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=4, initializer=child_process_init,
                                                              initargs=(classifier, X_test[test_example_idxs], y_test[test_example_idxs], test_example_idxs))
            start_time = 0
            evo_time_elapsed = np.zeros(num_generations + 1)

            # Initialize population
            init_pop = np.zeros((sol_per_pop, num_genes))
            for i in range(sol_per_pop):
                init_pop[i] = np.random.choice(gene_space, size=K, replace=False)

            ga_instance = MultiprocessGA(num_generations=num_generations,
                                num_parents_mating=num_parents_mating,
                                fitness_func=fitness_func_dummy,
                                sol_per_pop=sol_per_pop,
                                num_genes=num_genes,
                                initial_population=init_pop,
                                gene_type=int,
                                gene_space=gene_space,
                                parent_selection_type=parent_selection_type,
                                keep_elitism=keep_elitism,
                                crossover_type=crossover_func,
                                mutation_type=mutation_func_modified,
                                mutation_percent_genes=mutation_percent_genes,
                                allow_duplicate_genes=False,
                                random_seed=random_seed,
                                save_solutions=False,
                                save_best_solutions=False,
                                on_start=on_start,
                                on_generation=on_generation,
                                exectutor=executor,
                                start_time=start_time,
                                evo_time_elapsed=evo_time_elapsed,
                                sets_of_idxs_per_class=sets_of_idxs_per_class)

            # Start genetic algorithm
            ga_instance.run()

            best_solution, solution_fitness, _ = ga_instance.best_solution()
            acc_array[1, t-1] = solution_fitness
            print('t = {t}, Fitness value of the best solution = {solution_fitness:0.5f} \n'.format(t=t, solution_fitness=solution_fitness))

            # Move the best solution to M
            M = best_solution.tolist()

            executor.shutdown(wait=True)
            del ga_instance, executor

    print('Saving results to file...')
    results_dir_path = '../results/'
    if not os.path.exists(results_dir_path + dataset_name + '/'):
        os.makedirs(results_dir_path + dataset_name + '/')

    identifier = dataset_name + '_memory_size=' + f'{memory_size:.2f}' + '_' + seq_type + '_acc_seq' + '_trial=' + str(trial)
    np.save(results_dir_path + dataset_name + '/' + 'Incremental_learning_acc_array_' + identifier + '.npy', acc_array)

    print('Results saved. All done!')