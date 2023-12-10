# Python packages
import torch
import numpy as np
import concurrent.futures

# User defined modules
from data_utils import load_dataset, get_class_example_indices
from ga_utils import (MultiprocessGA, fitness_func_dummy, crossover_func, mutation_func, child_process_init)

dataset_name = 'FashionMNIST'

# Set the device
device = 'cuda' if torch.cuda.is_available() else "cpu"

# Load the dataset
train_set, test_set, X_train, y_train, X_test, y_test = load_dataset(dataset_name, device)

# Set some dummy genetic algorithm parameters required by MultiProcess GA instance
# Number of genes in every solution vector
subset_size = 100
gene_space = range(len(train_set))

num_generations = 400
num_parents_mating = 20
sol_per_pop = 500
num_genes = subset_size
parent_selection_type = 'sss'
mutation_percent_genes = 10
keep_elitism = 1

# Create a dummy ProcessPoolExecutor instance
executor = concurrent.futures.ProcessPoolExecutor(max_workers=4, initializer=child_process_init)
start_time = 0
evo_time_elapsed = np.zeros(num_generations+1)

sets_of_idxs_per_class = []
for i in range(len(train_set.classes)):
    class_idxs = get_class_example_indices(train_set, i)
    sets_of_idxs_per_class.append(set(class_idxs))

# Create a dummy MultiProcess GA instance to run the tests
ga_instance = MultiprocessGA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func_dummy,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_type=int,
                       gene_space=gene_space,
                       parent_selection_type=parent_selection_type,
                       keep_elitism=keep_elitism,
                       crossover_type=crossover_func,
                       mutation_type=mutation_func,
                       mutation_percent_genes=mutation_percent_genes,
                       allow_duplicate_genes=False,
                       save_solutions=False,
                       exectutor=executor,
                       start_time=start_time,
                       evo_time_elapsed=evo_time_elapsed,
                       sets_of_idxs_per_class=sets_of_idxs_per_class)


parents = np.zeros([num_parents_mating, subset_size])
for i in range(num_parents_mating):
    parents[i] = np.random.choice(gene_space, subset_size, replace=False)

def test_crossover_func():
    offsprings = crossover_func(parents, offspring_size=(sol_per_pop-keep_elitism, subset_size), ga_instance=ga_instance)

    # Check that no offspring contains duplicates
    for offspring in offsprings:
        assert len(offspring) == len(np.unique(offspring))

def test_mutation_func():
    offsprings = np.zeros([sol_per_pop-keep_elitism, subset_size])
    for i in range(sol_per_pop-keep_elitism):
        offsprings[i] = np.random.choice(gene_space, subset_size, replace=False)

    mut_offsprings = mutation_func(offsprings, ga_instance)

    # Check that no mutated offspring contains duplicates
    for mut_offspring in mut_offsprings:
        assert len(mut_offspring) == len(np.unique(mut_offspring))
