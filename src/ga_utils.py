import numpy as np
import random
import pygad
import time
from tqdm import tqdm

from data_utils import ACC

# Dummy fitness function - not used for computing fitness values, but required by GA instance constructor
def fitness_func_dummy(ga_instance, solution, solution_idx):
    global classifier, X_test, y_test
    fitness = ACC(classifier, X_test, y_test, solution)
    return fitness

# Actual fitness function used by cal_pop_fitness to compute fitness values
def fitness_func(solution):
    global classifier, X_test, y_test
    fitness = ACC(classifier, X_test, y_test, solution)
    return fitness
def crossover_func(parents, offspring_size, ga_instance):
    offspring = np.zeros(offspring_size, dtype=int)

    # Sort the contents of the parent subsets in ascending order to eliminate duplicates when doing the crossover
    parents = np.sort(parents, axis=1)

    for k in range(offspring_size[0]):
        # Determine the crossover index at random
        crossover_point = np.random.randint(low=0, high=parents.shape[1], size=1)[0]

        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]

        # Check the elements of the sorted parents at the crossover index. To avoid gene duplication, keep the lower
        # part (0:crossover_point) of the parent whose value is lower at the crossover index and the upper part
        # (crossover_point:) of the other parent
        if (parents[parent1_idx, crossover_point] <= parents[parent1_idx, crossover_point]):
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        else:
            offspring[k, 0:crossover_point] = parents[parent2_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent1_idx, crossover_point:]

    return offspring

def mutation_func(offspring, ga_instance):
    # Calculate the number of genes to be mutated based on the specified mutation percent
    num_of_genes_to_mutate = (int)((ga_instance.mutation_percent_genes/100) * offspring.shape[1])
    gene_space = set(ga_instance.gene_space)

    for k in range(offspring.shape[0]):
        # Create a set out of the genes included in the non-mutated offspring
        original_offspring = set(offspring[k])

        # Find the genes which are not included into the non-mutated offspring
        available_genes = list(gene_space - original_offspring)
        # Shuffle the list of available genes so that it is not sorted
        random.shuffle(available_genes)
        # Generate an array of random positions in the offspring which will be mutated
        random_idx = np.random.randint(low=0, high=offspring.shape[1], size=num_of_genes_to_mutate)

        for i in random_idx:
            offspring[k, i] = available_genes.pop()

    # Take a time reading at the end of the evolution process for every generation / start of new generation
    gen = ga_instance.generations_completed + 1
    time_elapsed = time.time() - ga_instance.start_time
    ga_instance.evo_time_elapsed[gen] = time_elapsed + ga_instance.evo_time_elapsed[gen-1]
    print("Gen = {generation}, Best top-1 acc/fitness = {fitness}, Time elapsed from start = {time:3.2} s".format(generation=gen,
    fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1], time=ga_instance.evo_time_elapsed[gen]))

    return offspring

def on_start(ga_instance):
    print('Genetic Algorithm start', flush=True)

# Initializer function used to pass classifier, X_test and y_test to child processes
def child_process_init(classifier_instance, test_data, test_labels):
    global classifier
    global X_test
    global y_test
    classifier = classifier_instance
    X_test = test_data
    y_test = test_labels

# Modification of the GA class to use multiprocessing. The built-in multiprocessing could not be used because there was
# no way to pass the classifier, X_test and y_test variables required by the fitness_func to the child processes.
class MultiprocessGA(pygad.GA):
    # Extend the original GA class
    def __init__(self, *args, exectutor, start_time, evo_time_elapsed, **kwargs):
        super().__init__(*args, **kwargs)

        self.executor = exectutor
        self.start_time = start_time
        self.evo_time_elapsed = evo_time_elapsed

    def cal_pop_fitness(self):
        pop_fitness = ['undefined'] * len(self.population)
        solution_idxs = list(range(0, len(self.population)))

        for idx, fitness in tqdm(zip(solution_idxs, self.executor.map(fitness_func, self.population))):
            # Start measuring the evolution time for this generation at the beginning of the fitness value computation.
            # The measurement could start before this for loop, but the initialization of the child processes
            # produces an incorrect measurement for the time between gen 0 to gen 1.
            if (idx == 0):
                self.start_time = time.time()

            pop_fitness[idx] = fitness

        pop_fitness = np.array(pop_fitness)

        return pop_fitness