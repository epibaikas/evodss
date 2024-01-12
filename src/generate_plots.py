import os
import matplotlib.pyplot as plt

# User-defined modules
from plot_utils import (best_fitness_vs_gen_plot, best_fitness_vs_time_plot, mean_time_per_gen,
                        acc_distribution_histograms_plot, violin_acc_plot, incremental_learning_plot)

# Set matplotlib font parameters
rc_params = {
    'figure.dpi': 300,
    'font.family': 'Linux Libertine O',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.title_fontsize': 8,
    'text.usetex': True,
}
plt.rcParams.update(rc_params)

dataset_names = ['MNIST', 'FashionMNIST']
subset_prop_list = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256]
memory_size_list = [0.125, 2, 16]
num_of_trials = 20
results_dir_path = '../results/'
plots_dir_path = '../plots/'

# Create directory for plots if it doesn't exist
if not os.path.exists(plots_dir_path):
    os.makedirs(plots_dir_path)

print('Generating plots...')
best_fitness_vs_gen_plot(dataset_names, subset_prop_list, num_of_trials, results_dir_path, plots_dir_path)
best_fitness_vs_time_plot(dataset_names, subset_prop_list, num_of_trials, results_dir_path, plots_dir_path)
mean_time_per_gen(dataset_names, subset_prop_list, num_of_trials, results_dir_path, plots_dir_path)
violin_acc_plot(dataset_names, subset_prop_list, num_of_trials, results_dir_path, plots_dir_path)
acc_distribution_histograms_plot(dataset_names, subset_prop_list, results_dir_path, plots_dir_path)
incremental_learning_plot(dataset_names, memory_size_list, num_of_trials, results_dir_path, plots_dir_path)
print('Plots generated!')