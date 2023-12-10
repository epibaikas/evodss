import numpy as np
import pickle

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

def best_fitness_vs_gen_plot(dataset_names, subset_prop_list, num_of_trials, results_dir_path, plots_dir_path):
    # Create figure and set size
    fig = plt.figure(figsize=(9, 4))

    # Set the width ratios for every subplot in the figure, leave 10% for the legend at the end
    width_ratios = [(1 - 0.1) / len(dataset_names) for i in range(len(dataset_names))]
    width_ratios.append(0.1)
    gs = GridSpec(nrows=1, ncols=len(dataset_names)+1, width_ratios=width_ratios)

    for plot_idx, dataset_name in enumerate(dataset_names):
        ax = fig.add_subplot(gs[plot_idx])

        with open(results_dir_path + dataset_name + '/' + 'Full_set_acc_' + dataset_name + '.pkl', 'rb') as handle:
            full_set_acc = pickle.load(handle)

        for subset_prop in subset_prop_list:
            # Load the first file, to get the number of generations
            identifier = dataset_name + '_subset_prop=' + f'{subset_prop:.4f}' + '_trial=' + str(1)
            with open(results_dir_path + dataset_name + '/' + 'GA_results_' + identifier + '.pkl', 'rb') as handle:
                best_solutions, best_solutions_fitness, evo_time_elapsed = pickle.load(handle)

            num_of_generations = len(best_solutions_fitness)

            best_fitness = np.zeros((num_of_trials, num_of_generations))
            best_fitness[0, :] = best_solutions_fitness

            for trial in range(2, num_of_trials+1):
                identifier = dataset_name + '_subset_prop=' + f'{subset_prop:.4f}' + '_trial=' + str(trial)
                with open(results_dir_path + dataset_name + '/' + 'GA_results_' + identifier + '.pkl', 'rb') as handle:
                    best_solutions, best_solutions_fitness, evo_time_elapsed = pickle.load(handle)

                best_fitness[trial-1, :] = best_solutions_fitness

            best_fitness_mean = np.mean(best_fitness, axis=0)
            best_fitness_std = np.std(best_fitness, axis=0)

            label = str(subset_prop * 100)
            ax.plot(range(num_of_generations), best_fitness_mean, label=label)
            ax.fill_between(range(num_of_generations), best_fitness_mean - best_fitness_std, best_fitness_mean + best_fitness_std, alpha=0.2)

        # Invert the order of the legend labels
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[::-1]
        labels = labels[::-1]

        # Add dashed line that shows the accuracy obtained on the full training set
        ax.axhline(y=full_set_acc, color='k', linestyle='--', label='Full set accuracy')
        ax.annotate('Full training set \(ACC\): ' + f"{full_set_acc:.2f}", (1, full_set_acc + 0.01), color='k')

        ax.set_xlabel('Generation')
        ax.set_title(dataset_name)
        ax.set_ylim(bottom=0.5, top=1.0)
        ax.grid()

    ax_legend = fig.add_subplot(gs[-1])
    ax_legend.axis('off')
    ax_legend.legend(title='Subset \n proportion (\%)', handles=handles, labels=labels, bbox_to_anchor=(-0.5, 1), loc='upper left')

    ax = fig.axes
    ax[0].set_ylabel('Best \(ACC\) / fitness')
    fig.suptitle('Generation vs Best fitness')
    plt.savefig(plots_dir_path + 'gen_vs_best_fitness.png', dpi=300)

def best_fitness_vs_time_plot(dataset_names, subset_prop_list, num_of_trials, results_dir_path, plots_dir_path):
    # Create figure and set size
    fig = plt.figure(figsize=(9, 4))

    # Set the width ratios for every subplot in the figure, leave 10% for the legend at the end
    width_ratios = [(1 - 0.1) / len(dataset_names) for i in range(len(dataset_names))]
    width_ratios.append(0.1)
    gs = GridSpec(nrows=1, ncols=len(dataset_names)+1, width_ratios=width_ratios)

    for plot_idx, dataset_name in enumerate(dataset_names):
        ax = fig.add_subplot(gs[plot_idx])

        with open(results_dir_path + dataset_name + '/' + 'Full_set_acc_' + dataset_name + '.pkl', 'rb') as handle:
            full_set_acc = pickle.load(handle)

        for subset_prop in subset_prop_list:
            # Load the first file, to get the number of generations
            identifier = dataset_name + '_subset_prop=' + f'{subset_prop:.4f}' + '_trial=' + str(1)
            with open(results_dir_path + dataset_name + '/' + 'GA_results_' + identifier + '.pkl', 'rb') as handle:
                best_solutions, best_solutions_fitness, evo_time_elapsed = pickle.load(handle)

            num_of_generations = len(best_solutions_fitness)

            best_fitness = np.zeros((num_of_trials, num_of_generations))
            evo_time = np.zeros((num_of_trials, num_of_generations))

            best_fitness[0, :] = best_solutions_fitness
            evo_time[0, :] = evo_time_elapsed

            for trial in range(2, num_of_trials+1):
                identifier = dataset_name + '_subset_prop=' + f'{subset_prop:.4f}' + '_trial=' + str(trial)
                with open(results_dir_path + dataset_name + '/' + 'GA_results_' + identifier + '.pkl', 'rb') as handle:
                    best_solutions, best_solutions_fitness, evo_time_elapsed = pickle.load(handle)

                best_fitness[trial-1, :] = best_solutions_fitness
                evo_time[trial-1] = evo_time_elapsed

            best_fitness_mean = np.mean(best_fitness, axis=0)
            best_fitness_std = np.std(best_fitness, axis=0)

            evo_time_mean = np.mean(evo_time, axis=0)

            label = str(subset_prop * 100)
            ax.plot(evo_time_mean, best_fitness_mean, label=label)
            ax.fill_between(evo_time_mean, best_fitness_mean - best_fitness_std, best_fitness_mean + best_fitness_std, alpha=0.2)

        # Invert the order of the legend labels
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[::-1]
        labels = labels[::-1]

        # Add dashed line that shows the accuracy obtained on the full training set
        ax.axhline(y=full_set_acc, color='k', linestyle='--', label='Full set accuracy')
        ax.annotate('Full training set \(ACC\): ' + f"{full_set_acc:.2f}", (1, full_set_acc + 0.01), color='k')

        ax.set_xlabel('Evolution time (s)')
        ax.set_title(dataset_name)
        ax.set_ylim(bottom=0.5, top=1.0)
        ax.grid()

    ax_legend = fig.add_subplot(gs[-1])
    ax_legend.axis('off')
    ax_legend.legend(title='Subset \n proportion (\%)', handles=handles, labels=labels, bbox_to_anchor=(-0.5, 1), loc='upper left')

    ax = fig.axes
    ax[0].set_ylabel('Best \(ACC\) / fitness')
    fig.suptitle('Evolution time vs Best fitness')
    plt.savefig(plots_dir_path + 'time_vs_best_fitness.png', dpi=300)

def acc_distribution_histograms_plot(dataset_names, subset_prop_list, results_dir_path, plots_dir_path):
    fig, ax = plt.subplots(nrows=len(dataset_names), ncols=1, figsize=(10, 5))

    for plot_idx, dataset_name in enumerate(dataset_names):
        # Set histogram bins
        bins = np.arange(0.30, 1.00, 0.001)

        for subset_prop in subset_prop_list:
            identifier = 'Random_subset_acc_' + dataset_name + '_subset_prop=' + f'{subset_prop:.4f}'
            acc_values = np.load(results_dir_path + dataset_name + '/' + identifier + '.npy')
            label = str(subset_prop * 100)
            ax[plot_idx].hist(acc_values, label=label, bins=bins, alpha=0.7)

        ax[plot_idx].set_ylabel('Frequency', fontsize=14)
        ax[plot_idx].grid()
    ax[plot_idx].set_xlabel('\(ACC\)', fontsize=14)
    ax[0].legend(title='Subset proportion (\%)', ncols=5)

    # Add dataset name to the left of the y-axis
    for ax, dataset_name in zip(ax[:], dataset_names):
        ax.annotate(dataset_name, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    fig.suptitle('Distribution of \(ACC\) for random balanced subsets - \n 10000 trials per subset proportion')
    plt.savefig(plots_dir_path + 'acc_distribution.png', dpi=300)

def violin_acc_plot(dataset_names, subset_prop_list, num_of_trials, results_dir_path, plots_dir_path):
    # Create figure and set size
    fig, ax = plt.subplots(nrows=1, ncols=len(dataset_names), figsize=(16, 8))

    for plot_idx, dataset_name in enumerate(dataset_names):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        random_subset_accs = []
        evo_sub_final_accs = np.zeros((len(subset_prop_list), num_of_trials))

        with open(results_dir_path + dataset_name + '/' + 'Full_set_acc_' + dataset_name + '.pkl', 'rb') as handle:
            full_set_acc = pickle.load(handle)

        # Setup inset axes
        acc_diff = 0.11
        x1, x2, y1, y2 = 5.5, 10.8, full_set_acc - acc_diff, full_set_acc  # subregion of the original image
        axins = ax[plot_idx].inset_axes(
            [x1 / 11, 0.1, (x2 - x1) / 11, 0.46],
            xlim=(x1, x2), ylim=(y1, y2), xticklabels=[])
        axins.set_ylim(bottom=full_set_acc - acc_diff, top=full_set_acc)

        for idx, subset_prop in enumerate(subset_prop_list):
            identifier_1 = 'Random_subset_acc_' + dataset_name + '_subset_prop=' + f'{subset_prop:.4f}'
            acc_values = np.load(results_dir_path + dataset_name + '/' + identifier_1 + '.npy')
            random_subset_accs.append(acc_values)

            for iter in range(num_of_trials):
                identifier_2 = dataset_name + '_subset_prop=' + f'{subset_prop:.4f}' + '_trial=' + str(iter+1)
                with open(results_dir_path + dataset_name + '/' + 'GA_results_' + identifier_2 + '.pkl', 'rb') as handle:
                    best_solutions, best_solutions_fitness, evo_time_elapsed = pickle.load(handle)
                    evo_sub_final_accs[idx, iter] = best_solutions_fitness[-1]

            random_acc_mean = np.mean(acc_values)
            evo_acc_mean = np.mean(evo_sub_final_accs[idx, :])
            ax[plot_idx].plot([idx+1+0.4, idx+1+0.4], [random_acc_mean  + 0.002, evo_acc_mean - 0.002], '_--b')
            ax[plot_idx].plot([idx+1+0.4, idx+1+0.4], [evo_acc_mean + 0.002, full_set_acc - 0.002], '_--r')

            # Plot accuracy gains and loss dashed lines
            acc_gain = 100*(evo_acc_mean - random_acc_mean)
            acc_loss = 100*(evo_acc_mean - full_set_acc)
            ax[plot_idx].annotate('+' +  "{:.1f}".format(acc_gain) + ' \%', (idx+1+0.25, random_acc_mean - 0.04), color='blue')

            # Print acc loss value above full training set acc line for larger subset proportions
            if idx < 5:
                ax[plot_idx].annotate("{:.1f}".format(acc_loss) + ' \%', (idx+1+0.45, evo_acc_mean + (full_set_acc - evo_acc_mean)/2), color='red')
            else:
                ax[plot_idx].annotate("{:.1f}".format(acc_loss) + ' \%', (idx+1+0.4, full_set_acc + 0.01), color='red')

        x_labels = [(str(subset_prop*100)) for subset_prop in subset_prop_list]
        violin_random = ax[plot_idx].violinplot(random_subset_accs, points=10000, showmeans=True)
        violin_random_inset = axins.violinplot(random_subset_accs, points=10000, showmeans=True)

        violin_evo = ax[plot_idx].violinplot(evo_sub_final_accs.T, points=num_of_trials, showmeans=True)
        violin_evo_inset = axins.violinplot(evo_sub_final_accs.T, points=10000, showmeans=True)

        # Set violin plot colors
        for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
            vp = violin_random[partname]
            vp.set_edgecolor('black')
            vp.set_facecolor('black')

            vp = violin_random_inset[partname]
            vp.set_edgecolor('black')
            vp.set_facecolor('black')

            vp = violin_evo[partname]
            vp.set_edgecolor('black')
            vp.set_facecolor('black')

            vp = violin_evo_inset[partname]
            vp.set_edgecolor('black')
            vp.set_facecolor('black')

        for vp_random, vp_random_inset, vp_evo, vp_evo_inset in zip(violin_random['bodies'], violin_random_inset['bodies'], violin_evo['bodies'], violin_evo_inset['bodies']):
            color = colors.pop(0)

            vp_random.set_edgecolor('black')
            vp_random.set_facecolor(color)
            vp_random.set_alpha(0.7)
            vp_random.set_hatch('///')

            vp_random_inset.set_edgecolor('black')
            vp_random_inset.set_facecolor(color)
            vp_random_inset.set_alpha(0.7)
            vp_random_inset.set_hatch('///')

            vp_evo.set_edgecolor('black')
            vp_evo.set_facecolor(color)
            vp_evo.set_alpha(0.7)

            vp_evo_inset.set_edgecolor('black')
            vp_evo_inset.set_facecolor(color)
            vp_evo_inset.set_alpha(0.7)

        # Add inset axes to plot
        axins.grid()
        ax[plot_idx].indicate_inset_zoom(axins, edgecolor="black")

        # Plot full training set acc line
        ax[plot_idx].axhline(y=full_set_acc, color='k', linestyle='--', label='Full set accuracy')
        ax[plot_idx].annotate('Full training set \(ACC\): ' + f'{full_set_acc:.2f}', (1, full_set_acc + 0.01), color='k')

        ax[plot_idx].set_xticks(range(1, len(subset_prop_list) + 1))
        ax[plot_idx].set_xticklabels(x_labels)
        ax[plot_idx].set_title(dataset_name)
        ax[plot_idx].set_xlabel('Subset proportions (\%)', fontsize=14)
        ax[plot_idx].set_ylim(bottom=0.25, top=1.0)
        ax[plot_idx].grid()

    legend_colors = ['lightgray', 'lightgray']
    legend_labels = ['EvoDSS', 'Random']
    legend_hatch = ['', '///']
    custom_handles = [mpatches.Patch(facecolor=color, label=label, hatch=hatch) for color, label, hatch in zip(legend_colors, legend_labels, legend_hatch)]
    ax[plot_idx].legend(handles=custom_handles, loc='lower right')

    ax[0].set_ylabel('\(ACC\)', fontsize=14)
    fig.suptitle('Distributions of \(ACC\) values for multiple subset proportions')
    plt.savefig(plots_dir_path + 'acc_distribution_violin_plot.png', dpi=300)

def incremental_learning_plot(dataset_names, memory_size_list, num_of_trials,  results_dir_path, plots_dir_path):
    seq_types = ['low', 'high']

    # Create figure and set size
    fig, ax = plt.subplots(nrows=len(dataset_names), ncols=len(seq_types), figsize=(6, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for plot_idx, dataset_name in enumerate(dataset_names):
        for i, seq_type in enumerate(seq_types):
            for j, memory_size in enumerate(memory_size_list):
                num_of_stages = 9
                random_subsets_acc = np.zeros((num_of_trials, num_of_stages))

                for trial in range(num_of_trials):
                    identifier = dataset_name + '_memory_size=' + f'{memory_size:.3f}' + '_' + seq_type + '_acc_seq_trial=' + str(trial+1)
                    acc_array = np.load(results_dir_path + dataset_name + '/' + 'Incremental_learning_acc_array_' + identifier + '.npy')
                    random_subsets_acc[trial, :] = acc_array[0, :]

                random_subsets_acc_mean = np.mean(random_subsets_acc, axis=0)
                random_subsets_acc_std = np.std(random_subsets_acc, axis=0)

                if memory_size > 0.125:
                    label = f'{memory_size:.1f}' + ' MB'
                else:
                    label = f'{memory_size:.3f}' + ' MB'

                ax[plot_idx, i].plot(np.arange(random_subsets_acc_mean.shape[0]), random_subsets_acc_mean, label=label, color=colors[j])
                ax[plot_idx, i].fill_between(np.arange(random_subsets_acc_mean.shape[0]), random_subsets_acc_mean - random_subsets_acc_std, random_subsets_acc_mean + random_subsets_acc_std, alpha=0.2, color=colors[j])

            for j, memory_size in enumerate(memory_size_list):
                num_of_stages = 9
                evo_subsets_acc = np.zeros((num_of_trials, num_of_stages))

                for trial in range(num_of_trials):
                    identifier = dataset_name + '_memory_size=' + f'{memory_size:.3f}' + '_' + seq_type + '_acc_seq_trial=' + str(trial+1)
                    acc_array = np.load(results_dir_path + dataset_name + '/' + 'Incremental_learning_acc_array_' + identifier + '.npy')
                    evo_subsets_acc[trial, :] = acc_array[1, :]

                evo_subsets_acc_mean = np.mean(evo_subsets_acc, axis=0)
                evo_subsets_acc_std = np.std(evo_subsets_acc, axis=0)

                if memory_size > 0.125:
                    label = f'{memory_size:.1f}' + ' MB'
                else:
                    label = f'{memory_size:.3f}' + ' MB'

                ax[plot_idx, i].plot(np.arange(evo_subsets_acc_mean.shape[0]), evo_subsets_acc_mean, color=colors[j], linestyle='--')
                ax[plot_idx, i].fill_between(np.arange(evo_subsets_acc_mean.shape[0]), evo_subsets_acc_mean - evo_subsets_acc_std, evo_subsets_acc_mean + evo_subsets_acc_std, alpha=0.2, color=colors[j])

            ax[plot_idx, i].set_ylim([0.55, 1])
            ax[plot_idx, i].set_xticks(range(0, random_subsets_acc_mean.shape[0]), labels=range(2, random_subsets_acc_mean.shape[0]+2))
            ax[plot_idx, i].grid()
            ax[len(dataset_names)-1 , i].set_xlabel('number of classes')
            ax[0, i].set_title(seq_type[0].upper() + seq_type[1:] + ' accuracy sequence')

        ax[plot_idx, 0].set_ylabel('\(ACC\)')

    # Add dataset name to the left of the y-axis
    for ax, dataset_name in zip(ax[:, 0], dataset_names):
        ax.annotate(dataset_name, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    custom_handles = [Line2D([0], [0], color='k', linestyle='-', lw=1, label='Random'),
                      Line2D([0], [0], color='k', linestyle='--', lw=1, label='EvoDSS')]

    # Invert the order of the legend labels
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]

    handles = custom_handles + handles
    labels = ['Random', 'EvoDSS'] + labels

    plt.legend(labels=labels, handles=handles)
    fig.suptitle('Class-incremental learning under memory constraints')
    plt.savefig(plots_dir_path + 'incremental_learning_plot.png', dpi=300)