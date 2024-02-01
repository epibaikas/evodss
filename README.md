# EvoDSS: Evolutinary Data Subset Selection for Incremental Learning on Memory-Constrained Systems

### This repository contains the open-source implementation for EvoDSS, a genetic algorithm for selecting subsets of data examples that need to be preserved in non-volatile memory for further reuse, while learning in a class-incremental supervised setting. This work has been submitted to GECCO 2024 (currently under review).


## Repository folder structure
This the original folder structure of the repo before running the scripts to generate results.
```
.
├── src         # Python source code 
├── scripts     # Bash scripts for generating results
└── tests       # Unit tests 
```

After running the scripts, the folder structure changes to:
```
.
├── artifacts   # Results from intermediate computations
├── datasets    # Dataset file, downloaded by PyTorch library
├── plots       # Plot figures created by generate_plots.py 
├── results     # Experimental results 
├── src         # Python source code 
├── scripts     # Bash scripts for generating results
└── tests       # Unit tests 
```

## Setup
### Setup the conda environment
Clone the repository and setup the conda environment to run EvoDSS code:
```
cd <path_to_evodss>
conda env create -f env.yml
conda activate evodss
```

## Reproduce experiments

To reproduce experimental results, follow the steps described below in the exact order.

(1) Generate the results for random balanced subsets:
```
cd scripts
sh random_subsets_script.sh mnist
sh random_subsets_script.sh fashion
```

(2) Run the greedy algorithm to find the "low" and "high" accuracy sequences:
```
python ../src/sequence_finder.py MNIST
python ../src/sequence_finder.py FashionMNIST
```

(3) Run experiment 1. The default list of subset proportions is `[0.005, 0.01, ..., 0.256]`. For every subset proportion, 
experiment is repeated for 20 trials. To change the subset proportions list and number of trials, modify the 
`subset_props` and `num_of_trials` parameters in `run_exp1.sh`.

```
sh run_exp1.sh mnist
sh run_exp1.sh fashion
```

(4) Run experiment 2. The default list of memory_sizes is `[0.125, 2, 16]`. For every memory size, the experiment is
repeated for 20 trials. To change the list of memory sizes and number of trials, modify the `memory_sizes` and
`num_of_trials` parameters in `run_exp2.sh`.
 
(5) Generate plots:
```
python ../src/generate_plots.py
```