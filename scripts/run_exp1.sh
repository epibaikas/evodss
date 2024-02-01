#!/bin/sh

# List of subset proportions
subset_props=("0.0005" "0.001" "0.002" "0.004" "0.008" "0.016" "0.032" "0.064" "0.128" "0.256")
num_of_trials=20

if [ "$1" = "fashion" ]; then
  dataset_name="FashionMNIST"
elif [ "$1" = "mnist" ]; then
  dataset_name="MNIST"
else
  echo "Incorrect dataset name"
  exit 1
fi

# Run the 1st experiment for all subset proportions
for subset_prop in "${subset_props[@]}"; do
  for ((trial=1; trial<=num_of_trials; trial++)); do
      echo "Now running: evo_subset_search.py dataset_name=${dataset_name}, subset_prop=${subset_prop}, trial=${trial}"
      python ../src/evo_subset_search.py $dataset_name $subset_prop $trial
  done
done

echo "Experiment 1 done!"