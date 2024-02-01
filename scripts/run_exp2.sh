#!/bin/sh

# List of subset proportions
memory_sizes=("0.125" "2" "16")
sequences=("low" "high")
num_of_trials=20

if [ "$1" = "fashion" ]; then
  dataset_name="FashionMNIST"
elif [ "$1" = "mnist" ]; then
  dataset_name="MNIST"
else
  echo "Incorrect dataset name"
  exit 1
fi

# Run the 2nd experiment for all memory sizes
for memory_size in "${memory_sizes[@]}"; do
  for sequence in "${sequences[@]}"; do
    for ((trial=1; trial<=num_of_trials; trial++)); do
        echo "Now running: class_incremental_learning.py dataset_name=${dataset_name}, memory_size=${memory_size}, sequence=${sequence}, trial=${trial}"
        python ../src/class_incremental_learning.py $dataset_name $memory_size $sequence $trial
    done
  done
done

echo "Experiment 2 done!"

