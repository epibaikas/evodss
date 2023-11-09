#!/bin/sh

# List of arguments
args=("0.0005" "0.001" "0.002" "0.004" "0.008" "0.016" "0.032" "0.064" "0.128" "0.256")

if [ "$1" = "fashion" ]; then
  dataset_name="FashionMNIST"
elif [ "$1" = "mnist" ]; then
  dataset_name="MNIST"
else
  echo "Incorrect dataset name"
  exit 1
fi

# Loop through the arguments and run the Python script
for arg in "${args[@]}"; do
    echo "Subset proportion: " $arg
    python ../src/random_subsets.py $dataset_name "$arg" 10000
done