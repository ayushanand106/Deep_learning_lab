#!/bin/bash

# Define all options for activation functions, weight initializations, and optimizers
activations=("relu" "tanh" "leaky_relu")
initializations=("xavier" "kaiming" "random")
optimizers=("sgd" "adam" "rmsprop")

# Generate all combinations
combinations=()
for activation in "${activations[@]}"; do
    for init in "${initializations[@]}"; do
        for optimizer in "${optimizers[@]}"; do
            combinations+=("$activation $init $optimizer")
        done
    done
done

# Run combinations in parallel using GNU Parallel, alternating CUDA_VISIBLE_DEVICES between 0 and 1
printf "%s\n" "${combinations[@]}" | parallel -j 4 --colsep ' ' \
    'CUDA_VISIBLE_DEVICES=$(({%} % 2)); python main.py --activation {1} --init {2} --optimizer {3}'

echo "All combinations have been processed."
