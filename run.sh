#!/bin/bash

# Define all options for activation functions, weight initializations, and optimizers
activations=("relu" "tanh" "leaky_relu")
initializations=("xavier" "kaiming" "random")
optimizers=("sgd" "adam" "rmsprop")

# Initialize a variable to track GPU alternation
gpu=0

# Iterate over all combinations of activation, initialization, and optimizer
for activation in "${activations[@]}"; do
    for init in "${initializations[@]}"; do
        for optimizer in "${optimizers[@]}"; do
            echo "Running combination: Activation=$activation, Init=$init, Optimizer=$optimizer on GPU=$gpu"
            
            # Set CUDA_VISIBLE_DEVICES to alternate between 0 and 1
            export CUDA_VISIBLE_DEVICES=$gpu
            
            # Run the command in the background
            python main.py --activation "$activation" --init "$init" --optimizer "$optimizer" &
            
            # Toggle GPU between 0 and 1
            if [ $gpu -eq 0 ]; then
                gpu=1
            else
                gpu=0
            fi
        done
    done
done

# Wait for all background processes to finish
wait

echo "All combinations have been processed."
