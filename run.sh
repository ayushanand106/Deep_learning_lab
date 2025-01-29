#!/bin/bash

# Define all options for activation functions, weight initializations, and optimizers
activations=("relu" "tanh" "leaky_relu")
initializations=("xavier" "kaiming" "random")
optimizers=("sgd" "adam" "rmsprop")

# Iterate over all combinations of activation, initialization, and optimizer
for activation in "${activations[@]}"; do
    for init in "${initializations[@]}"; do
        for optimizer in "${optimizers[@]}"; do
            echo "Running combination: Activation=$activation, Init=$init, Optimizer=$optimizer"
            
            # Construct and run the command
            python main.py --activation "$activation" --init "$init" --optimizer "$optimizer"
            
            # Check if the command succeeded
            if [ $? -ne 0 ]; then
                echo "Error occurred while running Activation=$activation, Init=$init, Optimizer=$optimizer"
            fi
            
            echo "----------------------------------------"
        done
    done
done
