import itertools
import subprocess

# Define all options for activation functions, weight initializations, and optimizers
activations = ['relu', 'tanh', 'leaky_relu']
initializations = ['xavier', 'kaiming', 'random']
optimizers = ['sgd', 'adam', 'rmsprop']

# Iterate over all combinations of activation, initialization, and optimizer
for activation_name, init_type, optimizer_type in itertools.product(activations, initializations, optimizers):
    print(f"Running combination: Activation={activation_name}, Init={init_type}, Optimizer={optimizer_type}")
    
    # Construct the command to run main.py with the current combination of arguments
    command = [
        "python", "main.py",
        "--activation", activation_name,
        "--init", init_type,
        "--optimizer", optimizer_type
    ]
    
    # Run the command using subprocess
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Print the output of the current run
    print(result.stdout)
    print(result.stderr)
