import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from datetime import datetime

torch.manual_seed(42)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, activation_fn):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.activation = activation_fn

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Argument parser
def get_args():
    parser = argparse.ArgumentParser(description='Train CNN on CIFAR-10 with different configurations')
    parser.add_argument('--activation', type=str, choices=['relu', 'tanh', 'leaky_relu'], required=True,
                        help='Activation function')
    parser.add_argument('--init', type=str, choices=['xavier', 'kaiming', 'random'], required=True,
                        help='Weight initialization')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'rmsprop'], required=True,
                        help='Optimizer')
    return parser.parse_args()

# Weight initialization function
def initialize_weights(model, init_type):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_type == 'random':
                nn.init.uniform_(m.weight, -0.1, 0.1)

# Evaluate the model on the test dataset
def evaluate_model(model, testloader, device='cuda'):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, total=len(testloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Main function
def main():
    args = get_args()

    # Map activation functions
    activation_map = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU()
    }

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    # Initialize model with the selected activation function
    model = CNN(activation_map[args.activation])
    initialize_weights(model, args.init)

    # Map optimizers
    optimizer_map = {
        'sgd': optim.SGD(model.parameters(), lr=1e-3),
        'adam': optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01),
        'rmsprop': optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=0.01)
    }

    optimizer = optimizer_map[args.optimizer]

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Variables to track the best model
    best_accuracy = 0.0
    device = 'cuda'
    model.to(device)
    
    # Create directory for logs and models
    model_log = "model_log"
    os.makedirs(model_log, exist_ok=True)

    # Create model path and set up file-based logging
    model_path = f'models_{args.optimizer}_{args.init}_{args.activation}'
    os.makedirs(os.path.join(model_log, model_path), exist_ok=True)
    model_path = os.path.join(model_log, model_path)

    # Define the log file path with .txt extension
    log_file_path = os.path.join(model_path, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    # Function to write logs to the text file
    def write_log(message):
        with open(log_file_path, 'a') as log_file:
            log_file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {message}\n')

    # Start training and logging
    write_log("Training started")

    # Training loop
    for epoch in tqdm(range(10)):  # Training for 10 epochs as an example
        model.train()  # Set model to training mode
        running_loss = 0.0

        for i, (inputs, labels) in tqdm(enumerate(trainloader), total=len(trainloader)):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 0:  # Log every 100 mini-batches
                write_log(f'[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        # Evaluate on test data after each epoch
        test_accuracy = evaluate_model(model, testloader)
        write_log(f'After Epoch {epoch+1}: Test Accuracy: {test_accuracy:.2f}%')

        # Save the best model based on test accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            write_log(f"Saving new best model with accuracy: {best_accuracy:.2f}%")
            torch.save(model.state_dict(), os.path.join(model_path,'best_model.pth'))

    write_log(f"Best accuracy: {best_accuracy:.2f}")

if __name__ == '__main__':
    main()
