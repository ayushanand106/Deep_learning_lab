import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from datetime import datetime

# Set manual seed for reproducibility
torch.manual_seed(42)

# Argument parser
def get_args():
    parser = argparse.ArgumentParser(description='Train ResNet on Dogs vs. Cats dataset')
    parser.add_argument('--activation', type=str, choices=['relu', 'tanh', 'leaky_relu'], required=True,
                        help='Activation function')
    parser.add_argument('--init', type=str, choices=['xavier', 'kaiming', 'random'], required=True,
                        help='Weight initialization')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'rmsprop'], required=True,
                        help='Optimizer')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the Dogs vs. Cats dataset')
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

# Evaluate the model on the validation dataset
def evaluate_model(model, testloader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float)
            outputs = model(inputs).squeeze()
            preds = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Main function
def main():
    args = get_args()

    activation_map = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU()
    }

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        activation_map[args.activation],
        nn.Linear(128, 1)
    )

    initialize_weights(model, args.init)

    optimizer_map = {
        'sgd': optim.SGD(model.parameters(), lr=1e-3),
        'adam': optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01),
        'rmsprop': optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=0.01)
    }

    optimizer = optimizer_map[args.optimizer]
    criterion = nn.BCEWithLogitsLoss()

    best_accuracy = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    model_log = "model_log"
    os.makedirs(model_log, exist_ok=True)

    model_path = f'models_{args.optimizer}_{args.init}_{args.activation}'
    os.makedirs(os.path.join(model_log, model_path), exist_ok=True)
    model_path = os.path.join(model_log, model_path)

    log_file_path = os.path.join(model_path, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    def write_log(message):
        with open(log_file_path, 'a') as log_file:
            log_file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {message}\n')

    write_log("Training started")

    for epoch in range(50):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 0:
                write_log(f'[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / (i+1):.3f}')

        val_accuracy = evaluate_model(model, valloader, device)
        write_log(f'After Epoch {epoch+1}: Validation Accuracy: {val_accuracy:.2f}%')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            write_log(f"Saving new best model with accuracy: {best_accuracy:.2f}%")
            torch.save(model.state_dict(), os.path.join(model_path, 'best_model.pth'))

    write_log(f"Best accuracy: {best_accuracy:.2f}")

if __name__ == '__main__':
    main()
