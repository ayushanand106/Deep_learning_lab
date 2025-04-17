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
    parser = argparse.ArgumentParser(description='Train ResNet-18 on Dogs vs. Cats dataset')
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
def evaluate_model(model, loader, device='cuda'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float)
            outputs = model(inputs).squeeze()
            preds = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100 * correct / total

# Main function
def main():
    args = get_args()

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Dataset and splits
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader  = DataLoader(valset,  batch_size=64, shuffle=False)

    # Load ResNet-18 model and adapt final layer
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    initialize_weights(model, args.init)

    # Optimizer and loss
    optimizer_map = {
        'sgd':    optim.SGD(model.parameters(),    lr=1e-3),
        'adam':   optim.Adam(model.parameters(),   lr=1e-3, weight_decay=0.01),
        'rmsprop':optim.RMSprop(model.parameters(),lr=1e-3, weight_decay=0.01)
    }
    optimizer = optimizer_map[args.optimizer]
    criterion = nn.BCEWithLogitsLoss()

    # Training setup
    best_accuracy = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Logging
    log_dir = 'model_log'
    os.makedirs(log_dir, exist_ok=True)
    exp_dir = f"{log_dir}/resnet18_{args.optimizer}_{args.init}"
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, f"training_{datetime.now():%Y%m%d_%H%M%S}.txt")

    def write_log(msg):
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}\n")

    write_log("Training started on ResNet-18")

    # Training loop
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:
                write_log(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss/(i+1):.3f}")

        # Validation
        val_acc = evaluate_model(model, valloader, device)
        write_log(f"After Epoch {epoch+1}: Validation Accuracy: {val_acc:.2f}%")
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
            write_log(f"Saved new best model: {best_accuracy:.2f}%")

    write_log(f"Training complete. Best Validation Accuracy: {best_accuracy:.2f}%")

if __name__ == '__main__':
    main()
