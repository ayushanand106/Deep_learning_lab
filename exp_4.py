# Experiment 6: Text Generation with RNNs and LSTM Embeddings
# -------------------------------------------------------------
# Usage example:
#   python experiment6_textgen.py --data_path data/poems.txt --epochs 100 --batch_size 64 --seq_len 50 --lr 1e-3

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# 1. Dataset and Vocabulary
class TextDataset(Dataset):
    def __init__(self, path, seq_len):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().lower()
        self.chars = sorted(set(text))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.seq_len = seq_len
        data = [self.char2idx[ch] for ch in text if ch in self.char2idx]
        self.data = torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y

# 2. Models
class OneHotRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(vocab_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h0=None):
        # x: (batch, seq, vocab)
        out, hn = self.rnn(x, h0)
        logits = self.fc(out)  # (batch, seq, vocab)
        return logits, hn

class EmbeddingLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hx=None):
        # x: (batch, seq)
        emb = self.embedding(x)
        out, hx = self.lstm(emb, hx)
        logits = self.fc(out)
        return logits, hx

# 3. Training Loop

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        if isinstance(model, OneHotRNN):
            # convert to one-hot
            x_oh = torch.zeros(x_batch.size(0), x_batch.size(1), model.rnn.input_size, device=device)
            x_oh.scatter_(2, x_batch.unsqueeze(-1), 1)
            logits, _ = model(x_oh)
        else:
            logits, _ = model(x_batch)
        loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 4. Generation

def generate_text(model, dataset, seed, length, device):
    model.eval()
    chars = []
    input_seq = torch.tensor([dataset.char2idx[ch] for ch in seed], device=device).unsqueeze(0)
    hx = None
    for _ in range(length):
        if isinstance(model, OneHotRNN):
            x_oh = torch.zeros(1, input_seq.size(1), model.rnn.input_size, device=device)
            x_oh.scatter_(2, input_seq.unsqueeze(-1), 1)
            logits, hx = model(x_oh, hx)
        else:
            logits, hx = model(input_seq, hx)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        chars.append(dataset.idx2char[idx.item()])
        input_seq = torch.cat([input_seq[:, 1:], idx], dim=1)
    return seed + ''.join(chars)

# 5. Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = TextDataset(args.data_path, args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # instantiate models
    onehot_rnn = OneHotRNN(dataset.vocab_size, args.hidden_size).to(device)
    embed_lstm = EmbeddingLSTM(dataset.vocab_size, args.embed_size, args.hidden_size).to(device)

    # criteria and optimizers
    criterion = nn.CrossEntropyLoss()
    opt_rnn = optim.Adam(onehot_rnn.parameters(), lr=args.lr)
    opt_lstm = optim.Adam(embed_lstm.parameters(), lr=args.lr)

    losses_rnn, losses_lstm = [], []

    # training
    start_rnn = time.time()
    for ep in range(1, args.epochs + 1):
        loss = train(onehot_rnn, dataloader, criterion, opt_rnn, device)
        losses_rnn.append(loss)
        if ep % 10 == 0:
            print(f"One-Hot RNN Epoch {ep}/{args.epochs} Loss: {loss:.4f}")
    time_rnn = time.time() - start_rnn

    start_lstm = time.time()
    for ep in range(1, args.epochs + 1):
        loss = train(embed_lstm, dataloader, criterion, opt_lstm, device)
        losses_lstm.append(loss)
        if ep % 10 == 0:
            print(f"Embedding LSTM Epoch {ep}/{args.epochs} Loss: {loss:.4f}")
    time_lstm = time.time() - start_lstm

    # plot and save loss curves
    plt.figure()
    plt.plot(losses_rnn, label='One-Hot RNN')
    plt.plot(losses_lstm, label='Embedding LSTM')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curves.png')

    # generation
    seed = input("Enter seed text: ")
    print("\nGenerated (One-Hot RNN):\n", generate_text(onehot_rnn, dataset, seed, 200, device))
    print("\nGenerated (Embedding LSTM):\n", generate_text(embed_lstm, dataset, seed, 200, device))

    # summary
    print(f"\nOne-Hot RNN training time: {time_rnn:.2f}s")
    print(f"Embedding LSTM training time: {time_lstm:.2f}s")

if __name__ == '__main__':
    main()
