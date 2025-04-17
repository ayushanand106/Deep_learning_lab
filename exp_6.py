# seq2seq_bahdanau_attention.py
# Train a Seq2Seq RNN with Bahdanau (additive) Attention on a toy translation dataset (Multi30k)

import argparse
import math
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy

# Tokenizers

def tokenize_de(text):
    return [tok.text for tok in spacy.load('de').tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy.load('en').tokenizer(text)]

# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [src_len, batch]
        embedded = self.dropout(self.embedding(src))
        # embedded: [src_len, batch, emb_dim]
        outputs, hidden = self.rnn(embedded)
        # outputs: [src_len, batch, hid_dim * 2]
        # hidden: [n_layers * 2, batch, hid_dim]
        # concatenate the final forward and backward hidden states
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        # hidden: [batch, hid_dim]
        return outputs, hidden.unsqueeze(0)

# Bahdanau (Additive) Attention
class BahdanauAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # hid_dim from encoder (bidirectional merged) and decoder hidden
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hid_dim]
        # encoder_outputs: [src_len, batch, hid_dim * 2]
        src_len = encoder_outputs.shape[0]
        # repeat hidden to [src_len, batch, hid_dim]
        hidden = hidden.unsqueeze(0).repeat(src_len, 1, 1)
        # energy: [src_len, batch, hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # attention: [src_len, batch]
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=0)

# Decoder using Bahdanau Attention
class DecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim, num_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim * 3 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch]
        input = input.unsqueeze(0)  # [1, batch]
        embedded = self.dropout(self.embedding(input))  # [1, batch, emb_dim]
        # calculate attention weights
        a = self.attention(hidden[-1], encoder_outputs)  # [src_len, batch]
        a = a.unsqueeze(1).permute(1,2,0)  # [batch, 1, src_len]
        encoder_outputs = encoder_outputs.permute(1,0,2)  # [batch, src_len, hid_dim*2]
        weighted = torch.bmm(a, encoder_outputs)  # [batch, 1, hid_dim*2]
        weighted = weighted.permute(1,0,2)  # [1, batch, hid_dim*2]
        # concatenate embedding and context
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [1, batch, emb_dim + hid_dim*2]
        output, hidden = self.rnn(rnn_input, hidden)
        # remove sequence dim
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        # concat output, context, and embedded
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction: [batch, output_dim]
        return prediction, hidden

# Seq2Seq Wrapper
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch], trg: [trg_len, batch]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0,:]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

# Training and Evaluation Functions

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        src, trg = batch.src, batch.trg
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            src, trg = batch.src, batch.trg
            output = model(src, trg, 0)  # no teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed = end_time - start_time
    return int(elapsed // 60), int(elapsed % 60)

# Main entry point

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--emb_size', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--clip', type=float, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
    TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=args.batch_size,
        device=device)

    encoder = EncoderRNN(len(SRC.vocab), args.emb_size, args.hidden_size, args.n_layers, args.dropout)
    attention = BahdanauAttention(args.hidden_size)
    decoder = DecoderRNN(len(TRG.vocab), args.emb_size, args.hidden_size, args.n_layers, args.dropout, attention)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])

    best_valid_loss = float('inf')

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, args.clip)
        valid_loss = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'seq2seq_bahdanau.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'	Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'	 Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

if __name__ == '__main__':
    main()
