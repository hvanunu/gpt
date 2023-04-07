
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Dataset():
    def __init__(self, batch_size, block_size):
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size

        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars) 

        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }

        self.encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
        self.decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
        
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9*len(data)) # first 90% will be train, rest val

        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
