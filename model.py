
import torch
import torch.nn as nn
import Block as block
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BigramLanguageModel(nn.Module):
    def __init__(self, n_embd, vocab_size, block_size, n_head, n_layer, learning_rate):
        super().__init__()
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_head = n_head
        self.learning_rate = learning_rate
        self.n_layer = n_layer
        self.dropout = 0.0

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        
        self.blocks = nn.Sequential(*[block.Block(self.n_embd, n_head=n_head, block_size=self.block_size, dropout=self.dropout ) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd) # final layer norm
        self.lm_head = nn.Linear(self.n_embd, vocab_size)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def backward (self, xb, yb):
        # evaluate the loss
        logits, loss = self(xb, yb)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    