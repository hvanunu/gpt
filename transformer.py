
import torch
import model
import dataset

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?

max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200

n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device:', device)

data = dataset.Dataset(batch_size, block_size)

model = model.BigramLanguageModel(n_embd, data.vocab_size, block_size, n_head, n_layer, learning_rate)
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# create a PyTorch optimizer
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = data.get_batch('train')
    model.backward(xb, yb)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(dataset.decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

