import torch
from torch import randint

from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper


model = ReformerLM(
    num_tokens= 20000,
    dim = 1024,
    depth = 12,
    max_seq_len = 4096,
    lsh_dropout = 0.1,
    causal = True,
    full_attn_thres = 1024
)

# 0 is used for padding and no loss to be calculated on it
model = TrainingWrapper(model, ignore_index = 0, pad_value = 0)

# the wrapper can handle evenly packed sequences
x_train = randint(0, 20000, (3, 357))

# or if you have a list of uneven sequences, it will be padded for you
x_train = [
    randint(0, 20000, (120,)),
    randint(0, 20000, (253,)),
    randint(0, 20000, (846,))
]

# when training, set return_loss equal to True
model.train()
loss = model(x_train, return_loss = True)
loss.backward()

# when evaluating, just use the generate function, which will default to top_k sampling with temperature of 1.
initial = torch.tensor([[0]]).long() # assume 0 is start token
sample = model.generate(initial, 100, temperature=1., filter_thres = 0.9, eos_token = 1) # assume end token is 1, or omit and it will sample up to 100
print(sample.shape) # (1, <=100) token ids