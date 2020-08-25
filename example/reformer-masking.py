import torch
from reformer_pytorch import ReformerLM

CONTEXT_LEN = 512
SEQ_LEN = 8192

model = ReformerLM(
    num_tokens= 20000,
    dim = 1024,
    depth = 1,
    max_seq_len = SEQ_LEN,
    ff_chunks = 8,
    causal = True
)

c = torch.randn(1, CONTEXT_LEN, 1024)
x = torch.randint(0, 20000, (1, SEQ_LEN)).long()

i_mask = torch.ones(1, SEQ_LEN).bool()
c_mask = torch.ones(1, CONTEXT_LEN).bool()

y = model(x, keys = c, input_mask = i_mask, context_mask = c_mask)
# masking done correctly in LSH attention