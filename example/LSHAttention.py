import torch
from reformer_pytorch import LSHAttention

attn = LSHAttention(
    bucket_size = 64,
    n_hashes = 16,
    causal = True
)

qk = torch.randn(10, 1024, 128)
v = torch.randn(10, 1024, 128)

out, attn, buckets = attn(qk, v) # (10, 1024, 128)