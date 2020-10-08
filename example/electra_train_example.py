import torch
from torch import nn
from reformer_pytorch import ReformerLM

from electra_pytorch import Electra

# (1) instantiate the generator and discriminator, making sure that the generator is roughly a quarter to a half of the size of the discriminator

generator = ReformerLM(
    num_tokens = 20000,
    emb_dim = 128,
    dim = 256,              # smaller hidden dimension
    heads = 4,              # less heads
    ff_mult = 2,            # smaller feed forward intermediate dimension
    dim_head = 64,
    depth = 12,
    max_seq_len = 1024
)

discriminator = ReformerLM(
    num_tokens = 20000,
    emb_dim = 128,
    dim = 1024,
    dim_head = 64,
    heads = 16,
    depth = 12,
    ff_mult = 4,
    max_seq_len = 1024
)

# (2) weight tie the token and positional embeddings of generator and discriminator

generator.token_emb = discriminator.token_emb
generator.pos_emb = discriminator.pos_emb
# weight tie any other embeddings if available, token type embeddings, etc.

# (3) instantiate electra

trainer = Electra(
    generator,
    discriminator,
    discr_dim = 1024,           # the embedding dimension of the discriminator, discriminator의 임베딩 차원
    discr_layer = 'reformer',   # the layer name in the discriminator, whose output would be used for predicting token is still the same or replaced
    mask_token_id = 2,          # the token id reserved for masking, 마스크 토큰 인덱스
    pad_token_id = 0,           # the token id for padding, 패딩 토큰 인덱스
    mask_prob = 0.15,           # masking probability for masked language modeling,
    mask_ignore_token_ids = []  # ids of tokens to ignore for mask modeling ex. (cls, sep)
)

# (4) train

data = torch.randint(0, 20000, (1, 1024))

results = trainer(data)
results.loss.backward()

# after much training, the discriminator should have improved

# torch.save(discriminator, f'./pretrained-model.pt')
