import torch
from reformer_pytorch import ReformerLM

model = ReformerLM(
    num_tokens= 20000,      # 전체 vocab내 있는 단어의 수
    dim = 1024,             # 임베딩 차원
    depth = 12,             # 레이어 수
    max_seq_len = 8192,     # 최대 Sequence 수
    heads = 8,              # 어텐션 헤더 갯수
    lsh_dropout = 0.1,      # lsh drop out 비율
    ff_dropout = 0.1,       # ff 레이어 내 drop out 비율
    post_attn_dropout = 0.1,
    layer_dropout = 0.1,  # layer dropout from 'Reducing Transformer Depth on Demand' paper
    causal = True,        # auto-regressive or not
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    emb_dim = 128,        # embedding factorization for further memory savings
    dim_head = 64,        # be able to fix the dimension of each head, making it independent of the embedding dimension and the number of heads
    ff_chunks = 200,      # number of chunks for feedforward layer, make higher if there are memory issues
    attn_chunks = 8,      # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    num_mem_kv = 128,       # persistent learned memory key values, from all-attention paper
    twin_attention = False, # both branches of the reversible network will be attention
    full_attn_thres = 1024, # use full attention if context length is less than set value
    reverse_thres = 1024,   # turn off reversibility for 2x speed for sequence lengths shorter or equal to the designated value
    use_scale_norm = False,  # use scale norm from 'Transformers without tears' paper
    use_rezero = False,      # remove normalization and use rezero from 'ReZero is All You Need'
    one_value_head = False,  # use one set of values for all heads from 'One Write-Head Is All You Need'
    weight_tie = False,           # tie parameters of each layer for no memory per additional depth
    weight_tie_embedding = False, # use token embedding for projection of output, some papers report better results
    n_local_attn_heads = 2,       # many papers suggest mixing local attention heads aids specialization and improves on certain tasks
    pkm_layers = (4,7),           # specify layers to use product key memory. paper shows 1 or 2 modules near the middle of the transformer is best
    pkm_num_keys = 128,           # defaults to 128, but can be increased to 256 or 512 as memory allows
    use_full_attn = False    # only turn on this flag to override and turn on full attention for all sequence lengths. for comparison with LSH to show that it is working
)

x = torch.randint(0, 20000, (1, 8192)).long()
y = model(x) # (1, 8192, 20000)

print(y)