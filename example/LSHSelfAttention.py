import torch
import torch.nn as nn
from transformers import BertTokenizer
from reformer_pytorch import LSHSelfAttention

attn = LSHSelfAttention(
    dim = 128,
    heads = 8,
    bucket_size = 64,
    n_hashes = 8,
    # use_full_attn=False,
    # causal = True
)

wordpiece_vocab_path = "../data/vocab.txt"
tokenizer = BertTokenizer(vocab_file=wordpiece_vocab_path, do_lower_case=False)
num_tokens = tokenizer.vocab_size
emb_dim = 128

str = "안녕하세요..나는 리포머를 공부하고 있어요. 정말 어렵네요"
str2 = "리포머는 트랜스포머의 효율성을 개선하기 위한 알고리즘입니다."

token_emb = nn.Embedding(num_tokens, emb_dim)
x = token_emb(torch.tensor(tokenizer.encode(str)))
# x = torch.randn(10, 1024, 128)
y = attn(x.unsqueeze(0)) # (10, 1024, 128)
print(y)