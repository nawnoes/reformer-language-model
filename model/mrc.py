import torch
import torch.nn as nn
from reformer_pytorch import ReformerLM
from torch.nn import CrossEntropyLoss
from transformers.activations import get_activation

class ReformerMRCHead(nn.Module):
  def __init__(self, dim, num_labels,hidden_dropout_prob=0.1):
    super().__init__()
    self.dense = nn.Linear(dim, 4*dim)
    self.dropout = nn.Dropout(hidden_dropout_prob)
    self.out_proj = nn.Linear(4*dim,num_labels)

  def forward(self, x, **kwargs):
    # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    x = self.dropout(x)
    x = self.dense(x)
    x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
    x = self.dropout(x)
    x = self.out_proj(x)
    return x

class ReformerMRCModel(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads, num_labels=2, causal=True):
        super().__init__()
        self.reformer = ReformerLM(
                num_tokens= num_tokens,
                dim= dim,
                depth= depth,
                heads= heads,
                max_seq_len= max_seq_len,
                causal= causal,           # auto-regressive 학습을 위한 설정
                return_embeddings=True    # reformer 임베딩을 받기 위한 설정
            )
        self.mrc_head = ReformerMRCHead(dim, num_labels)
    def forward(self,
                input_ids=None,
                start_positions=None,
                end_positions=None,
                **kwargs):
        # 1. reformer의 출력
        outputs = self.reformer(input_ids,**kwargs)

        # 2. mrc를 위한
        logits = self.mrc_head(outputs)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits