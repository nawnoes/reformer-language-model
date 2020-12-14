import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from transformers import BertTokenizer
from fairseq.optim.adafactor import Adafactor
import os
import json
import logging
from datetime import datetime
from dataloader.electra import ElectraDataset
from electra_pytorch import Electra
from reformer_pytorch import ReformerLM
from util.arg import ElectraConfig
from model.electra_discriminator import DiscriminatorMRCModel

if __name__ == '__main__':
    torch.manual_seed(9)
    # 1. Config
    train_config, gen_config, disc_config = ElectraConfig().get_config()

    # 2. Tokenizer
    tokenizer = BertTokenizer(vocab_file=train_config.vocab_path, do_lower_case=False)

    electra_discriminator = ReformerLM(
      num_tokens=tokenizer.vocab_size,
      emb_dim=disc_config.emb_dim,
      dim=disc_config.dim,
      dim_head=disc_config.dim_head,
      heads=disc_config.heads,
      depth=disc_config.depth,
      ff_mult=disc_config.ff_mult,
      max_seq_len=train_config.max_len,
      return_embeddings=True,
      weight_tie_embedding=True
    )

    model = DiscriminatorMRCModel(discriminator=electra_discriminator,dim=disc_config.dim)

