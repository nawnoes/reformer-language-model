from __future__ import absolute_import, division, print_function
import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import os
import random
import sys
# sys.path.append('/content/drive/My Drive/Colab Notebooks/reformer/')
from io import open

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler,SequentialSampler, TensorDataset)
from tqdm import tqdm, trange

from reformer_pytorch import ReformerLM
from util.arg import ElectraConfig
from model.electra import Electra
from model.electra_discriminator import DiscriminatorMRCModel
from transformers.optimization import AdamW
from util.schedule import WarmupLinearSchedule
from transformers import BertTokenizer
from util.korquad_utils import (read_squad_examples, convert_examples_to_features, RawResult, write_predictions,evaluate)

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
#######
# PATH
######
# gdrive_path = ".."
output_dir = ".."
# checkpoint = os.path.join(gdrive_path, "checkpoints/epoch27-reformer-small.pt")
train_file = "../data/korquad/KorQuAD_v1.0_train.json"
config_path = "../config/electra-korquad-finetuning.json"
#################
# Hyper Parameter
#################
doc_stride = 128
max_query_length = 96
max_answer_length = 30
n_best_size = 20

train_batch_size = 2
learning_rate = 5e-5
warmup_proportion = 0.1
num_train_epochs = 5.0

max_grad_norm = 1.0
adam_epsilon = 1e-6
weight_decay = 0.01

#################
# Device
#################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {} n_gpu: {}".format(device, n_gpu))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if n_gpu > 0:
        torch.cuda.manual_seed_all(SEED)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Config
train_config, gen_config, disc_config = ElectraConfig(config_path = config_path).get_config()

# 2. Tokenizer
tokenizer = BertTokenizer(vocab_file=train_config.vocab_path, do_lower_case=False)

# Generator
generator = ReformerLM(
    num_tokens= tokenizer.vocab_size,
    emb_dim= gen_config.emb_dim,
    dim= gen_config.emb_dim,  # smaller hidden dimension
    heads= gen_config.heads,  # less heads
    ff_mult= gen_config.ff_mult,  # smaller feed forward intermediate dimension
    dim_head= gen_config.dim_head,
    depth= gen_config.depth,
    max_seq_len= train_config.max_len
)

discriminator = ReformerLM(
    num_tokens= tokenizer.vocab_size,
    emb_dim= disc_config.emb_dim,
    dim= disc_config.dim,
    dim_head= disc_config.dim_head,
    heads= disc_config.heads,
    depth= disc_config.depth,
    ff_mult= disc_config.ff_mult,
    max_seq_len= train_config.max_len,
    return_embeddings=True,
)
# 4.2 weight tie the token and positional embeddings of generator and discriminator
# 제너레이터와 디스크리미네이터의 토큰, 포지션 임베딩을 공유한다(tie).
generator.token_emb = discriminator.token_emb
generator.pos_emb = discriminator.pos_emb
# weight tie any other embeddings if available, token type embeddings, etc.
# 다른 임베딩 웨이트도 있다면 공유 필요.

# 4.3 instantiate electra
# 엘렉트라 모델 초기화
discriminator_with_adapter = nn.Sequential(discriminator, nn.Linear(disc_config.dim, 1))

electra = Electra(
    generator,
    discriminator_with_adapter,
    mask_token_id = tokenizer.mask_token_id,           # the token id reserved for masking
    pad_token_id = tokenizer.pad_token_id,             # the token id for padding
    mask_prob = 0.15,                                  # masking probability for masked language modeling
    mask_ignore_token_ids = tokenizer.all_special_ids  # ids of tokens to ignore for mask modeling ex. (cls, sep)
)
electra.load_state_dict(torch.load(train_config.checkpoint_path, map_location=device),strict=False)

electra_discriminator = electra.discriminator[0]

model = DiscriminatorMRCModel(discriminator=electra_discriminator,dim=disc_config.dim)

###############
# Train Korquad
###############

model.electra.load_state_dict(torch.load(train_config.checkpoint_path, map_location=device),strict=False)
num_params = count_parameters(model)
logger.info("Total Parameter: %d" % num_params)
model.to(device)

cached_train_features_file = train_file + '_{0}_{1}_{2}'.format(str(train_config.max_len), str(doc_stride),
                                                                      str(max_query_length))
train_examples = read_squad_examples(input_file=train_file, is_training=True, version_2_with_negative=False)
try:
    with open(cached_train_features_file, "rb") as reader:
        train_features = pickle.load(reader)
except:
    train_features = convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=train_config.max_len,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=True)
    logger.info("  Saving train features into cached file %s", cached_train_features_file)
    with open(cached_train_features_file, "wb") as writer:
        pickle.dump(train_features, writer)

num_train_optimization_steps = int(len(train_features) / train_batch_size) * num_train_epochs

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
      'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,
                  lr=learning_rate,
                  eps=adam_epsilon)
scheduler = WarmupLinearSchedule(optimizer,
                                  warmup_steps=num_train_optimization_steps*0.1,
                                  t_total=num_train_optimization_steps)

logger.info("***** Running training *****")
logger.info("  Num orig examples = %d", len(train_examples))
logger.info("  Num split examples = %d", len(train_features))
logger.info("  Batch size = %d", train_batch_size)
logger.info("  Num steps = %d", num_train_optimization_steps)
num_train_step = num_train_optimization_steps

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_start_positions, all_end_positions)

train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

model.train()
global_step = 0
epoch = 0
for i in range(int(num_train_epochs)):
    iter_bar = tqdm(train_dataloader, desc=f"Epoch-{i} Train(XX Epoch) Step(XX/XX) (Mean loss=X.X) (loss=X.X)")
    tr_step, total_loss, mean_loss = 0, 0., 0.
    for step, batch in enumerate(iter_bar):
        if n_gpu == 1:
            batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
        input_ids, input_mask, segment_ids, start_positions, end_positions = batch
        loss = model(input_ids, start_positions, end_positions)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scheduler.step()
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        tr_step += 1
        total_loss += loss.item()
        mean_loss = total_loss / tr_step
        iter_bar.set_description(f"Epoch-{i} Train Step(%d / %d) (Mean loss=%5.5f) (loss=%5.5f)" %
                                  (global_step, num_train_step, mean_loss, loss.item()))

    logger.info("** ** * Saving file * ** **")
    model_checkpoint = "electra_korquad_%d.bin" % (epoch)
    logger.info(model_checkpoint)
    output_model_file = os.path.join(output_dir,model_checkpoint)

    torch.save(model.state_dict(), output_model_file)
    epoch += 1
