# coding=utf-8
from __future__ import absolute_import, division, print_function
import warnings
warnings.filterwarnings("ignore")

import logging
import os
import random
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler,SequentialSampler, TensorDataset)
from tqdm import tqdm, trange

from reformer_pytorch import ReformerLM
from util.arg import ElectraConfig
from model.electra import Electra
from model.electra_discriminator import DiscriminatorMRCModel
from transformers import BertTokenizer
from util.korquad_utils import (read_squad_examples, convert_examples_to_features, RawResult, write_predictions,evaluate)
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/electra_korquad_14.bin")
CONFIG_PATH = os.path.join(MODEL_PATH, "config/electra-korquad-finetuning.json")
VOCAB_PATH = os.path.join(MODEL_PATH, "data/wpm-vocab-all.txt")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main(input, output):
    max_seq_length = 512
    doc_stride = 64
    max_query_length = 64
    batch_size = 16
    n_best_size = 20
    max_answer_length = 30
    seed = 42
    fp16 = False

    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # 1. Config
    train_config, gen_config, disc_config = ElectraConfig(config_path=CONFIG_PATH).get_config()

    # 2. Tokenizer
    tokenizer = BertTokenizer(vocab_file=train_config.vocab_path, do_lower_case=False)

    # 3. Generator
    generator = ReformerLM(
      num_tokens=tokenizer.vocab_size,
      emb_dim=gen_config.emb_dim,
      dim=gen_config.emb_dim,  # smaller hidden dimension
      heads=gen_config.heads,  # less heads
      ff_mult=gen_config.ff_mult,  # smaller feed forward intermediate dimension
      dim_head=gen_config.dim_head,
      depth=gen_config.depth,
      max_seq_len=train_config.max_len
    )
    # 4. Discriminator
    discriminator = ReformerLM(
      num_tokens=tokenizer.vocab_size,
      emb_dim=disc_config.emb_dim,
      dim=disc_config.dim,
      dim_head=disc_config.dim_head,
      heads=disc_config.heads,
      depth=disc_config.depth,
      ff_mult=disc_config.ff_mult,
      max_seq_len=train_config.max_len,
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
      mask_token_id=tokenizer.mask_token_id,  # the token id reserved for masking
      pad_token_id=tokenizer.pad_token_id,  # the token id for padding
      mask_prob=0.15,  # masking probability for masked language modeling
      mask_ignore_token_ids=tokenizer.all_special_ids  # ids of tokens to ignore for mask modeling ex. (cls, sep)
    )
    # electra.load_state_dict(torch.load(train_config.checkpoint_path, map_location=device),strict=False)

    electra_discriminator = electra.discriminator[0]

    model = DiscriminatorMRCModel(discriminator=electra_discriminator, dim=disc_config.dim)

    eval_examples = read_squad_examples(input_file=input, is_training=False, version_2_with_negative=False)
    eval_features = convert_examples_to_features(
      examples=eval_examples,
      tokenizer=tokenizer,
      max_seq_length=max_seq_length,
      doc_stride=doc_stride,
      max_query_length=max_query_length,
      is_training=False)

    if fp16 is True:
      model.half()
    model.load_state_dict(torch.load(CHK_PATH, map_location=device))
    model.to(device)
    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()
    all_results = []
    logger.info("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    output_nbest_file = os.path.join("nbest_predictions.json")
    write_predictions(eval_examples, eval_features, all_results,
                        n_best_size, max_answer_length,
                        False, output, output_nbest_file,
                        None, False, False, 0.0)

if __name__ == "__main__":
    print(sys.argv)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    data_name = input_file.split('.')[0]
    print('input file:', input_file)
    print('output file:', output_file)
    main(input_file, output_file)