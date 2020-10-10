import warnings
warnings.filterwarnings("ignore")
from transformers import BertTokenizer
from fairseq.optim.adafactor import Adafactor
import os
import json
import torch
from model.autoregressive import ReformerAutoRegressiveModel

if __name__ =="__main__":
    wordpiece_vocab_path = "./data/wpm-vocab-all.txt"
    PATH= "./checkpoints/autoregressive_model_state_dict.pt"
    # dir_path ="/" # 2020-08-30 kowiki data path

    checkpoint_dir = "../checkpoints"
    checkpoint_path = f'{checkpoint_dir}/autoregressive_reformer.bin'

    # Model Hyperparameter
    """
    Model Name     layer      d_model      n_head    d_head    batchsize     learning rate     n_params
    GPT-3 samll     12         768           12        64         0.5M        6.0 x 10^-4       125M
    GPT-3 medium    24         1024          16        65         0.5M        3.0 x 10^-4       350M
    """
    max_len = 5120 # AxialPositionalEmbedding을 위한 (79,64) 값 and max_len/(bucket_size*2) ==0이어야함.
    batch_size = 2
    dim = 768
    depth = 12
    heads = 12
    causal = True # True for Auto Regressive,

    # Train Hyperparameter
    epochs = 3
    log_steps = 2
    ckpt_steps = 100
    ckpt_dir = checkpoint_path
    gradient_accumulation_steps = 1

    tokenizer = BertTokenizer(vocab_file=wordpiece_vocab_path, do_lower_case=False)

    model = ReformerAutoRegressiveModel(
        num_tokens=tokenizer.vocab_size,
        dim=dim,
        depth=depth,
        heads=heads,
        max_seq_len=max_len,
    )
    model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))

    sent = '2019년 한해를 보내며,'
    tokenized_sentence = tokenizer.encode(sent,add_special_tokens=False)
    while 1:
      input_ids = torch.tensor([tokenizer.cls_token_id,]  + tokenized_sentence).unsqueeze(0)
      pred = model(input_ids)[0]
      gen = tokenizer.decode(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]
      if gen == '[SEP]':
          break
      sent += gen.replace('▁', ' ')
      # toked = tok(sent)
    sent

