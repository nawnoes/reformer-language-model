import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')

import re
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from reformer_pytorch import Reformer, ReformerLM
from transformers import BertTokenizer, PreTrainedTokenizer
from fairseq.optim.adafactor import Adafactor
import os
import json
import logging
from datetime import datetime
from dataloader.wiki import NamuWikiDataset, NamuWikiDatasetForMLM

class ReformerTester(object):
    def __init__(self,
                 dataset,
                 model,
                 tokenizer,
                 max_len,
                 pretrained_checkpoint_path,
                 device=None,
                 train_batch_size=8,
                 eval_batch_size=None,
                 tb_writer=False,
                 tb_dir='./tb_logs',
                 log_dir='../logs'):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.device = device
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tb_writer = tb_writer
        self.log_dir = log_dir

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if eval_batch_size is None:
            self.eval_batch_size = train_batch_size

        if tb_writer:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=tb_dir)

        logging.basicConfig(filename=f'{log_dir}/{datetime.now().date()}.log', level=logging.INFO)

    def build_dataloaders(self, train_test_split=0.1, train_shuffle=True, eval_shuffle=True):
        train_loader = DataLoader(self.dataset, batch_size=self.train_batch_size, shuffle=train_shuffle)
        return train_loader
    def test(self,
              epochs,
              train_dataloader,
              log_steps,
              ckpt_steps,
              ckpt_dir=None,
              gradient_accumulation_steps=1):
        optimizer = Adafactor(self.model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        losses = {}
        global_steps = 0
        local_steps = 0
        step_loss = 0.0

        if ckpt_dir is not None:
            assert os.path.isdir(ckpt_dir)
            try:
                logging.info(f'{datetime.now()} | Continuing from checkpoint...')
                self.model.load_state_dict(torch.load(self.pretrained_checkpoint_path, map_location=self.device))
            except Exception as e:
                logging.info(f'{datetime.now()} | No checkpoint was found | {e}')

        self.model.eval()

        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
            logging.info(f'{datetime.now()} | Utilizing {self.n_gpu} GPUs')

        self.model.to(self.device)
        logging.info(f'{datetime.now()} | Moved model to: {self.device}')
        logging.info(f'{datetime.now()} | train_batch_size: {self.train_batch_size} | eval_batch_size: {self.eval_batch_size}')
        logging.info(f'{datetime.now()} | Epochs: {epochs} | log_steps: {log_steps} | ckpt_steps: {ckpt_steps}')
        logging.info(f'{datetime.now()} | gradient_accumulation_steps: {gradient_accumulation_steps}')

        for epoch in range(epochs): #tqdm(range(epochs), desc='Epochs', position=0):
            logging.info(f'{datetime.now()} | Epoch: {epoch}')
            pb = tqdm(enumerate(train_dataloader),
                      desc=f'Epoch-{epoch} Iterator',
                      total=len(train_dataloader),
                      bar_format='{l_bar}{bar:10}{r_bar}'
                      )
            for step, batch in pb:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                output = self.model(inputs)

                orgin_inputs = inputs.clone().squeeze()
                origin_labels = labels.clone().squeeze()
                # only calculating loss on masked tokens
                loss_mx = labels != -100
                output = output[loss_mx].view(-1, self.tokenizer.vocab_size)
                labels = labels[loss_mx].view(-1)
                argmax_ouput = torch.argmax(output,dim=1)
                # print(f'label- {labels}')
                # print(f'argmax - {argmax_ouput}')
                # print(f'decode: label  - {self.tokenizer.decode(origin_labels.tolist())}')
                # print(f'decode: output - {self.tokenizer.decode(orgin_inputs.tolist())}')
                de_labels = self.tokenizer.convert_ids_to_tokens(labels)
                de_outputs=  self.tokenizer.convert_ids_to_tokens(argmax_ouput)

                for i in range(len(de_labels)):
                    None
                    # print(f'decode: label- {de_labels[i]} output- {de_outputs[i]} ')

                loss = loss_fn(output, labels)

                if gradient_accumulation_steps > 1:
                    loss /= gradient_accumulation_steps

                try:
                    loss.backward()
                except Exception as err:
                    print('loss.backward():',err)
                    continue

                step_loss += loss.item()
                losses[global_steps] = loss.item()
                local_steps += 1
                global_steps += 1
                print("losses: ",losses)
                if global_steps % gradient_accumulation_steps == 0:
                    optimizer.step()
                    self.model.zero_grad()

                if global_steps % log_steps == 0:
                    pb.set_postfix_str(f'''{datetime.now()} | Train Loss: {step_loss / local_steps} | Steps: {global_steps}''')
                    step_loss = 0.0
                    local_steps = 0
        return self.model

if __name__ == '__main__':
    wordpiece_vocab_path = "../data/vocab.txt"
    mini_data_path ="../data/mini_namuwiki.txt"
    test_data_path = "../data/mlm-test-data.txt"
    namuwiki_path ="../data/namuwiki.txt"
    kowiki_path = '../data/kowiki-512.txt'

    checkpoint_dir = "../checkpoints"
    checkpoint_path = f'{checkpoint_dir}/reformer.bin'
    pretrained_checkpoint_path = f'{checkpoint_dir}/epoch3-reformer-model.pt'

    # Model Hyperparameter
    max_len = 512
    batch_size = 1
    dim = 512
    depth = 6
    heads = 8
    causal = True

    # Train Hyperparameter
    epochs = 30
    log_steps = 10
    ckpt_steps = 100
    ckpt_dir = checkpoint_path
    gradient_accumulation_steps = 1

    tokenizer = BertTokenizer(vocab_file=wordpiece_vocab_path, do_lower_case=False)


    dataset = NamuWikiDatasetForMLM(tokenizer, max_len, path=test_data_path)

    model = ReformerLM(
        num_tokens=tokenizer.vocab_size,
        dim=dim,
        depth=depth,
        heads=heads,
        max_seq_len=max_len,
        causal=causal # auto-regressive 학습을 위한 설정
    )
    tester = ReformerTester(dataset, model, tokenizer,max_len, train_batch_size=batch_size, eval_batch_size=batch_size, pretrained_checkpoint_path=pretrained_checkpoint_path,)
    train_dataloader = tester.build_dataloaders(train_test_split=0)

    model = tester.test(epochs=epochs,
                          train_dataloader=train_dataloader,
                          log_steps=log_steps,
                          ckpt_steps=ckpt_steps,
                          ckpt_dir= checkpoint_dir,
                          gradient_accumulation_steps=gradient_accumulation_steps)

    torch.save(model, checkpoint_path)