import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')

import re
import argparse
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


class ElectraTrainer(object):
    def __init__(self,
                 dataset,
                 model,
                 tokenizer,
                 max_len,
                 device=None,
                 train_batch_size=8,
                 eval_batch_size=None,
                 log_dir='../logs'):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.log_dir = log_dir

        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if eval_batch_size is None:
            self.eval_batch_size = train_batch_size


        logging.basicConfig(filename=f'{log_dir}/electra-{datetime.now().date()}.log', level=logging.INFO)

    def build_dataloaders(self, train_test_split=0.1, train_shuffle=True, eval_shuffle=True):
        """
        Builds the Training and Eval DataLoaders
        :param train_test_split: The ratio split of test to train data.
        :param train_shuffle: (bool) True if you wish to shuffle the train_dataset.
        :param eval_shuffle: (bool) True if you wish to shuffle the eval_dataset.
        :return: train dataloader and evaluation dataloader.
        """
        dataset_len = len(self.dataset)
        eval_len = int(dataset_len * train_test_split)
        train_len = dataset_len - eval_len
        train_dataset, eval_dataset = random_split(self.dataset, (train_len, eval_len))
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=train_shuffle)
        eval_loader = DataLoader(eval_dataset, batch_size=self.eval_batch_size, shuffle=eval_shuffle)
        logging.info(f'''train_dataloader size: {len(train_loader.dataset)} | shuffle: {train_shuffle}
                         eval_dataloader size: {len(eval_loader.dataset)} | shuffle: {eval_shuffle}''')
        return train_loader, eval_loader
    def train(self,
              epochs,
              train_dataloader,
              eval_dataloader,
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
                self.model.load_state_dict(torch.load(f'{ckpt_dir}/electra_model_state_dict.pt', map_location=self.device))
                optimizer.load_state_dict(torch.load(f'{ckpt_dir}/electra_optimizer_state_dict.pt'))
            except Exception as e:
                logging.info(f'{datetime.now()} | No checkpoint was found | {e}')

        self.model.train()

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
                lm_logit, loss = self.model(inputs,labels)

                loss.backward()

                step_loss += loss.item()
                losses[global_steps] = loss.item()
                local_steps += 1
                global_steps += 1

                if global_steps % gradient_accumulation_steps == 0:
                    optimizer.step()
                    self.model.zero_grad()

                if global_steps % log_steps == 0:
                    pb.set_postfix_str(f'''{datetime.now()} | Train Loss: {step_loss / local_steps} | Steps: {global_steps}''')
                    with open(f'{self.log_dir}/electra_train_results.json', 'w') as results_file:
                        json.dump(losses, results_file)
                        results_file.close()
                    step_loss = 0.0
                    local_steps = 0

                if global_steps % ckpt_steps == 0:
                    # evaluating before every checkpoint
                    # self.evaluate(eval_dataloader)
                    # self.model.train()
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    torch.save(model_to_save.state_dict(), f'{ckpt_dir}/electra_model_state_dict.pt')
                    torch.save(optimizer.state_dict(), f'{ckpt_dir}/electra_optimizer_state_dict.pt')

                    logging.info(f'{datetime.now()} | Saved checkpoint to: {ckpt_dir}')
            # Evaluate every epoch
            self.evaluate(eval_dataloader)
            self.model.train()

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), f'{ckpt_dir}/electra_model_state_dict.pt')
        torch.save(optimizer.state_dict(), f'{ckpt_dir}/electra_optimizer_state_dict.pt')

        return self.model

    def evaluate(self, dataloader):
        loss_fn = nn.CrossEntropyLoss()

        if self.n_gpu > 1 and not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)

        self.model.eval()

        eval_loss = 0.0
        perplexity = 0.0
        eval_steps = 0

        logging.info(f'{datetime.now()} | Evaluating...')
        for step, batch in tqdm(enumerate(dataloader),
                                desc='Evaluating',
                                leave=True,
                                total=len(dataloader),
                                bar_format='{l_bar}{bar:10}{r_bar}'):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                lm_logit, loss = self.model(inputs, labels)

            tmp_eval_loss = loss
            tmp_perplexity = torch.exp(tmp_eval_loss)

            if self.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
            perplexity += tmp_perplexity.item()
            eval_steps += 1

            total_eval_loss = eval_loss/eval_steps
            total_perplexity= perplexity/eval_steps

            logging.info(f'{datetime.now()} | Step: {step} | Eval Loss: {total_eval_loss} | Perplexity: {total_perplexity}')
            with open(f'{self.log_dir}/electra_eval_results.txt', 'a+') as results_file:
                results_file.write(f'{datetime.now()} | Step: {step} | Eval Loss: {total_eval_loss} | Perplexity: {total_perplexity}\n')
                results_file.close()

        return None

if __name__ == '__main__':
    config = ElectraConfig().get_config()

    tokenizer = BertTokenizer(vocab_file=wordpiece_vocab_path, do_lower_case=False)

    dataset = ElectraDataset(tokenizer, max_len, dir_path=dir_path)
    # Electra Model
    # (1) instantiate the generator and discriminator, making sure that the generator is roughly a quarter to a half of the size of the discriminator

    generator = ReformerLM(
        num_tokens=tokenizer.vocab_size,
        emb_dim=128,
        dim=256,  # smaller hidden dimension
        heads=4,  # less heads
        ff_mult=2,  # smaller feed forward intermediate dimension
        dim_head=64,
        depth=12,
        max_seq_len=1024
    )

    discriminator = ReformerLM(
        num_tokens=20000,
        emb_dim=128,
        dim=1024,
        dim_head=64,
        heads=16,
        depth=12,
        ff_mult=4,
        max_seq_len=1024,
        return_embeddings=True
    )
    # (2) weight tie the token and positional embeddings of generator and discriminator

    generator.token_emb = discriminator.token_emb
    generator.pos_emb = discriminator.pos_emb
    # weight tie any other embeddings if available, token type embeddings, etc.

    # (3) instantiate electra

    discriminator_with_adapter = nn.Sequential(discriminator, nn.Linear(1024, 1))

    model = Electra(
        generator,
        discriminator_with_adapter,
        mask_token_id = 2,          # the token id reserved for masking
        pad_token_id = 0,           # the token id for padding
        mask_prob = 0.15,           # masking probability for masked language modeling
        mask_ignore_token_ids = [3]  # ids of tokens to ignore for mask modeling ex. (cls, sep)
    )
    trainer = ElectraTrainer(dataset, model, tokenizer,max_len, train_batch_size=batch_size, eval_batch_size=batch_size)
    train_dataloader, eval_dataloader = trainer.build_dataloaders(train_test_split=0.1)

    model = trainer.train(epochs=epochs,
                          train_dataloader=train_dataloader,
                          eval_dataloader=eval_dataloader,
                          log_steps=log_steps,
                          ckpt_steps=ckpt_steps,
                          ckpt_dir= checkpoint_dir,
                          gradient_accumulation_steps=gradient_accumulation_steps)

    torch.save(model, checkpoint_path)

