import warnings
warnings.filterwarnings("ignore")

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

class ReformerTrainer(object):
    def __init__(self,
                 dataset,
                 model,
                 tokenizer,
                 max_len,
                 device=None,
                 train_batch_size=8,
                 eval_batch_size=None,
                 tb_writer=False,
                 tb_dir='./tb_logs',
                 log_dir='../logs'):
        """
        Provides an easy to use class for pretraining and evaluating a Reformer Model.
        :param dataset: (torch.utils.data.Dataset) containing all of the data you wish to utilize during training.
        :param model: (reformer_pytorch.Reformer)
        :param tokenizer: (transformers.PreTrainedTokenizer) defaults to BertTokenizer ('bert-base-case')
        :param device: provide manual device placement. If None, will default to cuda:0 if available.
        :param tb_writer: (bool) Whether to write to tensorboard or not.
        :param tb_dir: (str) Where to write TB logs to.
        :param log_dir: (str) Where to write generic logs to.
        """

        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
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
        """
        Trains the Reformer Model
        :param epochs: The number of times you wish to loop through the dataset.
        :param train_dataloader: (torch.utils.data.DataLoader) The data to train on.
        :param eval_dataloader: (torch.utils.data.DataLoader) The data to evaluate on.
        :param log_steps: The number of steps to iterate before logging.
        :param ckpt_steps: The number of steps to iterate before checkpointing.
        :param ckpt_dir: The directory to save the checkpoints to.
        :param gradient_accumulation_steps: Optional gradient accumulation.
        :return: Total number of steps, total loss, model
        """

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
                self.model.load_state_dict(torch.load(f'{ckpt_dir}/model_state_dict.pt', map_location=self.device))
                optimizer.load_state_dict(torch.load(f'{ckpt_dir}/optimizer_state_dict.pt'))

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
            pb = tqdm(enumerate(train_dataloader), desc=f'Epoch-{epoch} Iterator', total=len(train_dataloader))
            for step, batch in pb:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                output = self.model(inputs)

                # only calculating loss on masked tokens
                loss_mx = labels != -100
                output = output[loss_mx].view(-1, self.tokenizer.vocab_size)
                labels = labels[loss_mx].view(-1)

                loss = loss_fn(output, labels)

                if gradient_accumulation_steps > 1:
                    loss /= gradient_accumulation_steps

                try:
                    loss.backward()
                except Exception as err:
                    print(err)
                    continue

                step_loss += loss.item()
                losses[global_steps] = loss.item()
                local_steps += 1
                global_steps += 1

                if global_steps % gradient_accumulation_steps == 0:
                    optimizer.step()
                    self.model.zero_grad()

                if global_steps % log_steps == 0:
                    if self.tb_writer:
                        self.writer.add_scalar('Train/Loss', step_loss / local_steps, global_steps)
                        self.writer.close()
                    pb.set_postfix_str(f'''{datetime.now()} | Train Loss: {step_loss / local_steps} | Steps: {global_steps}''')
                    with open(f'{self.log_dir}/train_results.json', 'w') as results_file:
                        json.dump(losses, results_file)
                        results_file.close()
                    step_loss = 0.0
                    local_steps = 0

                if global_steps % ckpt_steps == 0:
                    # evaluating before every checkpoint
                    self.evaluate(eval_dataloader)
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    torch.save(model_to_save.state_dict(), f'{ckpt_dir}/model_state_dict.pt')
                    torch.save(optimizer.state_dict(), f'{ckpt_dir}/optimizer_state_dict.pt')

                    logging.info(f'{datetime.now()} | Saved checkpoint to: {ckpt_dir}')

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), f'{ckpt_dir}/model_state_dict.pt')
        torch.save(optimizer.state_dict(), f'{ckpt_dir}/optimizer_state_dict.pt')

        return self.model

    def evaluate(self, dataloader):
        """
        Runs through the provided dataloader with torch.no_grad()
        :param dataloader: (torch.utils.data.DataLoader) Evaluation DataLoader
        :return: None
        """
        loss_fn = nn.CrossEntropyLoss()

        if self.n_gpu > 1 and not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)

        self.model.eval()
        eval_loss = 0.0
        perplexity = 0.0
        eval_steps = 0

        logging.info(f'{datetime.now()} | Evaluating...')
        for step, batch in tqdm(enumerate(dataloader), desc='Evaluating', leave=True, total=len(dataloader)):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                output = self.model(inputs)

            loss_mx = labels != -100
            output_ids = output[loss_mx].view(-1, self.tokenizer.vocab_size)
            labels = labels[loss_mx].view(-1)
            tmp_eval_loss = loss_fn(output_ids, labels)
            tmp_perplexity = torch.exp(tmp_eval_loss)

            if self.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
            perplexity += tmp_perplexity.item()
            eval_steps += 1

            eval_loss /= eval_steps
            perplexity /= eval_steps

            if self.tb_writer:
                self.writer.add_scalar('Eval/Loss', eval_loss, eval_steps)
                self.writer.close()
                self.writer.add_scalar('Perplexity', perplexity, eval_steps)
                self.writer.close()
            logging.info(f'{datetime.now()} | Step: {step} | Eval Loss: {eval_loss} | Perplexity: {perplexity}')
            with open(f'{self.log_dir}/eval_results.txt', 'a+') as results_file:
                results_file.write(f'{datetime.now()} | Step: {step} | Eval Loss: {eval_loss} | Perplexity: {perplexity}\n')
                results_file.close()

        return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab_path", default="../data/vocab.txt", required=True)
    parser.add_argument("--data_path", default="../data/mini_namuwiki.txt", required=True)
    parser.add_argument("--config_path", default="../data/mini_namuwiki.txt", required=True)


    args = parser.parse_args()

    # args.model_name, args.data_dir, **hparams))
    wordpiece_vocab_path = args.vocab_path
    mini_data_path = args.data_path
    max_len = 256
    batch_size = 4

    tokenizer = BertTokenizer(vocab_file=wordpiece_vocab_path, do_lower_case=False)

    # dataset = NamuWikiDataset(tokenizer, max_len, path=mini_data_path)
    dataset = NamuWikiDatasetForMLM(tokenizer, max_len, path=mini_data_path)

    model = ReformerLM(
        num_tokens=tokenizer.vocab_size,
        dim=512,
        depth=6,
        heads=8,
        max_seq_len=max_len,
        causal=True  # auto-regressive 학습을 위한 설정
    )
    trainer = ReformerTrainer(dataset, model, tokenizer, max_len, train_batch_size=batch_size,
                              eval_batch_size=batch_size)
    train_dataloader, eval_dataloader = trainer.build_dataloaders(train_test_split=0.1)
    model = trainer.train(epochs=30,
                          train_dataloader=train_dataloader,
                          eval_dataloader=eval_dataloader,
                          log_steps=10,
                          ckpt_steps=100,
                          ckpt_dir='../checkpoints',
                          gradient_accumulation_steps=1)

    torch.save(model, '../checkpoints/model.bin')



if __name__ == '__main__':
    wordpiece_vocab_path = "../data/vocab.txt"
    mini_data_path ="../data/mini_namuwiki.txt"
    data_path ="../data/namuwiki.txt"

    checkpoint_dir = "../checkpoints"
    checkpoint_path = f'{checkpoint_dir}/reformer.bin'

    # Model Hyperparameter
    max_len = 256
    batch_size = 4
    dim = 512
    depth = 6
    heads = 8
    causal = True

    # Train Hyperparameter
    epochs = 30,
    log_steps = 10,
    ckpt_steps = 100,
    ckpt_dir = checkpoint_path,
    gradient_accumulation_steps = 1

    tokenizer = BertTokenizer(vocab_file=wordpiece_vocab_path, do_lower_case=False)


    # dataset = NamuWikiDataset(tokenizer, max_len, path=mini_data_path)
    dataset = NamuWikiDatasetForMLM(tokenizer, max_len, path=data_path)

    model = ReformerLM(
        num_tokens=tokenizer.vocab_size,
        dim=dim,
        depth=depth,
        heads=heads,
        max_seq_len=max_len,
        causal=causal # auto-regressive 학습을 위한 설정
    )
    trainer = ReformerTrainer(dataset, model, tokenizer,max_len, train_batch_size=batch_size, eval_batch_size=batch_size)
    train_dataloader, eval_dataloader = trainer.build_dataloaders(train_test_split=0.1)

    model = trainer.train(epochs=epochs,
                          train_dataloader=train_dataloader,
                          eval_dataloader=eval_dataloader,
                          log_steps=log_steps,
                          ckpt_steps=ckpt_steps,
                          ckpt_dir= checkpoint_dir,
                          gradient_accumulation_steps=gradient_accumulation_steps)

    torch.save(model, checkpoint_path)