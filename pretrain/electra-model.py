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
from dataset.electra import ElectraDataset
from electra_pytorch import Electra
from reformer_pytorch import ReformerLM
from util.arg import ElectraConfig

class ElectraTrainer(object):
    def __init__(self,
                 dataset,
                 model,
                 tokenizer,
                 max_len,
                 model_name,
                 checkpoint_path,
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
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tb_writer = tb_writer
        self.log_dir = log_dir

        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if eval_batch_size is None:
            self.eval_batch_size = train_batch_size

        if tb_writer:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=tb_dir)

        logging.basicConfig(filename=f'{log_dir}/{self.model_name}-{datetime.now().date()}.log', level=logging.INFO)

    def build_dataloaders(self, train_test_split=0.1, train_shuffle=True, eval_shuffle=True):
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
              gradient_accumulation_steps=1):

        optimizer = Adafactor(self.model.parameters())
        losses = {}
        global_steps = 0
        local_steps = 0
        step_loss = 0.0
        start_epoch = 0
        start_step = 0

        if os.path.isfile(f'{self.checkpoint_path}/{self.model_name}.pth'):
            checkpoint = torch.load(f'{self.checkpoint_path}/{self.model_name}.pth', map_location=self.device)
            start_epoch = checkpoint['epoch']
            losses = checkpoint['losses']
            global_steps = checkpoint['train_step']
            start_step = global_steps if start_epoch==0 else global_steps*self.train_batch_size % len(train_dataloader)

            self.model.load_state_dict(checkpoint['model_state_dict'],strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.model.train()

        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
            logging.info(f'{datetime.now()} | Utilizing {self.n_gpu} GPUs')

        self.model.to(self.device)
        logging.info(f'{datetime.now()} | Moved model to: {self.device}')
        logging.info(f'{datetime.now()} | train_batch_size: {self.train_batch_size} | eval_batch_size: {self.eval_batch_size}')
        logging.info(f'{datetime.now()} | Epochs: {epochs} | log_steps: {log_steps} | ckpt_steps: {ckpt_steps}')
        logging.info(f'{datetime.now()} | gradient_accumulation_steps: {gradient_accumulation_steps}')

        for epoch in range(start_epoch, epochs): #tqdm(range(epochs), desc='Epochs', position=0):
            logging.info(f'{datetime.now()} | Epoch: {epoch}')
            pb = tqdm(enumerate(train_dataloader),
                      desc=f'Epoch-{epoch} Iterator',
                      total=len(train_dataloader),
                      bar_format='{l_bar}{bar:10}{r_bar}'
                      )
            for step, batch in pb:
                # if step < start_step:
                #     continue
                input_data = batch
                input_data = input_data.to(self.device)
                output = self.model(input_data)

                loss = output.loss
                loss.backward()

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
                    with open(f'{self.log_dir}/{self.model_name}_train_results.json', 'w') as results_file:
                        json.dump(losses, results_file)
                        results_file.close()
                    step_loss = 0.0
                    local_steps = 0

                if global_steps % ckpt_steps == 0:
                    self.save(epoch, self.model, optimizer, losses, global_steps)
                    logging.info(f'{datetime.now()} | Saved checkpoint to: {self.checkpoint_path}')

            # Evaluate every epoch
            self.evaluate(eval_dataloader)
            self.model.train()
            start_step = 0

        self.save(epochs,self.model,optimizer,losses, global_steps)

        return self.model

    def evaluate(self, dataloader):
        if self.n_gpu > 1 and not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)

        self.model.eval()

        eval_loss = 0.0
        eval_steps = 0

        logging.info(f'{datetime.now()} | Evaluating...')
        for step, batch in tqdm(enumerate(dataloader),
                                desc='Evaluating',
                                leave=True,
                                total=len(dataloader),
                                bar_format='{l_bar}{bar:10}{r_bar}'):
            input_data = batch
            input_data = input_data.to(self.device)

            with torch.no_grad():
                output = self.model(input_data)

            tmp_eval_loss = output.loss

            if self.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
            eval_steps += 1

            total_eval_loss = eval_loss/eval_steps

            if self.tb_writer:
                self.writer.add_scalar('Eval/Loss', eval_loss, eval_steps)
                self.writer.close()
            logging.info(f'{datetime.now()} | Step: {step} | Eval Loss: {total_eval_loss}')
            with open(f'{self.log_dir}/{self.model_name}_eval_results.txt', 'a+') as results_file:
                results_file.write(f'{datetime.now()} | Step: {step} | Eval Loss: {total_eval_loss}\n')
                results_file.close()

        return None
    def save(self, epoch, model, optimizer, losses, train_step):
        torch.save({
            'epoch': epoch,  # 현재 학습 epoch
            'model_state_dict': model.state_dict(),  # 모델 저장
            'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
            'losses': losses,  # Loss 저장
            'train_step': train_step,  # 현재 진행한 학습
        }, f'{self.checkpoint_path}/{self.model_name}.pth')


def main():
    torch.manual_seed(9)
    # 1. Config
    train_config, gen_config, disc_config = ElectraConfig(config_path='../config/electra/electra-train.json').get_config()

    # 2. Tokenizer
    tokenizer = BertTokenizer(vocab_file=train_config.vocab_path, do_lower_case=False)

    # 3. Dataset
    dataset = ElectraDataset(tokenizer, train_config.max_len, data_path=train_config.data_path)

    # 4. Electra Model
    # 4.1. instantiate the generator and discriminator,
    # making sure that the generator is roughly a quarter to a half of the size of the discriminator
    # 제너레이터의 크기는 디스크리미네이터의 1/4~ 1/2 크기로
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

    model = Electra(
        generator,
        discriminator_with_adapter,
        mask_token_id = tokenizer.mask_token_id,           # the token id reserved for masking
        pad_token_id = tokenizer.pad_token_id,             # the token id for padding
        mask_prob = 0.15,                                  # masking probability for masked language modeling
        mask_ignore_token_ids = tokenizer.all_special_ids  # ids of tokens to ignore for mask modeling ex. (cls, sep)
    )


    trainer = ElectraTrainer(dataset, model, tokenizer, train_config.max_len,checkpoint_path=train_config.checkpoint_path, model_name=train_config.model_name, train_batch_size=train_config.batch_size, eval_batch_size=train_config.batch_size)
    train_dataloader, eval_dataloader = trainer.build_dataloaders(train_test_split=0.1)

    model = trainer.train(epochs=train_config.epochs,
                          train_dataloader=train_dataloader,
                          eval_dataloader=eval_dataloader,
                          log_steps=train_config.log_steps,
                          ckpt_steps=train_config.ckpt_steps,
                          gradient_accumulation_steps=train_config.gradient_accumulation_steps)

if __name__ == '__main__':
    main()
