import warnings
warnings.filterwarnings("ignore")

import os
import re
import kss
import json
import logging
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer


class DatasetForMLM(Dataset):
    def __init__(self, tokenizer, max_len, path="../data/namuwiki.txt"):
        logging.info('start wiki data load')

        self.tokenizer = tokenizer
        self.max_len =max_len
        self.docs = []
        doc = ""
        # 파일 리스트
        file_list = os.listdir(path)

        # num_lines = sum(1 for line in open(path, 'r',encoding='utf-8'))
        file_progress_bar = tqdm(file_list, position=0, leave=True, bar_format='{l_bar}{bar:10}{r_bar}')
        for file_name in file_progress_bar:
            path = f'{path}/{file_name}'
            data_file = open(path, 'r', encoding='utf-8')
            for line in tqdm(data_file,
                             desc='Data load for pretraining',
                             position=1, leave=True):
                line = line[:-1]
                self.docs.append(line)
        logging.info('complete data load')

    def mask_tokens(self, inputs: torch.Tensor, mlm_probability=0.15, pad=True):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = inputs.clone()
        # mlm_probability defaults to 0.15 in Bert
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        if pad:
            input_pads = self.max_len - inputs.shape[-1]
            label_pads = self.max_len - labels.shape[-1]

            inputs = F.pad(inputs, pad=(0, input_pads), value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, pad=(0, label_pads), value=self.tokenizer.pad_token_id)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def _tokenize_input_ids(self, input_ids: list, pad_to_max_length: bool = True):
        inputs = torch.tensor(self.tokenizer.encode(input_ids, add_special_tokens=False, max_length=self.max_len, pad_to_max_length=pad_to_max_length, return_tensors='pt',truncation=True))
        return inputs
    def __len__(self):
        """ Returns the number of documents. """
        return len(self.docs)
    def __getitem__(self, idx):
        inputs = self._tokenize_input_ids(self.docs[idx], pad_to_max_length=True)
        inputs, labels = self.mask_tokens(inputs,pad=True)

        inputs= inputs.squeeze()
        inputs_mask = inputs != 0
        labels= labels.squeeze()

        return inputs, inputs_mask, labels

class DatasetForAutoRegressive(Dataset):
    def __init__(self, tokenizer, max_len, dir_path):
        logging.info('Start pretraining data load!')

        self.tokenizer = tokenizer
        self.max_len =max_len
        self.docs = []

        # 파일 리스트
        file_list = os.listdir(dir_path)

        # num_lines = sum(1 for line in open(path, 'r',encoding='utf-8'))
        file_progress_bar = tqdm(file_list, position=0, leave=True, bar_format='{l_bar}{bar:10}{r_bar}')
        for file_name in file_progress_bar:
            path = f'{dir_path}/{file_name}'
            data_file =  open(path, 'r',encoding='utf-8')
            for line in tqdm(data_file,
                             desc='Data load for pretraining',
                             position=1, leave=True):
                line = line[:-1]
                self.docs.append(line)
        logging.info('Complete data load')

    def _tokenize_input_ids(self, input_ids: list, add_special_tokens:bool = False, pad_to_max_length: bool = True):
        inputs = torch.tensor(self.tokenizer.encode(input_ids, add_special_tokens=add_special_tokens, max_length=self.max_len, pad_to_max_length=pad_to_max_length, return_tensors='pt',truncation=True))
        return inputs
    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        inputs = self._tokenize_input_ids(self.docs[idx], pad_to_max_length=True)
        labels = inputs.clone()

        inputs= inputs.squeeze()
        labels= labels.squeeze()
        inputs_mask = inputs != 0


        return inputs, labels, inputs_mask

"""
문장 분리
"""
def kss_sentence_seperator(file, text):
    for sent in kss.split_sentences(text):
        file.write(sent.replace('\n', '')+"\n")
if __name__ == '__main__':
    wordpiece_vocab_path = "../data/vocab.txt"
    wiki_data_path = "../data/kowiki_origin.txt"
    # processed_wiki_data_path = "../data/processed_kowiki.txt"
    processed_wiki_data_path2 = "../data/kowiki.txt"
    test_path ="../data/mini_namuwiki.txt"


    tokenizer = BertTokenizer(vocab_file=wordpiece_vocab_path, do_lower_case=False)
    # dataset =NamuWikiDataset(tokenizer,512)
    # print(dataset)
    # make_data_upto_maxlen(tokenizer,512)
    #
    # f = open(processed_wiki_data_path,'r',encoding='utf-8')
    # f2 = open(processed_wiki_data_path2,'w',encoding="utf-8")
    #
    # while True:
    #     line = f.readline()
    #     if not line: break
    #     kss_sentence_seperator(f2,line)
    #
    # f.close()
    # f2.close()
    # make_data_upto_maxlen(tokenizer, 512,path=processed_wiki_data_path2)
    dataset = WikiDatasetForAutoRegressive(tokenizer,512,test_path)
    for data in dataset:
        print(data)

