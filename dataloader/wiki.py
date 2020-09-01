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


class EnWikiDataset(Dataset):

    def __init__(self, path="", prefix="train"):

        assert os.path.isdir(path)

        self.documents = []
        filename_list = os.listdir(path)
        for file in filename_list:
            path_to_file = os.path.join(path, file)
            if not os.path.isfile(path_to_file):
                continue
            self.documents.append(path_to_file)

    def __len__(self):
        """ Returns the number of documents. """
        return len(self.documents)

    def __getitem__(self, idx):
        document_path = self.documents[idx]
        document_name = document_path.split("/")[-1]

        items = []

        with open(document_path, encoding="utf-8") as source:
            raw_text = source.readlines()
            for obj in raw_text:
                text = json.loads(obj)['text']
                text = re.sub('\\n', ' ', text)
                text = re.sub('\\s+', ' ', text)
                items.append(text)

        return items

class NamuWikiDataset(Dataset):
    def __init__(self, tokenizer, max_len, path="../data/namuwiki.txt"):
        print('namu wiki data load start')

        data_file = open(path, 'r', encoding="utf-8")
        self.docs = []
        doc = ""
        while True:
            line = data_file.readline()
            if not line: break

            line = line[:-1]
            if len(tokenizer.encode(doc)) <max_len and len(tokenizer.encode(doc + line))<max_len:
                doc += line
            elif len(tokenizer.encode(doc + line))>= max_len and len(tokenizer.encode(doc))<max_len:
                self.docs.append(doc)
                # print(f"max_len-{max_len} real_len-{len(tokenizer.encode(doc))} doc-{doc}\n\n")
                doc = line
        print('namu wiki data load complete')
    def __len__(self):
        """ Returns the number of documents. """
        return len(self.docs)
    def __getitem__(self, idx):
        return self.docs[idx]

class NamuWikiDatasetForMLM(Dataset):
    def __init__(self, tokenizer, max_len, path="../data/namuwiki.txt"):
        logging.info('start wiki data load')

        data_file = open(path, 'r')
        self.tokenizer = tokenizer
        self.max_len =max_len
        self.docs = []
        doc = ""

        num_lines = sum(1 for line in open(path, 'r',encoding='utf-8'))
        print('data line numbers:',num_lines)
        data_file =  open(path, 'r',encoding='utf-8')

        for line in tqdm(data_file,
                         desc='namuwiki data loader',
                         total=num_lines):
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
        inputs = torch.tensor(self.tokenizer.encode(input_ids, add_special_tokens=True, max_length=self.max_len, pad_to_max_length=pad_to_max_length, return_tensors='pt',truncation=True))
        return inputs
    def __len__(self):
        """ Returns the number of documents. """
        return len(self.docs)
    def __getitem__(self, idx):
        inputs = self._tokenize_input_ids(self.docs[idx], pad_to_max_length=True)
        inputs, labels = self.mask_tokens(inputs)

        inputs= inputs.squeeze()
        labels= labels.squeeze()

        return inputs, labels

class WikiDatasetForAutoRegressive(Dataset):
    def __init__(self, tokenizer, max_len, path="../data/namuwiki.txt"):
        logging.info('start wiki data load')

        self.tokenizer = tokenizer
        self.max_len =max_len
        self.docs = []

        num_lines = sum(1 for line in open(path, 'r',encoding='utf-8'))
        data_file =  open(path, 'r',encoding='utf-8')

        for line in tqdm(data_file,
                         desc='namuwiki data loader',
                         total=num_lines):
            line = line[:-1]
            self.docs.append(line)
        logging.info('complete data load')

    def _tokenize_input_ids(self, input_ids: list, pad_to_max_length: bool = True):
        inputs = torch.tensor(self.tokenizer.encode(input_ids, add_special_tokens=True, max_length=self.max_len, pad_to_max_length=pad_to_max_length, return_tensors='pt',truncation=True))
        return inputs
    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        inputs = self._tokenize_input_ids(self.docs[idx], pad_to_max_length=True)
        labels = inputs.clone()

        inputs= inputs.squeeze()
        labels= labels.squeeze()

        return inputs, labels
def make_data_upto_maxlen( tokenizer, max_len, path="../data/namuwiki.txt"):
    split_data = path.split('.')
    split_data[-2]+= f'-{max_len}'
    return_file_path = '.'.join(split_data)
    logging.info('file name:'+return_file_path)

    return_file= open(return_file_path,'w',encoding='utf-8')
    docs = []
    doc = ""
    doc_len = 0

    num_lines = sum(1 for line in open(path, 'r',encoding='utf-8'))
    logging.info('file line number: '+str(num_lines))
    data_file = open(path, 'r')

    for line in tqdm(data_file,
                     desc='namuwiki data maker',
                     total=num_lines):
        line = line[:-1]
        line_len = len(tokenizer.encode(line))
        added_doc_len = doc_len +line_len
        if line =="":
            return_file.write(doc + "\n")
            doc = ""
            doc_len = 0
        elif  doc_len <max_len and added_doc_len<max_len:
            doc += line
            doc_len += line_len
        elif added_doc_len>= max_len and doc_len<max_len:
            return_file.write(doc+"\n")
            # print(f"max_len-{max_len} real_len-{len(tokenizer.encode(doc))} doc-{doc}\n\n")
            doc = line
            doc_len = line_len
    return_file.close()
    data_file.close()

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

