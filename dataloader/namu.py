import os
import re
import json
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

        data_file = open(path, 'r')
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

if __name__ == '__main__':
    wordpiece_vocab_path = "../data/vocab.txt"

    tokenizer = BertTokenizer(vocab_file=wordpiece_vocab_path, do_lower_case=False)
    dataset =NamuWikiDataset(tokenizer,512)
    print(dataset)