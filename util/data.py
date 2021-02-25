import warnings
warnings.filterwarnings("ignore")

import os
import re
import kss
import json
import logging
import torch
from tqdm import tqdm
from transformers import BertTokenizer

def make_data_under_maxlen( tokenizer, max_len, dir_path, file_name, empty_line = True):
    file_name = file_name.split('.')
    path = f'{dir_path}/{file_name[0]}.{file_name[1]}'
    return_file_path = f'{dir_path}/data/{file_name[0]}-{max_len}.{file_name[1]}'
    # logging.info('file name:'+return_file_path)

    return_file= open(return_file_path,'w',encoding='utf-8')
    docs = []
    doc = "[CLS] "
    doc_len = 1

    num_lines = sum(1 for line in open(path, 'r',encoding='utf-8'))
    # logging.info('file line number: '+str(num_lines))
    data_file = open(path, 'r')

    for line in tqdm(data_file,
                     desc='Data Maker',
                     total=num_lines):
        line = line[:-1]
        line_len = len(tokenizer.encode(line,add_special_tokens=False,padding=False,max_length=max_len-2,truncation=True))
        added_doc_len = doc_len + line_len +1
        if empty_line and line =="":
            return_file.write(doc + "\n")
            doc = "[CLS] "
            doc_len = 1
        elif  doc_len <max_len+1 and added_doc_len<max_len+1:
            doc += line+" [SEP] "
            doc_len += line_len+1
        elif added_doc_len>= max_len+1 and doc_len<max_len+1:
            return_file.write(doc+"\n")
            # print(f"max_len-{max_len} real_len-{len(tokenizer.encode(doc))} doc-{doc}\n\n")
            doc = "[CLS] "+line
            doc_len = line_len+1
        # elif doc_len>=max_len+1:

    return_file.close()
    data_file.close()

if __name__ == '__main__':
    # vocab 경로
    wordpiece_vocab_path = "../data/vocab-v2.txt"

    # 데이터 경로
    dir_path = "/Volumes/My Passport for Mac/00_nlp/wiki"
    # dir_path = "../data/novel"


    # 토크나이즈
    tokenizer = BertTokenizer(vocab_file=wordpiece_vocab_path, do_lower_case=False)

    # 파일 리스트
    file_list = os.listdir(dir_path)

    # 목록 내에 json 파일 읽기
    progress_bar = tqdm(file_list, position=1)
    for file_name in progress_bar:
        if ".txt"in file_name :
            progress_bar.set_description(f'file name: {file_name}')
            full_file_path = f'{dir_path}/{file_name}'
            make_data_under_maxlen(tokenizer, 1024,dir_path,file_name,empty_line=False)