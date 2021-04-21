import os
import json
import argparse
from tokenizers import BertWordPieceTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_file", type=str, default="/Users/a60058238/Desktop/Google Drive/nlp_data/plain_data")
parser.add_argument("--vocab_file", type=str, default='../data/vocab-v3.txt')
parser.add_argument("--vocab_size", type=int, default=22000)
parser.add_argument("--limit_alphabet", type=int, default=6000)
parser.add_argument("--wordpieces_prefix", type=str, default='##')


args = parser.parse_args()

dir_path =  args.corpus_file # 파일 경로
file_list = os.listdir(dir_path) # 파일 경로 내 코퍼스 목록
wordpiece_train_file = "./ch-{}-wpm-{}-wiki".format(args.limit_alphabet, args.vocab_size) # 워드피스 학습 파일
vocab_file = args.vocab_file # 생성할 vocab 파일

# 코퍼스 목록
corpus_files =[]
for file_name in file_list:
    if '.txt' in file_name: # txt 파일인 경우
        corpus_files.append(f'{dir_path}/{file_name}')

tokenizer = BertWordPieceTokenizer(
    vocab_file=None,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False, # Must be False if cased model
    lowercase=False,
    wordpieces_prefix=args.wordpieces_prefix)

tokenizer.train(
    files=corpus_files,
    limit_alphabet=args.limit_alphabet,
    vocab_size=args.vocab_size,
    wordpieces_prefix=args.wordpieces_prefix
)
tokenizer.save(wordpiece_train_file,True)

f = open(vocab_file,'w',encoding='utf-8')
with open(wordpiece_train_file) as json_file:
    json_data = json.load(json_file)
    for item in json_data["model"]["vocab"].keys():
        f.write(item+'\n')
    f.close()