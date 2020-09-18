import os
import argparse
from tokenizers import BertWordPieceTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_file", type=str, default="../data/namuwiki.txt")
parser.add_argument("--vocab_size", type=int, default=22000)
parser.add_argument("--limit_alphabet", type=int, default=6000)

args = parser.parse_args()

# 파일 경로
dir_path = "/Volumes/My Passport for Mac/00_nlp/PretrainingData/raw"

# 파일 경로 내 코퍼스 목록
file_list = os.listdir(dir_path)

# 코퍼스 목
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
    wordpieces_prefix="##"
)

tokenizer.train(
    files=corpus_files,
    limit_alphabet=args.limit_alphabet,
    vocab_size=args.vocab_size
)

tokenizer.save("./ch-{}-wpm-{}-pretty-all".format(args.limit_alphabet, args.vocab_size),True)