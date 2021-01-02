## 📘 Vocab & Tokenizer
`Sentencepiece`와 `Wordpiece` 중 기존에 사용해보지 않은 Wordpiece Tokenizer를 이용한 Tokenizer 사용. 로
센텐스피스와 워드피스 모두 Subword Tokenizer중 하나인 **BPE(Byte Pair Encoding)** 에 속한다. 센텐스피스의 경우 빈도수를 기반으로, 워드피스의 경우 likelihood를 기반으로 BPE을 수행한다. 
#### WordPiece Tokenizer
1. huggingface tokenizers 설치
```shell script
pip install tokenizers
```
2. 말뭉치 파일로 Wordpiece vocab 생성
```python
import argparse
from tokenizers import BertWordPieceTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_file", type=str)
parser.add_argument("--vocab_size", type=int, default=22000) # 만들 Vocab의 숫자 
parser.add_argument("--limit_alphabet", type=int, default=6000)

args = parser.parse_args()

tokenizer = BertWordPieceTokenizer(
    vocab_file=None,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False, # Must be False if cased model
    lowercase=False,
    wordpieces_prefix="##"
)

tokenizer.train(
    files=[args.corpus_file],
    limit_alphabet=args.limit_alphabet,
    vocab_size=args.vocab_size
)

tokenizer.save("./ch-{}-wpm-{}-pretty".format(args.limit_alphabet, args.vocab_size),True)
```
3. 생성된 vocab 파일 전처리
```python
import json # import json module

vocab_path = "../vocab/ch-6000-wpm-22000-pretty"

vocab_file = '../data/wpm-vocab-all.txt'
f = open(vocab_file,'w',encoding='utf-8')
with open(vocab_path) as json_file:
    json_data = json.load(json_file)
    for item in json_data["model"]["vocab"].keys():
        f.write(item+'\n')

    f.close()
```
4. Tokenizer 테스트
```python
from transformers.tokenization_bert import BertTokenizer

vocab_path = "../data/wpm-vocab-all.txt"

tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)

test_str = ' [CLS] 나는 워드피스 토크나이저를 써요. 성능이 좋은지 테스트 해보려 합니다. [SEP]'
print('테스트 문장: ',test_str)

encoded_str = tokenizer.encode(test_str,add_special_tokens=False)
print('문장 인코딩: ',encoded_str)

decoded_str = tokenizer.decode(encoded_str)
print('문장 디코딩: ',decoded_str)

"""
테스트 문장:   [CLS] 나는 워드피스 토크나이저를 써요. 성능이 좋은지 테스트 해보려 합니다. [SEP]
문장 인코딩:  [2, 9310, 4868, 6071, 12467, 21732, 12200, 6126, 6014, 4689, 6100, 18, 11612, 6037, 9389, 6073, 16784, 17316, 6070, 10316, 18, 3]
문장 디코딩:  [CLS] 나는 워드피스 토크나이저를 써요. 성능이 좋은지 테스트 해보려 합니다. [SEP]
"""
```