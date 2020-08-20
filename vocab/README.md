# Wordpiece vocab 생성하기 
한국어 tokenizer에는 mecab과 기존 tokenizer외에 sentencepiece, wordpiece 등이 사용된다.
sentencepiece의 경우 기존에 방법으로 쉽게 vocab을 만들수 있다.  
  
BERT와 ELECTRA의 경우 wordpiece를 사용하고 있기 때문에, `huggingface tokenizer`를 이용해 wordpiece 토크나이저를
만들어본다.  `huggingface tokenizer`의 경우 코퍼스가 커도 메모리 이슈가 발생하지 않고, rust로 구현 되어 있어 속도가 매우 빠르다고 한다.

## BPE와 Wordpiece와의 차이
wordpiece는 BPE의 변형된 알고리즘이다. BPE가 subword를 만들 때, 많이 등장한 빈도수에 기반하여 병합한다. 하지만 wordpiece의 경우 코퍼스의 `Likelihood`를 가장 높이는 쌍으로 병합한다


### Wordpiece 만들기 
> `lowercase=False`인 경우 `strip_accent=False`로 해줘야 한다.
> `[UNK]`의 비중을 최대한 줄이기 위해 모든 character를 커버할 수 있도록 **limit_alphabet** 설정
#### huggingface tokenizers 설치
```sh
pip install tokenizers
```

#### wordpiece vocab 만들기
```python
import argparse
from tokenizers import BertWordPieceTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_file", type=str)
parser.add_argument("--vocab_size", type=int, default=32000)
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

tokenizer.save("./", "ch-{}-wpm-{}".format(args.limit_alphabet, args.vocab_size))
```
# References
- https://monologg.kr/2020/04/27/wordpiece-vocab/
= 