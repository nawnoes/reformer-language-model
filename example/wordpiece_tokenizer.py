from transformers.tokenization_bert import BertTokenizer

vocab_path = "../data/wordpiece_pretty.txt"
vocab_path2 = "../data/vocab.txt"

tokenizer = BertTokenizer(vocab_file=vocab_path2, do_lower_case=False)

"""  """
print(tokenizer.decode(tokenizer.encode('나는 워드피스 토크나이저를 써요. 예대마진이 요즘 많이 나오네요',add_special_tokens=False)))
print(tokenizer.decode(tokenizer.encode('[코로나 범정부 대응- 文 ‘서울시 방역’ 긴급 점검·3개 부처 대국민담화]')))