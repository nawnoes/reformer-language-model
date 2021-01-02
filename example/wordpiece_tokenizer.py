from transformers.tokenization_bert import BertTokenizer

vocab_path = "../data/wordpiece_pretty.txt"
vocab_path2 = "../data/wpm-vocab-all.txt"
vocab_path3 = "../data/wiki-vocab.txt"

tokenizer = BertTokenizer(vocab_file=vocab_path3, do_lower_case=False)

"""  """
test_str = ' [CLS] 나는 워드피스 토크나이저를 써요. 하리보가 걸어다니네? 하리보보 하리보 [SEP]'
print('테스트 문장: ',test_str)
encoded_str = tokenizer.encode(test_str,add_special_tokens=False)
print('문장 인코딩: ',encoded_str)
decoded_str = tokenizer.decode(encoded_str)
print('문장 디코딩: ',decoded_str)

"""
겱과
테스트 문장:   [CLS] 나는 워드피스 토크나이저를 써요. 성능이 좋은지 테스트 해보려 합니다. [SEP]
문장 인코딩:  [2, 9310, 4868, 6071, 12467, 21732, 12200, 6126, 6014, 4689, 6100, 18, 11612, 6037, 9389, 6073, 16784, 17316, 6070, 10316, 18, 3]
문장 디코딩:  [CLS] 나는 워드피스 토크나이저를 써요. 성능이 좋은지 테스트 해보려 합니다. [SEP]
"""