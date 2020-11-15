import matplotlib.pyplot as plt
import json

def print_json_line_graph(json_path):
  f = open(json_path, 'r')
  json_data = json.load(f)
  lists = json_data.items()  # sorted by key, return a list of tuples
  # lists = list(filter(lambda x: int(x[0]) % 1000==0, lists))
  x, y = zip(*lists)  # unpack a list of pairs into two tuples

  plt.plot(x, y, 'r')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.title('MLM Eval Perplexity')
  plt.show()
if __name__=='__main__':
  print_json_line_graph('../logs/mlm_eval_perplexity.json')


"""
참고
https://bcho.tistory.com/1201
"""