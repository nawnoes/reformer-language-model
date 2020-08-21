import json # import json module

vocab_path = "../data/wordpiece_pretty.txt"
vocab_path2 = "../vocab/wordpiece.txt"

vocab_file = '../data/vocab.txt'
f = open(vocab_file,'w',encoding='utf-8')
with open(vocab_path) as json_file:
    json_data = json.load(json_file)
    for item in json_data["model"]["vocab"].keys():
        f.write(item+'\n')

    f.close()