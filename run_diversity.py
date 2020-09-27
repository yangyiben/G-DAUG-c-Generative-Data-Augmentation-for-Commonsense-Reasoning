import os
import csv
from tqdm import tqdm
from tqdm import trange
import re



import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir",
    default= "/net/nfs.websail/yyv959/winogrande/train_l/fake-medium-sym-influence-600000/",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)

parser.add_argument(
    "--output_dir",
    default= "/net/nfs.websail/yyv959/winogrande/train_l/fake-medium-sym-200000-unigram-influence/",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)

parser.add_argument(
    "--sample_size",
    default=132848 ,
    type=int,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)

parser.add_argument(
    "--num_choices",
    default=5,
    type=int,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)
args = parser.parse_args()
dir = args.dir
output_dir = args.output_dir

sample_size = args.sample_size
#sample_size = 10
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data = []
vocab = []


def tokenize(str):
    str = re.findall(r"[\w']+|[.,!?;]", str)

    return " ".join(str)


print("build vocab")
with open(dir + "train.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for i, line in tqdm(enumerate(reader)):
        if len(line) < (3 + args.num_choices):
            continue
        data.append(line)

        vocab_set = tokenize(line[1]).split(" ")
        for i in range(args.num_choices):
            vocab_set += tokenize(line[3 + i]).split(" ")
        vocab_set = set(vocab_set)
        vocab.append(vocab_set)
print(len(data))
selected = [False] * len(data)
selected_vocab = set([])
output_data = []
previous_vocab_increase = [None] * len(data)

for i in trange(sample_size):
    max_vocab_increase = -1
    max_idx = -1
    for j, (s,v,p_v_i) in enumerate( zip(selected,vocab,previous_vocab_increase) ):
        if s:
            continue
        if p_v_i is not None:
            if p_v_i <= max_vocab_increase:
                continue

        vocab_increase = len(v - selected_vocab)
        previous_vocab_increase[j] = vocab_increase

        if vocab_increase > max_vocab_increase:
            max_idx = j
            max_vocab_increase = vocab_increase
    if max_idx == -1:
        o = 2
        print("error")
    else:
        #print(max_vocab_increase)
        output_data.append(data[max_idx])
        selected_vocab.update(vocab[max_idx])
        selected[max_idx] = True


with open(output_dir + "train.csv", 'w', encoding='utf8',
          newline='') as tsv_file:
    tsv_writer = csv.writer(tsv_file, delimiter=',', lineterminator='\n')

    tsv_writer.writerow(header)
    for it in output_data:
        tsv_writer.writerow(it)
