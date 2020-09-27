import random
from random import shuffle
import os
import csv
from nltk.corpus import wordnet
from tqdm import tqdm, trange
random.seed(1)

#stop words list
stop_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', ''
]

#cleaning up text
import re


def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")  #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(
        set([word for word in words if word not in stop_words and word.islower() ]))
    random.shuffle(random_word_list)
    num_replaced = 0


    for random_word in random_word_list:
        if n == 0:
            break
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [
                synonym if word == random_word else word for word in new_words
            ]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n:  #only replace up to n words
            break

    #this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([
                char for char in synonym
                if char in ' qwertyuiopasdfghjklzxcvbnm'
            ])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--replace_list",
    default="1,3,4,5,6",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)

parser.add_argument(
    "--dir",
    default="/net/nfs.websail/yyv959/arc/arc-hard/test/",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)

args = parser.parse_args()
alpha = 0.1
dir = args.dir

output_dir = dir + "syn_replaced/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

replace_list = args.replace_list.strip().split(',')
replace_list = [int(it) for it in replace_list]

with open(dir + "dev.csv", 'r',
          encoding='utf-8') as f, open(output_dir + "dev.csv",
                                       'w',
                                       encoding='utf8',
                                       newline='') as out:

    tsv_writer = csv.writer(out, delimiter=',', lineterminator='\n')
    reader = csv.reader(f)
    line = next(reader)
    tsv_writer.writerow(line)
    for line in tqdm(reader):
        for i in replace_list:
            word_list = line[i].split()
            length = len(word_list)
            n = round(alpha * length)
            print(n)
            new_words = synonym_replacement(word_list, n)
            line[i] = " ".join(new_words)
        tsv_writer.writerow(line)
