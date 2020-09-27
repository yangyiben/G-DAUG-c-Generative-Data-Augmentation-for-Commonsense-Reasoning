from transformers import (WEIGHTS_NAME, GPT2Config, GPT2Tokenizer)
from generative_qg import GenerativeGPT2QG, GenerativeGPT2QGWrapper
from generative_qa import GenerativeGPT2QA2, GenerativeGPT2QD, GenerativeGPT2QA2Wrapper, GenerativeGPT2QDWrapper
import random
import sys
import csv

import numpy as np
import torch
from tqdm import tqdm, trange
from math import exp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    default="train_fake_1000000",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)
parser.add_argument(
    "--epochs",
    default=3125,
    type=int,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)
args = parser.parse_args()

dir = "/scratch/yyv959/commonsenseqa/"
qg_model_path = "/scratch/yyv959/commonsenseqa/outputs/gpt2-large/qg/"

qg_model = GenerativeGPT2QG.from_pretrained(qg_model_path)

qg_tokenizer = GPT2Tokenizer.from_pretrained(qg_model_path)
qg_model.add_tokenizer(qg_tokenizer)




qa_model_path = "/scratch/yyv959/commonsenseqa/outputs/gpt2-large/qa-v2/"

qa_model = GenerativeGPT2QA2.from_pretrained(qa_model_path)
qa_tokenizer = GPT2Tokenizer.from_pretrained(qa_model_path)
qa_model.add_tokenizer(qa_tokenizer)


qd_model_path = "/scratch/yyv959/commonsenseqa/outputs/gpt2-large/qd/"

qd_model = GenerativeGPT2QD.from_pretrained(qd_model_path)
qd_tokenizer = GPT2Tokenizer.from_pretrained(qd_model_path)
qd_model.add_tokenizer(qd_tokenizer)

distractor_size = 4
data = []
#qg_model.cuda()

qg_model = GenerativeGPT2QGWrapper(qg_model)
qa_model = GenerativeGPT2QA2Wrapper(qa_model)
qd_model = GenerativeGPT2QDWrapper(qd_model)
qg_model.eval()
qa_model.eval()
qd_model.eval()

#qa_model = torch.nn.DataParallel(qa_model)
#qd_model = torch.nn.DataParallel(qd_model)
#qg_model = torch.nn.DataParallel(qg_model)
qa_model.cuda()
qd_model.cuda()
qg_model.cuda()



with open(dir + args.name + ".csv",
          'w',
          encoding='utf8',
          newline='') as tsv_file:
    tsv_writer = csv.writer(tsv_file, delimiter=',', lineterminator='\n')

    tsv_writer.writerow([
        "id", "question", "concept", "true_answer", "wrong1", "wrong2",
        "wrong3", "wrong4"
    ])

    questions = []

    for i in trange(args.epochs):
        with torch.no_grad():
            #qg_model.cuda()
            questions += qg_model(80, 62, sample=True, tmp=1)
            #qg_model.cpu()
    qg_model.cpu()
    qg_model = None
    with torch.no_grad():
        for question in tqdm(questions):
            question = question.split("\t")[0]
            question = "Q: " + question
            input_ids_qa = qa_tokenizer.tokenize(question)
            input_ids_qa += ["A", ":"]
            input_ids_qa = torch.tensor(
                qa_tokenizer.convert_tokens_to_ids(input_ids_qa), dtype=torch.long)
            input_ids_qa = input_ids_qa.view(1, -1)
            input_ids_qd = qd_tokenizer.tokenize(question)
            input_ids_qd += ["A", ":"]
            input_ids_qd = torch.tensor(
                qd_tokenizer.convert_tokens_to_ids(input_ids_qd), dtype=torch.long)
            input_ids_qd = input_ids_qd.view(1, -1)
            #qa_model.cuda()
            res = qa_model(input_ids_qa.cuda(),
                           12,
                           sample=False,
                           tmp=1.0,
                           label=None,
                           top_p=0.9)
            #qa_model.cpu()
            ans = res.split("\t")[1]

            #                distractors = set({})
            distractors = qd_model(input_ids_qd.cuda(),
                                   12,
                                   num_distractors=distractor_size+1,
                                   sample=True,
                                   tmp=1,
                                   label=None,
                                   top_p=1.0)
            distractors = set(distractors)
            distractors = list(distractors)[:4]
            distractors = set(distractors)
            #if len(distractors) > distractor_size:
            #    distractors = list(distractors)
            while len(distractors) < distractor_size:
                new_distractors = qd_model(input_ids_qd.cuda(),
                                           12,
                                           num_distractors=distractor_size -
                                           len(distractors),
                                           sample=True,
                                           tmp=1,
                                           label=None,
                                           top_p=1.0)
                distractors = distractors.union(set(new_distractors))
            if len(distractors) != distractor_size:
                x = input(str(distractors))
            #qd_model.cuda()
            #                for _ in range(distractor_size):
            #                    question, distractor = qd_model.generate(input_ids_qd.cuda(),
            #                                                             12,
            #                                                             sample=True,
            #                                                             tmp=1.0,
            #                                                             top_p=1.0,
            #                                                             label=None)
            #                    while distractor in distractors or distractor == ans:
            #                        question, distractor = qd_model.generate(
            #                            input_ids_qd.cuda(),
            #                            12,
            #                            sample=True,
            #                            tmp=1.0,
            #                            top_p=1.0,
            #                            label=None)
            #                    distractors.add(distractor.strip())
            #qd_model.cpu()
            #output = [question] + [ans] + list(distractors)
            tsv_writer.writerow(
                ["n/a",
                 question.replace("Q: ", "").strip(), "n/a",
                 ans.strip()] + list(distractors))
        #data.append(tuple(output))
#random.shuffle(data)
#with open(dir + "train_fake_100000" + ".csv", 'w', encoding='utf8', newline='') as tsv_file:
#    tsv_writer = csv.writer(tsv_file, delimiter=',', lineterminator='\n')
#
#    tsv_writer.writerow(["id", "question", "concept", "true_answer", "wrong1" , "wrong2", "wrong3" , "wrong4"])

#    for  q,  t, w1, w2, w3, w4 in data:
#        tsv_writer.writerow(["n/a",q,"n/a",t,w1,w2,w3,w4])
