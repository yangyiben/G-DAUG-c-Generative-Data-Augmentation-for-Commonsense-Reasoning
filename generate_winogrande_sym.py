from transformers import (WEIGHTS_NAME, GPT2Config, GPT2Tokenizer)
from generative_qg import GenerativeGPT2QGWinogrande
from generative_qa import GenerativeGPT2WinograndeChoice
from transformers import (RobertaConfig, RobertaForMultipleChoice,
                          RobertaTokenizer)
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
    default="train_fake_medium_sym_sample",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)
parser.add_argument(
    "--epochs",
    default=1,
    type=int,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)

parser.add_argument(
    "--dir",
    default="/net/nfs.websail/yyv959/winogrande/train_l/",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)

parser.add_argument(
    "--qg_model_path",
    default="/net/nfs.websail/yyv959/winogrande/outputs/gpt2-medium-scratch/train-l-qg-winogrande/",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)
parser.add_argument(
    "--qa_model_path",
    default="/net/nfs.websail/yyv959/winogrande/outputs/gpt2-medium-scratch/train-l-winogrande-choice/",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)
parser.add_argument(
    "--mc_model_path",
    default="/net/nfs.websail/yyv959/winogrande/outputs/roberta-large/train-l-mc-10/",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Avoid using CUDA when available")
args = parser.parse_args()

dir = args.dir
qg_model_path = args.qg_model_path

qg_model = GenerativeGPT2QGWinogrande.from_pretrained(qg_model_path)

qg_tokenizer = GPT2Tokenizer.from_pretrained(qg_model_path)
qg_model.add_tokenizer(qg_tokenizer)

qa_model_path = args.qa_model_path

qa_model = GenerativeGPT2WinograndeChoice.from_pretrained(qa_model_path)
qa_tokenizer = GPT2Tokenizer.from_pretrained(qa_model_path)
qa_model.add_tokenizer(qa_tokenizer)

mc_model_path = args.mc_model_path

mc_model = RobertaForMultipleChoice.from_pretrained(mc_model_path)

mc_tokenizer = RobertaTokenizer.from_pretrained(mc_model_path)

data = []
#qg_model.cuda()

qg_model.eval()
qa_model.eval()
mc_model.eval()

device = torch.device("cuda" if torch.cuda.is_available()
                      and not args.no_cuda else "cpu")
#qa_model = torch.nn.DataParallel(qa_model)
#qd_model = torch.nn.DataParallel(qd_model)
#qg_model = torch.nn.DataParallel(qg_model)
qa_model.to(device)
qg_model.to(device)
mc_model.to(device)

stage1_fail = 0
stage2_fail = 0
count = 0

with open(dir + args.name + ".csv", 'w', encoding='utf8',
          newline='') as tsv_file:
    tsv_writer = csv.writer(tsv_file, delimiter=',', lineterminator='\n')

    tsv_writer.writerow(["id", "question", "concept", "answer1", "answer2"])

    questions = []

    for i in trange(args.epochs):
        with torch.no_grad():
            #qg_model.cuda()
            questions += qg_model.generate_context(160, 62, sample=True, tmp=1)

            #qg_model.cpu()

    with torch.no_grad():
        for question in tqdm(questions):
            question = question.split("\t")[0]

            input_ids_qg = qg_tokenizer.tokenize(question)
            input_ids_qg = ["\n"] + input_ids_qg
            input_ids_qg = torch.tensor(
                qg_tokenizer.convert_tokens_to_ids(input_ids_qg),
                dtype=torch.long)
            questions = qg_model.continue_generate(
                input_ids_qg.to(device).expand(2, -1), 30)

            questions = [q.split("\t")[0] for q in questions]
            ans_set = set({})
            for question in questions:
                input_ids_qa = qa_tokenizer.tokenize(question)
                input_ids_qa += ["[SEP]"]
                input_ids_qa = torch.tensor(
                    qa_tokenizer.convert_tokens_to_ids(input_ids_qa),
                    dtype=torch.long)
                input_ids_qa = input_ids_qa.view(1, -1)
                #print(input_ids_qa)
                res = qa_model.generate(input_ids_qa.to(device), 10)

                #qa_model.cpu()
                #print(res)
                ans = res.split("\t")[1:]
                ans = set({it.strip() for it in ans})
                #print(ans)
                ans_set = ans_set.union(ans)
            if len(ans_set) != 2:

                stage1_fail += 1
                continue
            else:
                ans = list(ans_set)
            sent1 = [mc_tokenizer.cls_token] + mc_tokenizer.tokenize(
                questions[0].replace("_", ans[0])) + [mc_tokenizer.sep_token]
            sent2 = [mc_tokenizer.cls_token] + mc_tokenizer.tokenize(
                questions[0].replace("_", ans[1])) + [mc_tokenizer.sep_token]
            sent3 = [mc_tokenizer.cls_token] + mc_tokenizer.tokenize(
                questions[1].replace("_", ans[0])) + [mc_tokenizer.sep_token]
            sent4 = [mc_tokenizer.cls_token] + mc_tokenizer.tokenize(
                questions[1].replace("_", ans[1])) + [mc_tokenizer.sep_token]

            input_ids_1 = mc_tokenizer.convert_tokens_to_ids(sent1)
            input_mask_1 = [1] * len(input_ids_1)
            input_ids_2 = mc_tokenizer.convert_tokens_to_ids(sent2)
            input_mask_2 = [1] * len(input_ids_2)
            max_len = max(len(input_ids_1), len(input_ids_2))
            if max_len > 500:
                print("length failed")
                continue
            pad_length_1 = max_len - len(input_ids_1)
            pad_length_2 = max_len - len(input_ids_2)
            input_ids_1 = input_ids_1 + [mc_tokenizer.pad_token_id] * pad_length_1
            input_mask_1 = input_mask_1 + [0] * pad_length_1
            input_ids_2 = input_ids_2 + [mc_tokenizer.pad_token_id] * pad_length_2
            input_mask_2 = input_mask_2 + [0] * pad_length_2
            input_ids = torch.tensor([input_ids_1,input_ids_2],dtype=torch.long).to(device).view(1,2,-1)
            input_mask = torch.tensor([input_mask_1,input_mask_2],dtype=torch.long).to(device).view(1,2,-1)
            output = mc_model(input_ids = input_ids, attention_mask = input_mask)
            logits = output[0]
            pred_1 = np.argmax(logits.data.cpu().numpy())

            input_ids_1 = mc_tokenizer.convert_tokens_to_ids(sent3)
            input_mask_1 = [1] * len(input_ids_1)
            input_ids_2 = mc_tokenizer.convert_tokens_to_ids(sent4)
            input_mask_2 = [1] * len(input_ids_2)
            max_len = max(len(input_ids_1), len(input_ids_2))
            if max_len > 500:
                print("length failed")
                continue
            pad_length_1 = max_len - len(input_ids_1)
            pad_length_2 = max_len - len(input_ids_2)
            input_ids_1 = input_ids_1 + [mc_tokenizer.pad_token_id] * pad_length_1
            input_mask_1 = input_mask_1 + [0] * pad_length_1
            input_ids_2 = input_ids_2 + [mc_tokenizer.pad_token_id] * pad_length_2
            input_mask_2 = input_mask_2 + [0] * pad_length_2
            input_ids = torch.tensor([input_ids_1,input_ids_2],dtype=torch.long).to(device).view(1,2,-1)
            input_mask = torch.tensor([input_mask_1,input_mask_2],dtype=torch.long).to(device).view(1,2,-1)
            output = mc_model(input_ids = input_ids, attention_mask = input_mask)
            logits = output[0]
            pred_2 = np.argmax(logits.data.cpu().numpy())

            if pred_1 == pred_2:
                stage2_fail += 1
                continue
            else:
                count += 1
                print(count)

                tsv_writer.writerow([
                    "n/a",
                    questions[0].strip(),
                    "n/a",
                ] + [ans[pred_1], ans[pred_2] ])
                tsv_writer.writerow([
                    "n/a",
                    questions[1].strip(),
                    "n/a",
                ] + [ans[pred_2], ans[pred_1] ])
        #data.append(tuple(output))
print(stage1_fail)
print(stage2_fail)
#random.shuffle(data)
#with open(dir + "train_fake_100000" + ".csv", 'w', encoding='utf8', newline='') as tsv_file:
#    tsv_writer = csv.writer(tsv_file, delimiter=',', lineterminator='\n')
#
#    tsv_writer.writerow(["id", "question", "concept", "true_answer", "wrong1" , "wrong2", "wrong3" , "wrong4"])

#    for  q,  t, w1, w2, w3, w4 in data:
#        tsv_writer.writerow(["n/a",q,"n/a",t,w1,w2,w3,w4])
