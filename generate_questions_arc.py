from transformers import (WEIGHTS_NAME, GPT2Config, GPT2Tokenizer)
from generative_qg import GenerativeGPT2QG, GenerativeGPT2QGWrapper
from generative_qa import GenerativeGPT2QA2, GenerativeGPT2QD, GenerativeGPT2QA2Wrapper, GenerativeGPT2QDWrapper
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
    default="train_fake_sample",
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
    default="/net/nfs.websail/yyv959/arc/arc-hard/",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)

parser.add_argument(
    "--qg_model_path",
    default="/net/nfs.websail/yyv959/arc/outputs/gpt2-medium/qg-hard/",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)
parser.add_argument(
    "--qa_model_path",
    default="/net/nfs.websail/yyv959/arc/outputs/gpt2-medium/qa-hard/",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)
parser.add_argument(
    "--qd_model_path",
    default="/net/nfs.websail/yyv959/arc/outputs/gpt2-medium/qd-hard/",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)
parser.add_argument(
    "--mc_model_path",
    default="/net/nfs.websail/yyv959/arc/outputs/roberta-large/mc-hard-6",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)

args = parser.parse_args()

dir = args.dir
qg_model_path = args.qg_model_path

qg_model = GenerativeGPT2QG.from_pretrained(qg_model_path)

qg_tokenizer = GPT2Tokenizer.from_pretrained(qg_model_path)
qg_model.add_tokenizer(qg_tokenizer)

qa_model_path = args.qa_model_path

qa_model = GenerativeGPT2QA2.from_pretrained(qa_model_path)
qa_tokenizer = GPT2Tokenizer.from_pretrained(qa_model_path)
qa_model.add_tokenizer(qa_tokenizer)

qd_model_path = args.qd_model_path

qd_model = GenerativeGPT2QD.from_pretrained(qd_model_path)
qd_tokenizer = GPT2Tokenizer.from_pretrained(qd_model_path)
qd_model.add_tokenizer(qd_tokenizer)

mc_model_path = args.mc_model_path

mc_model = RobertaForMultipleChoice.from_pretrained(mc_model_path)

mc_tokenizer = RobertaTokenizer.from_pretrained(mc_model_path)

distractor_size = 3
data = []
#qg_model.cuda()

qg_model = GenerativeGPT2QGWrapper(qg_model)
qa_model = GenerativeGPT2QA2Wrapper(qa_model)
qd_model = GenerativeGPT2QDWrapper(qd_model)

qg_model.eval()
qa_model.eval()
qd_model.eval()
mc_model.eval()

#qa_model = torch.nn.DataParallel(qa_model)
#qd_model = torch.nn.DataParallel(qd_model)
#qg_model = torch.nn.DataParallel(qg_model)
qa_model.cuda()
qd_model.cuda()
qg_model.cuda()
mc_model.cuda()

with open(dir + args.name + ".csv", 'w', encoding='utf8',
          newline='') as tsv_file:
    tsv_writer = csv.writer(tsv_file, delimiter=',', lineterminator='\n')

    tsv_writer.writerow([
        "id", "question", "concept", "true_answer", "wrong1", "wrong2",
        "wrong3"
    ])

    questions = []

    for i in trange(args.epochs):
        with torch.no_grad():
            #qg_model.cuda()
            questions += qg_model(80, 90, sample=True, tmp=1)
            #qg_model.cpu()
    qg_model.cpu()
    qg_model = None
    with torch.no_grad():
        for question in tqdm(questions):
            question = question.split("\t")[0]
            if question.strip() == "":
                print("skipped")
                continue
            question = "Q: " + question

            input_ids_qa = qa_tokenizer.tokenize(question)
            input_ids_qa += ["A", ":"]

            input_ids_qa = torch.tensor(
                qa_tokenizer.convert_tokens_to_ids(input_ids_qa),
                dtype=torch.long)
            input_ids_qa = input_ids_qa.view(1, -1)
            input_ids_qd = qd_tokenizer.tokenize(question)
            input_ids_qd += ["A", ":"]
            input_ids_qd = torch.tensor(
                qd_tokenizer.convert_tokens_to_ids(input_ids_qd),
                dtype=torch.long)
            input_ids_qd = input_ids_qd.view(1, -1)
            #qa_model.cuda()
            generation_length = 120 - input_ids_qa.size(1)

            res = qa_model(input_ids_qa.cuda(),
                           generation_length,
                           sample=False,
                           tmp=1.0,
                           label=None,
                           top_p=0.9)
            #qa_model.cpu()
            ans = res.split("\t")[1]

            #                distractors = set({})
            if ans.strip() == "":
                print("skipped")
                continue
            distractors = qd_model(
                input_ids_qd.cuda(),
                generation_length,
                num_distractors=distractor_size,
                sample=True,
                tmp=1,
                label=None,
                top_p=0.9)
            distractors = [it.strip() for it in distractors]
            distractors = set(distractors)
            if "" in distractors:
                print("discarded")
                distractors.discard("")
            distractors = list(distractors)[:3]
            distractors = set(distractors)
            #if len(distractors) > distractor_size:
            #    distractors = list(distractors)

            count = 0
            while len(distractors) < distractor_size:
                if count == 10:
                    break
                new_distractors = qd_model(
                    input_ids_qd.cuda(),
                    generation_length,
                    num_distractors=distractor_size - len(distractors),
                    sample=True,
                    tmp=1,
                    label=None,
                    top_p=0.9)
                new_distractors = [it.strip() for it in new_distractors]
                distractors = distractors.union(set(new_distractors))
                if "" in distractors:
                    print("discarded")
                    distractors.discard("")
                count += 1


            if len(distractors) != distractor_size:
                print("skipped")
                continue
            options = list(distractors) + [ans]
            sent1 = [mc_tokenizer.cls_token
                     ] + mc_tokenizer.tokenize( "Q: " + question) + [
                         mc_tokenizer.sep_token
                     ] + mc_tokenizer.tokenize("A: " + options[0]) + [
                         mc_tokenizer.sep_token
                     ]
            sent2 = [mc_tokenizer.cls_token
                     ] + mc_tokenizer.tokenize("Q: " + question) + [
                         mc_tokenizer.sep_token
                     ] + mc_tokenizer.tokenize("A: " + options[1]) + [
                         mc_tokenizer.sep_token
                     ]
            sent3 = [mc_tokenizer.cls_token
                     ] + mc_tokenizer.tokenize("Q: " + question) + [
                         mc_tokenizer.sep_token
                     ] + mc_tokenizer.tokenize("A: " + options[2]) + [
                         mc_tokenizer.sep_token
                     ]
            sent4 = [mc_tokenizer.cls_token
                     ] + mc_tokenizer.tokenize("Q: " + question) + [
                         mc_tokenizer.sep_token
                     ] + mc_tokenizer.tokenize("A: " + options[3]) + [
                         mc_tokenizer.sep_token
                     ]
            input_ids_1 = mc_tokenizer.convert_tokens_to_ids(sent1)
            input_mask_1 = [1] * len(input_ids_1)
            input_ids_2 = mc_tokenizer.convert_tokens_to_ids(sent2)
            input_mask_2 = [1] * len(input_ids_2)
            input_ids_3 = mc_tokenizer.convert_tokens_to_ids(sent3)
            input_mask_3 = [1] * len(input_ids_3)
            input_ids_4 = mc_tokenizer.convert_tokens_to_ids(sent4)
            input_mask_4 = [1] * len(input_ids_4)
            max_len = max(len(input_ids_1), len(input_ids_2), len(input_ids_3),
                          len(input_ids_4))
            pad_length_1 = max_len - len(input_ids_1)
            pad_length_2 = max_len - len(input_ids_2)
            pad_length_3 = max_len - len(input_ids_3)
            pad_length_4 = max_len - len(input_ids_4)
            input_ids_1 = input_ids_1 + [mc_tokenizer.pad_token_id
                                         ] * pad_length_1
            input_mask_1 = input_mask_1 + [0] * pad_length_1
            input_ids_2 = input_ids_2 + [mc_tokenizer.pad_token_id
                                         ] * pad_length_2
            input_mask_2 = input_mask_2 + [0] * pad_length_2
            input_ids_3 = input_ids_3 + [mc_tokenizer.pad_token_id
                                         ] * pad_length_3
            input_mask_3 = input_mask_3 + [0] * pad_length_3
            input_ids_4 = input_ids_4 + [mc_tokenizer.pad_token_id
                                         ] * pad_length_4
            input_mask_4 = input_mask_4 + [0] * pad_length_4
            input_ids = torch.tensor(
                [input_ids_1, input_ids_2, input_ids_3, input_ids_4],
                dtype=torch.long).cuda().view(1, 4, -1)
            input_mask = torch.tensor(
                [input_mask_1, input_mask_2, input_mask_3, input_mask_4],
                dtype=torch.long).cuda().view(1, 4, -1)
            output = mc_model(input_ids=input_ids, attention_mask=input_mask)
            logits = output[0]
            pred_1 = np.argmax(logits.data.cpu().numpy())
            ans = options[pred_1]

            wrongs = [it for it in options if it != ans]
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
                ["n/a", question.replace("Q: ", "").strip(), "n/a",
                 ans.strip()] + list(wrongs))
        #data.append(tuple(output))
#random.shuffle(data)
#with open(dir + "train_fake_100000" + ".csv", 'w', encoding='utf8', newline='') as tsv_file:
#    tsv_writer = csv.writer(tsv_file, delimiter=',', lineterminator='\n')
#
#    tsv_writer.writerow(["id", "question", "concept", "true_answer", "wrong1" , "wrong2", "wrong3" , "wrong4"])

#    for  q,  t, w1, w2, w3, w4 in data:
#        tsv_writer.writerow(["n/a",q,"n/a",t,w1,w2,w3,w4])
