from nltk.corpus import stopwords
from transformers import (RobertaConfig, RobertaForMultipleChoice,
                          RobertaTokenizer)
import gensim
from sentence_transformers import SentenceTransformer
import random
import sys
import csv
import spacy
from scipy.spatial.distance import cosine

import numpy as np
import torch
from tqdm import tqdm, trange
from math import exp
import argparse
from nltk.corpus import stopwords


def randargmax(b, axis=1):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b == np.repeat(
        np.expand_dims(b.max(axis=axis), axis), b.shape[axis], axis=axis)),
                     axis=axis)

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    # However, since we'd better not to remove tokens of options and questions, you can choose to use a bigger
    # length or only pop from context
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            #logger.info(
            #    'Attention! you are removing from token_b (swag task is ok). '
            #    'If you are training ARC and RACE (you are poping question + options), '
            #    'you need to try to use a bigger max seq length!')
            tokens_b.pop()


def convert_text_to_feature(A, B1, B2, B3, B4, mc_tokenizer, max_seq_length=128):
    Q = mc_tokenizer.tokenize( A)
    ending1 = mc_tokenizer.tokenize('I ' + B1)
    ending2 = mc_tokenizer.tokenize('I ' + B2)
    ending3 = mc_tokenizer.tokenize('I ' + B3)
    ending4 = mc_tokenizer.tokenize('I ' + B4)



    _truncate_seq_pair(Q, ending1, max_seq_length - 3)
    _truncate_seq_pair(Q, ending2, max_seq_length - 3)
    _truncate_seq_pair(Q, ending3, max_seq_length - 3)
    _truncate_seq_pair(Q, ending4, max_seq_length - 3)

    sent1 = [mc_tokenizer.cls_token] + Q + [mc_tokenizer.sep_token] + ending1 + [mc_tokenizer.sep_token]
    sent2 = [mc_tokenizer.cls_token] + Q + [mc_tokenizer.sep_token] + ending2 + [mc_tokenizer.sep_token]
    sent3 = [mc_tokenizer.cls_token] + Q + [mc_tokenizer.sep_token] + ending3 + [mc_tokenizer.sep_token]
    sent4 = [mc_tokenizer.cls_token] + Q + [mc_tokenizer.sep_token] + ending4 + [mc_tokenizer.sep_token]

    input_ids_1 = mc_tokenizer.convert_tokens_to_ids(sent1)
    input_mask_1 = [1] * len(input_ids_1)
    input_ids_2 = mc_tokenizer.convert_tokens_to_ids(sent2)
    input_mask_2 = [1] * len(input_ids_2)
    input_ids_3 = mc_tokenizer.convert_tokens_to_ids(sent3)
    input_mask_3 = [1] * len(input_ids_3)
    input_ids_4 = mc_tokenizer.convert_tokens_to_ids(sent4)
    input_mask_4 = [1] * len(input_ids_4)

    max_len = max(len(input_ids_1), len(input_ids_2), len(input_ids_3), len(input_ids_4))
    pad_length_1 = max_len - len(input_ids_1)
    pad_length_2 = max_len - len(input_ids_2)
    pad_length_3 = max_len - len(input_ids_3)
    pad_length_4 = max_len - len(input_ids_4)

    input_ids_1 = input_ids_1 + [mc_tokenizer.pad_token_id] * pad_length_1
    input_mask_1 = input_mask_1 + [0] * pad_length_1
    input_ids_2 = input_ids_2 + [mc_tokenizer.pad_token_id] * pad_length_2
    input_mask_2 = input_mask_2 + [0] * pad_length_2
    input_ids_3 = input_ids_3 + [mc_tokenizer.pad_token_id] * pad_length_3
    input_mask_3 = input_mask_3 + [0] * pad_length_3
    input_ids_4 = input_ids_4+ [mc_tokenizer.pad_token_id] * pad_length_4
    input_mask_4 = input_mask_4 + [0] * pad_length_4

    input_ids = torch.tensor([input_ids_1, input_ids_2, input_ids_3, input_ids_4],
                             dtype=torch.long).to(device).view(1, 4, -1)
    input_mask = torch.tensor([input_mask_1, input_mask_2, input_mask_3, input_mask_4],
                              dtype=torch.long).to(device).view(1, 4, -1)

    return input_ids, input_mask


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dir",
    default="/home/yyv959/winogrande/train_l/",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)

parser.add_argument(
    "--mc_model_path",
    default="/net/nfs.websail/yyv959/winogrande/outputs/roberta-large/train-l-mc-fake-medium-sym-200000-unigram-8/",
    type=str,
    help=
    "The input data dir. Should contain the .tsv files (or other data files) for the task."
)
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Avoid using CUDA when available")
args = parser.parse_args()

dir = args.dir

mc_model_path = args.mc_model_path

mc_model = RobertaForMultipleChoice.from_pretrained(mc_model_path)

mc_tokenizer = RobertaTokenizer.from_pretrained(mc_model_path)

mc_model.eval()

device = torch.device(
    "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

sent_encoder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens',
                                   device=device)
mc_model.to(device)

tagger = spacy.load("en_core_web_lg")
word_vector = gensim.models.KeyedVectors.load_word2vec_format(
    '/net/nfs.websail/yyv959/counter-fitted-vectors.txt', binary=False)
stop_words = stopwords.words('english')

data = []

adv_data = []

num_correct = 0

with open(dir + "dev.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for line in reader:
        data.append(line)
with torch.no_grad():
    for example in tqdm(data):
        q = example[1]
        q_token = list(tagger(q))
        token_length = len(q_token)
        num_attempt = 0
        num_word_swap = 0
        if_success = False
        candidate_pos = []
        for i, token in enumerate(q_token):
            #print(token.pos_)
            if token.text.lower() in stop_words:
                continue
            if "_" in token.text:
                continue
            if token.pos_ == "PUNCT":
                continue
            if token.text not in word_vector:

                continue
            #if token.text in example[3] or token.text in example[4]:
            #    continue
            candidate_pos.append(i)

        input_ids, input_mask = convert_text_to_feature(
            q, example[3], example[4],example[5], example[6], mc_tokenizer)
        output = mc_model(input_ids=input_ids, attention_mask=input_mask)
        logits = output[0]
        pred_org = np.argmax(logits.data.cpu().numpy())
        if_correct_org = (pred_org == 0)
        org_score_list = torch.nn.functional.softmax(logits,
                                                     -1)[0, :].cpu().numpy()
        org_score = org_score_list[pred_org]

        q_emb = sent_encoder.encode([q])[0]
        #print(org_score)
        candidate_score = []
        for pos in candidate_pos:
            q_token_deleted = [
                token.text for i, token in enumerate(q_token) if i != pos
            ]
            q_deleted = " ".join(q_token_deleted)
            input_ids, input_mask = convert_text_to_feature(
                q_deleted, example[3], example[4],example[5], example[6], mc_tokenizer)
            output = mc_model(input_ids=input_ids, attention_mask=input_mask)
            logits = output[0]
            pred_deleted = np.argmax(logits.data.cpu().numpy())
            deleted_score_list = torch.nn.functional.softmax(
                logits, -1)[0, :].cpu().numpy()

            if pred_deleted == pred_org:
                importance = org_score - deleted_score_list[pred_deleted]
            else:
                importance = (org_score - deleted_score_list[pred_org]) + (
                    deleted_score_list[pred_deleted] -
                    org_score_list[pred_deleted])
            #importance = np.sum( deleted_score_list *  (np.log(deleted_score_list) - np.log(org_score_list)) )
            #importance = np.sum( org_score_list *  (np.log(org_score_list) - np.log(deleted_score_list)) )
            candidate_score.append(importance * -1)
        sorted_candidate_pos = [
            x for _, x in sorted(zip(candidate_score, candidate_pos))
        ]
        min_confidence = 1
        adv_q = [it for it in q_token]


        for p in sorted_candidate_pos:
            w = q_token[p]

            successful = False
            max_semantic_similarity = 0
            best_q = None
            best_if_correct = None
            lc_q = None
            for (syn_w, similarity) in word_vector.similar_by_word(w.text,
                                                                   topn=50):
                if similarity < 0.7:
                    continue

                new_q = " ".join( [it.text if i != p else syn_w for i,it in enumerate(adv_q)])
                new_q_token = list(tagger(new_q))
                if new_q_token[p].pos_ != w.pos_:
                    continue
                new_q_emb = sent_encoder.encode([new_q])[0]
                semantic_similarity = 1-cosine(new_q_emb,q_emb)
                if semantic_similarity < 0.7:

                    continue
                input_ids, input_mask = convert_text_to_feature(
                    new_q, example[3], example[4],example[5], example[6],mc_tokenizer)
                output = mc_model(input_ids=input_ids, attention_mask=input_mask)
                logits = output[0]
                pred_new = np.argmax(logits.data.cpu().numpy())
                score_new = torch.nn.functional.softmax(
                    logits, -1)[0, pred_new].cpu().numpy()
                num_attempt += 1
                if pred_new != pred_org:
                    successful = True
                    if semantic_similarity > max_semantic_similarity:
                        max_semantic_similarity = semantic_similarity
                        best_q = new_q
                        best_if_correct = (pred_new == 0)
                elif score_new < min_confidence:
                    min_confidence = score_new
                    lc_q = new_q

            if successful:
                num_word_swap += 1
                if_success = True

                break
            elif lc_q:
                num_word_swap += 1
                adv_q = [it for it in  tagger(lc_q) ]




        if if_success:
            if not if_correct_org:
                if best_if_correct:
                    num_correct += 1
            adv_data.append((best_q, num_attempt, num_word_swap,len(sorted_candidate_pos),token_length,max_semantic_similarity,"successful"))
        else:
            if if_correct_org:
                num_correct += 1
            adv_data.append((q, num_attempt, num_word_swap,len(sorted_candidate_pos),token_length,0,"failed"))


with open(mc_model_path + "text_fooler.csv", 'w', encoding='utf8',
          newline='') as tsv_file:
    tsv_writer = csv.writer(tsv_file, delimiter=',', lineterminator='\n')

    tsv_writer.writerow(["q","num_attempt", "num_word_swap","num_candidate_pos","token_length","max_semantic_similarity","if_success" ])

    for it in adv_data:
        tsv_writer.writerow(it)
num_fail = 0
total = 0
total_attempts = 0
total_num_word_swap = 0
average_ratio = 0
average_semantic_similarity = 0
for it in adv_data:
    if it[-1] == "failed":
        num_fail += 1
    else:
        average_ratio += it[2] / it[4]
        average_semantic_similarity += it[5]
    total_attempts += it[1]
    total_num_word_swap += it[2]


    total += 1

with open(mc_model_path + "text_fooler_stats.txt","w" ) as out:


    out.write("total_attempts" + "\n")
    out.write(str(total_attempts) + "\n")
    out.write("average_attempts" + "\n")
    out.write(str(total_attempts/total) + "\n")
    out.write("total_num_word_swap" + "\n")
    out.write(str(total_num_word_swap) )

    out.write("accuracy" + "\n")
    out.write(str(num_correct/total) + "\n")
    out.write("failure rate" + "\n")
    out.write(str(num_fail/total) + "\n")
    out.write("average purturb ratio" + "\n")
    out.write(str(average_ratio/ ( total - num_fail ) ) )
    out.write("average semantic similarity" + "\n")
    out.write(str(average_semantic_similarity/ ( total - num_fail ) ) )
