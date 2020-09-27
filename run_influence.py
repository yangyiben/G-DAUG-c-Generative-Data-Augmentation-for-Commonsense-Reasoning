# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer, XLNetConfig,
                          XLNetForMultipleChoice, XLNetTokenizer,
                          RobertaConfig, RobertaForMultipleChoice,
                          BertForMultipleChoice, RobertaTokenizer)

from transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from torch.optim import Adam

from data_utils import (convert_examples_to_features, processors)
from winogrande_data_utils import convert_examples_to_features as convert_examples_to_features_winogrande

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (BertConfig, XLNetConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForMultipleChoice, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer),
    'roberta': (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer)
}


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features]
            for feature in features]


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def randargmax(b, axis=1):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b == np.repeat(
        np.expand_dims(b.max(axis=axis), axis), b.shape[axis], axis=axis)),
                     axis=axis)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_validation_grad(args, eval_dataset, model):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running validation grad*****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", 4)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    random_state = torch.get_rng_state()
    model.zero_grad()
    #eval_dataloader = eval_dataloader[:10]
    count = 0
    for batch in tqdm(eval_dataloader, desc="Calculating validation grad"):
        #if count > 10:

        #    break
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        inputs = {
            'input_ids':
            batch[0],
            'attention_mask':
            batch[1],
            'token_type_ids':
            batch[2] if args.model_type in ['bert', 'xlnet'] else
            None,  # XLM don't use segment_ids
            'labels':
            None
        }
        outputs = model(**inputs)
        logits = outputs[0]
        loss = F.cross_entropy(logits, batch[3], reduction='sum')
        loss.backward()
        count += 1
    grad = []
    for p in model.parameters():
        if p.grad is None:
            print("wrong")
        #print(len(eval_dataset))
        grad.append((p.grad.data / len(eval_dataset)).cpu())

    #print(grad)
    return grad


def get_influence(args, train_dataset, model, HVP, args):
    eval_sampler = SequentialSampler(train_dataset)
    eval_dataloader = DataLoader(train_dataset,
                                 sampler=eval_sampler,
                                 batch_size=1)

    # Eval!
    logger.info("***** Running influence *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", 1)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    random_state = torch.get_rng_state()
    model.zero_grad()
    #eval_dataloader = eval_dataloader[:10]
    HVP = [it.cuda() for it in HVP]
    no_decay = ['bias', 'LayerNorm.weight']
    count = 0
    negative_count = 0
    influence_list = []
    for batch in tqdm(eval_dataloader, desc="Calculating validation grad"):
        #if count > 10:
        #    break
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        inputs = {
            'input_ids':
            batch[0],
            'attention_mask':
            batch[1],
            'token_type_ids':
            batch[2] if args.model_type in ['bert', 'xlnet'] else
            None,  # XLM don't use segment_ids
            'labels':
            None
        }
        model.zero_grad()
        outputs = model(**inputs)
        logits = outputs[0]
        loss = F.cross_entropy(logits, batch[3], reduction='mean')
        loss.backward()
        count += 1
        influence = 0
        for i, ((n, p), v) in enumerate(zip(model.named_parameters(), HVP)):
            if p.grad is None:
                print("wrong")
            else:
                if not any(nd in n for nd in no_decay):
                    influence += (
                        (p.grad.data.add_(args.weight_decay, p.data)) *
                        v).sum() * -1


#                    influence += ((p.grad.data)*v).sum() * -1
                else:
                    influence += ((p.grad.data) * v).sum() * -1

        if influence.item() < 0:
            negative_count += 1
        influence_list.append(influence.item())
        if count % 100 == 0:
            print(influence.item())
            print(negative_count / count)
    influence_list = np.array(influence_list)
    return influence_list


def get_HVP(args, train_dataset, model, v, args):
    train_sampler = RandomSampler(train_dataset,
                                  replacement=True,
                                  num_samples=args.t)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size //
                                  args.gradient_accumulation_steps)
    #grad = torch.autograd.grad(y, linear.parameters(), create_graph=True)

    no_decay = ['bias', 'LayerNorm.weight']
    final_res = None

    #v = [it.cpu() for it in v]
    #torch.cuda.empty_cache()
    for r in trange(args.r):
        res = [w.clone().cuda() for w in v]
        model.zero_grad()
        for step, batch in enumerate(
                tqdm(train_dataloader, desc="Calculating HVP")):
            model.eval()

            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                'input_ids':
                batch[0],
                'attention_mask':
                batch[1],
                'token_type_ids':
                batch[2] if args.model_type in ['bert', 'xlnet'] else
                None,  # XLM don't use segment_ids
                'labels':
                None
            }

            length = inputs["attention_mask"].sum(2).max().item()
            #print(length)
            #print(inputs["attention_mask"])
            #x = input("stop")
            inputs["input_ids"] = inputs["input_ids"][:, :length]
            inputs["attention_mask"] = inputs["attention_mask"][:, :length]
            outputs = model(**inputs)
            logits = outputs[0]

            loss = F.cross_entropy(logits, batch[3], reduction='mean')

            grad_list = torch.autograd.grad(loss,
                                            model.parameters(),
                                            create_graph=True)
            grad = []
            H = 0
            for i, (g, g_v) in enumerate(zip(grad_list, res)):

                H += (g * g_v).sum() / args.gradient_accumulation_steps
            #H = grad @ v
            H.backward()

            #grad = []
            if (step + 1) % args.gradient_accumulation_steps == 0:

                print(res[20])

                for i, ((n, p),
                        v_p) in enumerate(zip(model.named_parameters(), res)):
                    try:
                        if not any(nd in n for nd in no_decay):
                            res[i] = (1 - args.damping) * v_p - (
                                p.grad.data.add_(args.weight_decay,
                                                 v_p)) / args.c + v[i].cuda()
                        else:
                            res[i] = (1 - args.damping) * v_p - (
                                p.grad.data) / args.c + v[i].cuda()
                    except RuntimeError:

                        v_p = v_p.cpu()

                        p_grad = p.grad.data.cpu()

                        if not any(nd in n for nd in no_decay):
                            res[i] = ((1 - args.damping) * v_p -
                                      (p_grad.add_(args.weight_decay, v_p)) /
                                      args.c + v[i]).cuda()
                        else:
                            res[i] = ((1 - args.damping) * v_p -
                                      (p_grad) / args.c + v[i]).cuda()
                model.zero_grad()

        if final_res is None:
            final_res = [(b / args.c).cpu().float() for b in res]
        else:
            final_res = [
                a + (b / args.c).cpu().float() for a, b in zip(final_res, res)
            ]

    final_res = [a / float(args.r) for a in final_res]
    return final_res
    #    grad.append(p.grad.data)
    #grad = torch.cat(grad, 0)
    #print(grad)


def load_and_cache_examples(args,
                            task,
                            tokenizer,
                            evaluate=False,
                            test=False,
                            fake=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if fake:
        data_dir = args.fake_data_dir
    else:
        data_dir = args.data_dir
    processor = processors[task]()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = 'dev_random'
    elif test:
        cached_mode = 'test'
    else:
        cached_mode = 'train'
    assert (evaluate == True and test == True) == False
    cached_features_file = os.path.join(
        data_dir, 'cached_{}_{}_{}_{}{}'.format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length), str(task),
            "_no_q" if args.mask_question else ""))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        if evaluate:
            examples = processor.get_dev_examples(data_dir)
        elif test:
            examples = processor.get_test_examples(data_dir)
        else:
            examples = processor.get_train_examples(data_dir)

        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(data_dir)
        elif test:
            examples = processor.get_test_examples(data_dir)
        else:
            examples = processor.get_train_examples(data_dir)
        logger.info("Training number: %s", str(len(examples)))
        if args.task_name != "winogrande":
            features = convert_examples_to_features(
                examples,
                label_list,
                args.max_seq_length,
                tokenizer,
                cls_token_at_end=bool(args.model_type in ['xlnet']
                                      ),  # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                sep_token=tokenizer.sep_token,
                sep_token_extra=False,
                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(
                    args.model_type in ['xlnet']),  # pad on the left for xlnet
                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                pad_token=tokenizer.pad_token_id,
                mask_question=args.mask_question,
                pad_qa=args.task_name in ["commonsenseqa", "arc"])
        else:
            features = convert_examples_to_features_winogrande(
                examples,
                label_list,
                args.max_seq_length,
                tokenizer,
                cls_token_at_end=bool(args.model_type in ['xlnet']
                                      ),  # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                sep_token=tokenizer.sep_token,
                sep_token_extra=False,
                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(
                    args.model_type in ['xlnet']),  # pad on the left for xlnet
                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                pad_token=tokenizer.pad_token_id,
                mask_question=args.mask_question,
                pad_qa=True)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, 'input_ids'),
                                 dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'),
                                  dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'),
                                   dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_label_ids)
    if test:
        return dataset, examples
    else:
        return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default="/scratch/yyv959/commonsenseqa/",
        type=str,
        help=
        "The input data dir. Should contain the .tsv files (or other data files) for the task."
    )
    parser.add_argument(
        "--fake_data_dir",
        default="/scratch/yyv959/commonsenseqa/",
        type=str,
        help=
        "The input data dir. Should contain the .tsv files (or other data files) for the task."
    )
    parser.add_argument("--model_type",
                        default='roberta',
                        type=str,
                        help="Model type selected in the list: " +
                        ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument(
        "--model_name_or_path",
        default="roberta-large",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: "
    )
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: "
    )
    parser.add_argument(
        "--task_name",
        default="commonsenseqa",
        type=str,
        help="The name of the task to train selected in the list: " +
        ", ".join(processors.keys()))

    ## Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help=
        "Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument(
        "--max_seq_length",
        default=70,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Set this flag if you are using an uncased model.")
    parser.add_argument(
        "--mask_question",
        action='store_true',
        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=10,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--gradient_accumulation_steps",
                        default=10,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--damping",
                        default=0.01,
                        type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--c",
                        default=1e7,
                        type=float,
                        help="Weight deay if we apply some.")

    parser.add_argument("--r",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--t",
                        default=8000,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--logging_steps',
                        type=int,
                        default=609,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps',
                        type=int,
                        default=100000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action='store_true',
        help=
        "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
    )
    parser.add_argument(
        "--no_hessian",
        action='store_true',
        help=
        "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
    )
    parser.add_argument(
        "--load_hvp",
        action='store_true',
        help=
        "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
    )
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument(
        '--overwrite_cache',
        action='store_true',
        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.device = torch.device("cuda")

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config)

    model.to(args.device)
    #if args.fp16:
    #    model.half()
    train_dataset = load_and_cache_examples(args,
                                            args.task_name,
                                            tokenizer,
                                            evaluate=False,
                                            test=False)
    eval_dataset = load_and_cache_examples(args,
                                           args.task_name,
                                           tokenizer,
                                           evaluate=True,
                                           test=False)
    fake_dataset = load_and_cache_examples(args,
                                           args.task_name,
                                           tokenizer,
                                           evaluate=False,
                                           test=False,
                                           fake=True)

    #eval_dataset = eval_dataset[:200,:]
    #print(eval_dataset
    #)
    if not args.load_hvp:
        grad = get_validation_grad(args, eval_dataset, model)
        if args.no_hessian:
            HVP = grad
        else:
            HVP = get_HVP(args, train_dataset, model, grad, args)
        torch.save(
            HVP, args.output_dir + "HVP_" + str(args.train_batch_size) + "b_" +
            str(args.t) + "t_" + str(args.r) + "r")
    else:
        HVP = torch.load(args.output_dir + "HVP_" +
                         str(args.train_batch_size) + "b_" + str(args.t) +
                         "t_" + str(args.r) + "r")
    influences = get_influence(args, fake_dataset, model, HVP, args)
    if args.no_hessian:
        np.save(
            os.path.join(args.output_dir,
                         "train_data_influences_no_hessian" + ".npy"),
            influences)
    else:
        np.save(
            os.path.join(
                args.output_dir,
                "fake_data_300000_influences" + str(args.train_batch_size) +
                "b_" + str(args.t) + "t_" + str(args.r) + "r" + ".npy"),
            influences)


if __name__ == "__main__":
    main()
