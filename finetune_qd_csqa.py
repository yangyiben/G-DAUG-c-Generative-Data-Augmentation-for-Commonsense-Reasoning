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
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from math import exp
from transformers import (WEIGHTS_NAME, GPT2Config, GPT2Tokenizer)
from modeling import GenerativeGPT2QD
import torch.nn.functional as F
from transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule

from data_utils import (convert_examples_to_features_gpt2, processors)

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (GPT2Config, )), ())

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GenerativeGPT2QD, GPT2Tokenizer),
}


def select_field(features, field):

    return [[it[field] for it in feature.choices_features]
            for feature in features]


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def randargmin(b, axis=1):
    """ a random tie-breaking argmax"""
    return np.argmin(np.random.random(b.shape) * (b == np.repeat(
        np.expand_dims(b.min(axis=axis), axis), b.shape[axis], axis=axis)),
                     axis=axis)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(
            train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader
        ) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        args.weight_decay
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.0
    }]
    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(0.9, 0.98),
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    if args.linear_decay:
        scheduler = WarmupLinearSchedule(optimizer,
                                         warmup_steps=int(args.warmup_ratio *
                                                          t_total),
                                         t_total=t_total)
    else:
        scheduler = WarmupConstantSchedule(optimizer,
                                           warmup_steps=int(args.warmup_ratio *
                                                            t_total))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps *
        (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc, best_dev_loss = 0.0, 99999999999.0
    best_steps = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(
        args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        max_step = len(epoch_iterator)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids':
                batch[0][:, 1:, :].contiguous().view(-1, batch[0].size(2)),
                'output_ids':
                batch[1][:, 1:, :].contiguous().view(-1, batch[1].size(2)),
                'attention_mask':
                batch[2][:, 1:, :].contiguous().view(-1, batch[2].size(2)),
                'token_type_ids':
                batch[3] if args.model_type in ['xlnet'] else
                None,  # XLM don't use segment_ids
            }

            outputs = model(**inputs)
            lm_loss = outputs.sum() / ((outputs != 0).float().sum())
            loss = lm_loss

            #loss = mc_loss
            if args.n_gpu > 1:
                loss = loss.mean(
                )  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    step + 1) == max_step:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [
                        -1, 0
                ] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well

                        results = evaluate(args, model, tokenizer)

                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value,
                                                 global_step)
                        if results["eval_ppl"] > best_dev_acc:
                            best_dev_acc = results["eval_ppl"]
                            best_dev_loss = results["eval_avg_ppl"]
                            best_steps = global_step
                            if args.do_test:
                                results_test = evaluate(args,
                                                        model,
                                                        tokenizer,
                                                        test=True)
                                for key, value in results_test.items():
                                    tb_writer.add_scalar(
                                        'test_{}'.format(key), value,
                                        global_step)
                                logger.info(
                                    "test acc: %s, loss: %s, global steps: %s",
                                    str(results_test['eval_acc']),
                                    str(results_test['eval_loss']),
                                    str(global_step))
                    tb_writer.add_scalar('lr',
                                         scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) /
                                         args.logging_steps, global_step)
                    logger.info(
                        "Average loss: %s at global step: %s",
                        str((tr_loss - logging_loss) / args.logging_steps),
                        str(global_step))
                    logging_loss = tr_loss

                if args.local_rank in [
                        -1, 0
                ] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(
                        model, 'module'
                    ) else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(args,
                               os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_steps


def evaluate(args,
             model,
             tokenizer,
             prefix="",
             test=False,
             mc=True,
             generation=False):

    eval_task_names = (args.task_name, )
    eval_outputs_dirs = (args.output_dir, )

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args,
                                               eval_task,
                                               tokenizer,
                                               evaluate=not test,
                                               test=test)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(
            1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(
            eval_dataset) if args.local_rank == -1 else DistributedSampler(
                eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_avg_ppl = 0.0
        eval_ppl = 0.0
        nb_eval_steps = 1e-8
        nb_eval_time_steps = 1e-8
        preds = None
        preds_normalized = None
        preds_mc_head = None
        out_label_ids = None
        if generation:
            output_file = open(
                os.path.join(eval_output_dir, "dev_generated_ans.tsv"), 'w')
        random_state = torch.get_rng_state()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids':
                    batch[0][:, 1:, :].contiguous().view(-1, batch[0].size(2)),
                    'output_ids':
                    batch[1][:, 1:, :].contiguous().view(-1, batch[1].size(2)),
                    'attention_mask':
                    batch[2][:, 1:, :].contiguous().view(-1, batch[2].size(2)),
                    'token_type_ids':
                    batch[3] if args.model_type in ['xlnet'] else
                    None,  # XLM don't use segment_ids
                }

                mc_inputs = {
                    'input_ids':
                    batch[0].view(-1, batch[0].size(2)),
                    'output_ids':
                    batch[5].view(-1, batch[5].size(2)),
                    'attention_mask':
                    batch[2].view(-1, batch[2].size(2)),
                    'token_type_ids':
                    batch[3] if args.model_type in ['xlnet'] else
                    None,  # XLM don't use segment_ids
                }
                #output_ids = mc_inputs['input_ids'].clone().detach()
                #output_ids = output_ids[:, 1:]
                #output_ids[output_ids == tokenizer.pad_token_id] = -1
                #output_ids[output_ids == tokenizer.cls_token_id] = -1
                #mc_inputs['output_ids'] = output_ids
                outputs = model(**inputs)

                if mc:
                    logits_mc_outputs = model(**mc_inputs)
                    mc_outputs = logits_mc_outputs
                    mc_score = (-1 * mc_outputs.sum(1)).view(
                        -1, batch[0].size(1)).data.cpu().numpy()
                    mc_prob = F.softmax((-1 * mc_outputs.sum(1)).view(-1, batch[0].size(1)),
                                        -1).data.cpu().numpy()
                    mc_score_normalized = mc_outputs.sum(1) / (
                        (mc_outputs != 0).float().sum(1))
                    mc_score_normalized = (mc_score_normalized * -1).view(
                        -1, batch[0].size(1)).data.cpu().numpy()

                    if preds is None:
                        preds = mc_score
                        preds_normalized = mc_score_normalized
                        out_label_ids = batch[4].detach().cpu().numpy()

                    else:
                        preds = np.append(preds, mc_score, axis=0)
                        preds_normalized = np.append(preds_normalized,
                                                     mc_score_normalized,
                                                     axis=0)

                        out_label_ids = np.append(
                            out_label_ids,
                            batch[4].detach().cpu().numpy(),
                            axis=0)
                #output_ids = mc_inputs['input_ids'].clone().detach()
                #output_ids = output_ids[:, 1:]
                #output_ids[output_ids == tokenizer.pad_token_id] = -1
                #output_ids[output_ids == tokenizer.cls_token_id] = -1
                #mc_inputs['output_ids'] = output_ids
                outputs = model(**inputs)

                batch_total_ppl = torch.exp(
                    outputs.sum(1) / ((outputs != 0).float().sum(1))).sum()
                batch_total_loss = outputs.sum()

                eval_avg_ppl += batch_total_ppl.item()
                eval_ppl += batch_total_loss.item()
                if generation:
                    distractor_size = 3

                    sep_id = tokenizer.eos_token_id
                    for i in range(0, inputs['input_ids'].size(0),
                                   distractor_size):
                        tmp_output_ids = inputs['output_ids'][i, :]

                        question_length = (tmp_output_ids !=
                                           -1).nonzero().min() + 1
                        distractors = set({})
                        for _ in range(distractor_size):
                            question, distractor = model.generate(
                                inputs['input_ids'][i:i + 1, :question_length],
                                30,
                                sample=True,
                                tmp=1.0,
                                top_p=1.0,
                                label=inputs['input_ids'][i:i +
                                                          1, question_length:])
                            while distractor in distractors:
                                question, distractor = model.generate(
                                    inputs['input_ids'][i:i +
                                                        1, :question_length],
                                    30,
                                    sample=True,
                                    tmp=1.0,
                                    top_p=1.0,
                                    label=inputs['input_ids']
                                    [i:i + 1, question_length:])
                            distractors.add(distractor)

                        res = [question] + list(distractors)
                        output_file.write("\t".join(res) + "\n")

            nb_eval_steps += outputs.size(0)
            nb_eval_time_steps += ((outputs != 0).float().sum()).item()
        torch.set_rng_state(random_state)
        if generation:
            output_file.close()

        eval_avg_ppl = eval_avg_ppl / nb_eval_steps
        eval_ppl = exp(eval_ppl / nb_eval_time_steps)
        if mc:
            preds = F.softmax(torch.tensor(preds), -1).detach().cpu().numpy()
            np.save(os.path.join(eval_output_dir, "preds.npy"), preds)
            preds = np.argmax(preds, axis=1)
            preds_normalized = np.argmax(preds_normalized, axis=1)
            acc = simple_accuracy(preds, out_label_ids)
            acc_normalized = simple_accuracy(preds_normalized, out_label_ids)

            result = {
                "eval_ppl": eval_ppl,
                "eval_avg_ppl": eval_avg_ppl,
                "eval_acc": acc,
                "eval_acc_normalized": acc_normalized
            }
        else:
            result = {"eval_ppl": eval_ppl, "eval_avg_ppl": eval_avg_ppl}
        results.update(result)
        output_eval_file = os.path.join(
            eval_output_dir,
            "is_test_" + str(test).lower() + "_eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(
                str(prefix) + " is test:" + str(test)))
            writer.write("model           =%s\n" %
                         str(args.model_name_or_path))
            writer.write("total batch size=%d\n" %
                         (args.per_gpu_train_batch_size *
                          args.gradient_accumulation_steps *
                          (torch.distributed.get_world_size()
                           if args.local_rank != -1 else 1)))
            writer.write("train num epochs=%d\n" % args.num_train_epochs)
            writer.write("fp16            =%s\n" % args.fp16)
            writer.write("max seq length  =%d\n" % args.input_max_seq_length)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = 'dev'
    elif test:
        cached_mode = 'test'
    else:
        cached_mode = 'train'
    assert (evaluate == True and test == True) == False
    cached_features_file = os.path.join(
        args.data_dir, 'cached_{}_{}_{}_{}_{}_{}_{}v2'.format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.input_max_seq_length), str(args.output_max_seq_length),
            "generative", str(task), "qd"))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif test:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)
        logger.info("Training number: %s", str(len(examples)))
        features = convert_examples_to_features_gpt2(
            examples,
            label_list,
            args.input_max_seq_length,
            tokenizer,
            cls_token_at_end=True,  # xlnet has a cls token at the end
            sep_token_extra=bool(args.model_type in ['roberta']),
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(
                args.model_type in ['xlnet']),  # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
            distractor=True)
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
    all_output_ids = torch.tensor(select_field(features, 'output_ids'),
                                  dtype=torch.long)
    all_mc_output_ids = torch.tensor(select_field(features, 'mc_output_ids'),
                                     dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'),
                                  dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'),
                                   dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_output_ids, all_input_mask,
                            all_segment_ids, all_label_ids, all_mc_output_ids)
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
    parser.add_argument("--model_type",
                        default='gpt2',
                        type=str,
                        help="Model type selected in the list: " +
                        ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(ALL_MODELS))
    parser.add_argument(
        "--task_name",
        default="commonsenseqa",
        type=str,
        help="The name of the task to train selected in the list: " +
        ", ".join(processors.keys()))
    parser.add_argument(
        "--output_dir",
        default="/scratch/yyv959/commonsenseqa/outputs/gpt2/qa/",
        type=str,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )

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
        "--input_max_seq_length",
        default=70,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument(
        "--output_max_seq_length",
        default=10,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help='Whether to run test on the test set')
    parser.add_argument(
        "--evaluate_during_training",
        action='store_true',
        help="Rul evaluation during training at each logging step.")
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size",
                        default=2,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=32,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=8,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-6,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument("--warmup_ratio",
                        default=0.0,
                        type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--linear_decay",
                        action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--logging_steps',
                        type=int,
                        default=609,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps',
                        type=int,
                        default=5000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action='store_true',
        help=
        "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
    )
    parser.add_argument(
        "--reverse",
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
    parser.add_argument(
        '--multi_task',
        action='store_true',
        help="Overwrite the cached training and evaluation sets")
    parser.add_argument(
        '--mc',
        action='store_true',
        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument(
        '--fp16',
        action='store_true',
        help=
        "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
    )
    parser.add_argument(
        '--fp16_opt_level',
        type=str,
        default='O1',
        help=
        "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip',
                        type=str,
                        default='',
                        help="For distant debugging.")
    parser.add_argument('--server_port',
                        type=str,
                        default='',
                        help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir
    ) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            .format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port),
                            redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1),
        args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training will download model & vocab

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

    if args.local_rank == 0:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training will download model & vocab

    tokenizer.add_special_tokens({'pad_token': "[PAD]"})
    tokenizer.add_special_tokens({'cls_token': "[CLS]"})
    #print(len(tokenizer))

    #print(len(tokenizer))

    #x = input("stop")
    ##tokenizer.add_special_tokens({'sep_token': "[SEP]"})
    #print(model.transformer.generator.weight.size())
    #print(model.transformer.wte.weight[0,:10])
    model.transformer._resize_token_embeddings(len(tokenizer))
    #tokenizer.encoder.update(tokenizer.added_tokens_encoder)
    #tokenizer.decoder.update(tokenizer.added_tokens_decoder)
    #print(model.transformer.wte.weight[0,:10])
    #print(model.transformer.generator.weight.size())
    model.transformer.tie_weights()
    #print(model.transformer.generator.weight[0,:10])
    #print(model.transformer.generator.weight.size())
    model.transformer.config.vocab_size += 2
    #print(model.transformer.wte.weight.size())
    #print(model.generator.weight.size())
    #quit()
    #model.config.vocab_size += 3
    model.add_tokenizer(tokenizer)
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args,
                                                args.task_name,
                                                tokenizer,
                                                evaluate=False)
        global_step, tr_loss, best_steps = train(args, train_dataset, model,
                                                 tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step,
                    tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1
                          or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(
            model,
            'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
        model.add_tokenizer(tokenizer)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        #if not args.do_train:
        #    args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(
                    glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME,
                              recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                '-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.add_tokenizer(tokenizer)
            model.to(args.device)
            result = evaluate(args,
                              model,
                              tokenizer,
                              prefix=global_step,
                              generation=False)
            result = dict(
                (k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_test and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        # if args.eval_all_checkpoints: # can not use this to do test!!
        #     checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        #     logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                '-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args,
                              model,
                              tokenizer,
                              prefix=global_step,
                              test=True)
            result = dict(
                (k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
    if best_steps:
        logger.info("best steps of eval acc is the following checkpoints: %s",
                    best_steps)
    return results


if __name__ == "__main__":
    main()
