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
""" BERT multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

from __future__ import absolute_import, division, print_function

import logging
import os
import sys
from io import open
import json
import csv
import glob
import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""
    def __init__(self,
                 example_id,
                 question,
                 contexts,
                 endings,
                 concept=None,
                 label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (qustion).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label
        self.concept = concept

class InputExampleIS(object):
    """A single training/test example for multiple choice"""
    def __init__(self,
                 example_id,
                 question,
                 contexts,
                 endings,
                 concept=None,
                 weight=None,
                 label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (qustion).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label
        self.weight = weight
        self.concept = concept



class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [{
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids
        } for _, input_ids, input_mask, segment_ids in choices_features]
        self.label = label


class InputFeaturesIS(object):
    def __init__(self, example_id, choices_features, label, weight):
        self.example_id = example_id
        self.choices_features = [{
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids
        } for _, input_ids, input_mask, segment_ids in choices_features]
        self.label = label

        self.weight = weight

class InputFeaturesGPT2(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [{
            'input_ids': input_ids,
            'output_ids': output_ids,
            'mc_output_ids': mc_output_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids
        } for _, input_ids, output_ids, mc_output_ids, input_mask, segment_ids
                                 in choices_features]
        self.label = label


class InputFeaturesGPT2V3(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [{
            'input_ids': input_ids,
            'output_ids': output_ids,
            'mc_output_ids': mc_output_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            "a_mask": a_mask,
            "b_mask": b_mask
        } for _, input_ids, output_ids, mc_output_ids, input_mask, segment_ids,
                                 a_mask, b_mask in choices_features]
        self.label = label


class InputGenerativeFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [{
            'input_ids': input_ids,
            'output_ids': output_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids
        } for _, _, input_ids, output_ids, input_mask, segment_ids in
                                 choices_features]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, 'train/high')
        middle = os.path.join(data_dir, 'train/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, 'dev/high')
        middle = os.path.join(data_dir, 'dev/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, 'test/high')
        middle = os.path.join(data_dir, 'test/middle')
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, 'test')

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, 'r', encoding='utf-8') as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw['answers'][i]) - ord('A'))
                question = data_raw['questions'][i]
                options = data_raw['options'][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article
                                  ],  # this is not efficient but convenient
                        endings=[
                            options[0], options[1], options[2], options[3]
                        ],
                        label=truth))
        return examples


class CommonsenseqafakeProcessor(DataProcessor):
    """Processor for the Commonsenseqa data set."""
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train_fake_100000.csv")),
            "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!")
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        examples = [
            InputExample(
                example_id=line[0],
                question="",  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[1], line[1], line[1], line[1], line[1]],
                endings=[line[3], line[4], line[5], line[6], line[7]],
                label="0")
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples





class CommonsenseqaProcessor(DataProcessor):
    """Processor for the Commonsenseqa data set."""
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))

        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        examples = [
            InputExample(
                example_id=line[0],
                question="",  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[1], line[1], line[1], line[1], line[1]],
                endings=[line[3], line[4], line[5], line[6], line[7]],
                concept=[line[2], line[2], line[2], line[2], line[2]],
                label="0")
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples



class ARCProcessor(DataProcessor):
    """Processor for the Commonsenseqa data set."""
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))

        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        examples = [
            InputExample(
                example_id=line[0],
                question="",  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[1], line[1], line[1], line[1]],
                endings=[line[3], line[4], line[5], line[6]],
                concept=[line[2], line[2], line[2], line[2]],
                label="0")
            for line in lines[1:] if len(line) > 6  # we skip the line with the column names
        ]

        return examples


class WinograndeProcessor(DataProcessor):
    """Processor for the Commonsenseqa data set."""
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))

        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        examples = [
            InputExample(
                example_id=line[0],
                question="",  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[1], line[1]],
                endings=[line[3], line[4]],
                concept=[line[2], line[2]],
                label="0")
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples







class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))

        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        examples = [
            InputExample(
                example_id=line[0],
                question="",  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[1], line[1], line[1], line[1]],
                endings=[line[3], line[4], line[5], line[6]],
                concept=[line[2], line[2], line[2], line[2]],
                label="0")
            for line in lines[1:] if len(line) > 6 # we skip the line with the column names
        ]

        return examples

class RankProcessor(DataProcessor):
    """Processor for the SWAG data set."""
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev_rank.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!")
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(13542)]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != 'label':
            raise ValueError(
                "For training, the input file must contain a label column.")

        examples = [
            InputExample(
                example_id=line[0],
                question="",  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[1]],
                endings=[""],
                label=line[2])
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples






def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 sep_token_extra=False,
                                 pad_token_segment_id=0,
                                 pad_on_left=False,
                                 pad_token=0,
                                 mask_padding_with_zero=True,
                                 mask_question=False,
                                 pad_qa=False):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples),
                                         desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(
                zip(example.contexts, example.endings)):
            if pad_qa:
                context = 'Q: ' + context
                ending = 'A: ' + ending
                tokens_b = tokenizer.tokenize(ending)
            else:
                ending = "I " + ending
                tokens_b = tokenizer.tokenize(ending)
                tokens_b = tokens_b[1:]

            tokens_a = tokenizer.tokenize(context)


                # you can add seq token between quesiotn and ending. This does not make too much difference.
                # tokens_b = tokenizer.tokenize(example.question)
                # tokens_b += [sep_token]
                # if sep_token_extra:
                #     tokens_b += [sep_token]
                # tokens_b += tokenizer.tokenize(ending)
            #if pad_qa:
            #    tokens_a = ["Q:"] + tokens_a
            #    tokens_b = ["A:"] + tokens_b
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b,
                               max_seq_length - special_tokens_count)
            if mask_question:
                tokens_a = []
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = tokens_a + [sep_token]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] *
                              padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] *
                               padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] *
                                             padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            choices_features.append(
                (tokens, input_ids, input_mask, segment_ids))
        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (tokens, input_ids, input_mask,
                             segment_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(
                    str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(
                    map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(
                    map(str, segment_ids))))
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(example_id=example.example_id,
                          choices_features=choices_features,
                          label=label))

    return features


def convert_examples_to_features_is(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 sep_token_extra=False,
                                 pad_token_segment_id=0,
                                 pad_on_left=False,
                                 pad_token=0,
                                 mask_padding_with_zero=True,
                                 mask_question=False,
                                 pad_qa=False):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples),
                                         desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(
                zip(example.contexts, example.endings)):
            if pad_qa:
                context = 'Q: ' + context
                ending = 'A: ' + ending

            tokens_a = tokenizer.tokenize(context)
            tokens_b = None
            if example.question.find("_") != -1:
                #this is for cloze question
                tokens_b = tokenizer.tokenize(
                    example.question.replace("_", ending))
            else:
                tokens_b = tokenizer.tokenize(ending)
                # you can add seq token between quesiotn and ending. This does not make too much difference.
                # tokens_b = tokenizer.tokenize(example.question)
                # tokens_b += [sep_token]
                # if sep_token_extra:
                #     tokens_b += [sep_token]
                # tokens_b += tokenizer.tokenize(ending)
            #if pad_qa:
            #    tokens_a = ["Q:"] + tokens_a
            #    tokens_b = ["A:"] + tokens_b
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b,
                               max_seq_length - special_tokens_count)
            if mask_question:
                tokens_a = []
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = tokens_a + [sep_token]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] *
                              padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] *
                               padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] *
                                             padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            choices_features.append(
                (tokens, input_ids, input_mask, segment_ids))
        label = label_map[example.label]
        weight = float(example.weight)

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (tokens, input_ids, input_mask,
                             segment_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(
                    str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(
                    map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(
                    map(str, segment_ids))))
                logger.info("label: {}".format(label))

        features.append(
            InputFeaturesIS(example_id=example.example_id,
                          choices_features=choices_features,
                          label=label,
                          weight =weight))

    return features


def convert_examples_to_features_gpt2(examples,
                                      label_list,
                                      max_seq_length,
                                      tokenizer,
                                      cls_token_at_end=True,
                                      cls_token_segment_id=1,
                                      sequence_a_segment_id=0,
                                      sequence_b_segment_id=1,
                                      sep_token_extra=False,
                                      pad_token_segment_id=0,
                                      pad_on_left=False,
                                      mask_padding_with_zero=True,
                                      reverse=False,
                                      distractor=False):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token_id
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples),
                                         desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(
                zip(example.contexts, example.endings)):
            context = "Q: " + context
            ending = "A: " + ending
            tokens_a = tokenizer.tokenize(context)
            tokens_b =tokenizer.tokenize(ending)

                # you can add seq token between quesiotn and ending. This does not make too much difference.
                # tokens_b = tokenizer.tokenize(example.question)
                # tokens_b += [sep_token]
                # if sep_token_extra:
                #     tokens_b += [sep_token]
                # tokens_b += tokenizer.tokenize(ending)
            tokens_a = tokens_a
            tokens_b = tokens_b
            special_tokens_count = 2
            _truncate_seq_pair(tokens_a, tokens_b,
                               max_seq_length - special_tokens_count)

            if reverse:
                tmp = tokens_a
                tokens_a = tokens_b
                tokens_b = tmp
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            #tokens = tokens_a + [sep_token]

            tokens = tokens_a

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)
            question_length = len(tokens) + 2
            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            output_ids = input_ids[1:]
            output_ids = np.array(output_ids)
            output_ids[0:(question_length - 1)] = -1
            output_ids[-1] = -1
            output_ids = list(output_ids)
            mc_output_ids = output_ids

            if ending_idx != label_map[example.label] and (not distractor):
                output_ids = [-1] * (len(input_ids) - 1)
            if ending_idx == label_map[example.label] and distractor:
                output_ids = [-1] * (len(input_ids) - 1)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] *
                              padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] *
                               padding_length) + segment_ids
                output_ids = ([-1] * padding_length) + output_ids
                mc_output_ids = ([-1] * padding_length) + mc_output_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] *
                                             padding_length)
                output_ids = output_ids + ([-1] * padding_length)
                mc_output_ids = mc_output_ids + ([-1] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(output_ids) == (max_seq_length - 1)
            assert len(mc_output_ids) == (max_seq_length - 1)
            choices_features.append((tokens, input_ids, output_ids,
                                     mc_output_ids, input_mask, segment_ids))
        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (tokens, input_ids, output_ids, mc_output_ids,
                             input_mask,
                             segment_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(
                    str, input_ids))))
                #logger.info("input_mask: {}".format(' '.join(
                #    map(str, input_mask))))
                #logger.info("segment_ids: {}".format(' '.join(
                #    map(str, segment_ids))))
                logger.info("output_ids: {}".format(' '.join(
                    map(str, output_ids))))

        #x = input("pause")
        features.append(
            InputFeaturesGPT2(example_id=example.example_id,
                              choices_features=choices_features,
                              label=label))

    return features


def convert_examples_to_features_gpt2_v5(examples,
                                      label_list,
                                      max_seq_length,
                                      tokenizer,
                                      cls_token_at_end=True,
                                      cls_token_segment_id=1,
                                      sequence_a_segment_id=0,
                                      sequence_b_segment_id=1,
                                      sep_token_extra=False,
                                      pad_token_segment_id=0,
                                      pad_on_left=False,
                                      mask_padding_with_zero=True,
                                      distractor=False):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token_id
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples),
                                         desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(
                zip(example.contexts, example.endings)):
            context_tmp = "A: " + ending
            ending_tmp = "Q: " + context
            context = context_tmp
            ending = ending_tmp
            tokens_a = tokenizer.tokenize(context)
            tokens_b = None
            if example.question.find("_") != -1:
                #this is for cloze question
                tokens_b = tokenizer.tokenize(
                    example.question.replace("_", ending))
            else:
                tokens_b = tokenizer.tokenize(example.question + " " + ending)
                # you can add seq token between quesiotn and ending. This does not make too much difference.
                # tokens_b = tokenizer.tokenize(example.question)
                # tokens_b += [sep_token]
                # if sep_token_extra:
                #     tokens_b += [sep_token]
                # tokens_b += tokenizer.tokenize(ending)
            tokens_a = tokens_a
            tokens_b = tokens_b
            special_tokens_count = 2
            _truncate_seq_pair(tokens_a, tokens_b,
                               max_seq_length - special_tokens_count)


            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            #tokens = tokens_a + [sep_token]

            tokens = tokens_a

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)
            question_length = len(tokens) + 2
            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            output_ids = input_ids[1:]
            output_ids = np.array(output_ids)
            #output_ids[0:(question_length - 1)] = -1
            output_ids[-1] = -1
            output_ids = list(output_ids)
            mc_output_ids = output_ids

            if ending_idx != label_map[example.label] and (not distractor):
                output_ids = [-1] * (len(input_ids) - 1)
            if ending_idx == label_map[example.label] and distractor:
                output_ids = [-1] * (len(input_ids) - 1)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] *
                              padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] *
                               padding_length) + segment_ids
                output_ids = ([-1] * padding_length) + output_ids
                mc_output_ids = ([-1] * padding_length) + mc_output_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] *
                                             padding_length)
                output_ids = output_ids + ([-1] * padding_length)
                mc_output_ids = mc_output_ids + ([-1] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(output_ids) == (max_seq_length - 1)
            assert len(mc_output_ids) == (max_seq_length - 1)
            choices_features.append((tokens, input_ids, output_ids,
                                     mc_output_ids, input_mask, segment_ids))
        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (tokens, input_ids, output_ids, mc_output_ids,
                             input_mask,
                             segment_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(
                    str, input_ids))))
                #logger.info("input_mask: {}".format(' '.join(
                #    map(str, input_mask))))
                #logger.info("segment_ids: {}".format(' '.join(
                #    map(str, segment_ids))))
                logger.info("output_ids: {}".format(' '.join(
                    map(str, output_ids))))

        #x = input("pause")
        features.append(
            InputFeaturesGPT2(example_id=example.example_id,
                              choices_features=choices_features,
                              label=label))

    return features






def convert_examples_to_features_gpt2_qg(examples,
                                         label_list,
                                         max_seq_length,
                                         tokenizer,
                                         cls_token_at_end=True,
                                         cls_token_segment_id=1,
                                         sequence_a_segment_id=0,
                                         sequence_b_segment_id=1,
                                         sep_token_extra=False,
                                         pad_token_segment_id=0,
                                         pad_on_left=False,
                                         mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token_id
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples),
                                         desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(
                zip(example.contexts, example.concept)):
            tokens_a = tokenizer.tokenize(context)
            tokens_b = None
            if example.question.find("_") != -1:
                #this is for cloze question
                tokens_b = tokenizer.tokenize(
                    example.question.replace("_", ending))
            else:
                tokens_b = tokenizer.tokenize(example.question + " " + ending)
                # you can add seq token between quesiotn and ending. This does not make too much difference.
                # tokens_b = tokenizer.tokenize(example.question)
                # tokens_b += [sep_token]
                # if sep_token_extra:
                #     tokens_b += [sep_token]
                # tokens_b += tokenizer.tokenize(ending)
            tokens_a = tokens_a
            tokens_b = tokens_b
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b,
                               max_seq_length - special_tokens_count)

            tmp = tokens_a
            tokens_a = tokens_b
            tokens_b = tmp
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            #tokens = tokens_a + [sep_token]

            tokens = tokens_a + ["Q:"]

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)
            concept_length = len(tokens)
            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            output_ids = input_ids[1:]
            output_ids = np.array(output_ids)
            output_ids[0:(concept_length - 1)] = -1
            output_ids[-1] = -1
            output_ids = list(output_ids)
            mc_output_ids = output_ids

            if ending_idx != label_map[example.label]:
                output_ids = [-1] * (len(input_ids) - 1)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] *
                              padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] *
                               padding_length) + segment_ids
                output_ids = ([-1] * padding_length) + output_ids
                mc_output_ids = ([-1] * padding_length) + mc_output_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] *
                                             padding_length)
                output_ids = output_ids + ([-1] * padding_length)
                mc_output_ids = mc_output_ids + ([-1] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(output_ids) == (max_seq_length - 1)
            assert len(mc_output_ids) == (max_seq_length - 1)
            choices_features.append((tokens, input_ids, output_ids,
                                     mc_output_ids, input_mask, segment_ids))
        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (tokens, input_ids, output_ids, mc_output_ids,
                             input_mask,
                             segment_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(
                    str, input_ids))))
                #logger.info("input_mask: {}".format(' '.join(
                #    map(str, input_mask))))
                #logger.info("segment_ids: {}".format(' '.join(
                #    map(str, segment_ids))))
                logger.info("output_ids: {}".format(' '.join(
                    map(str, output_ids))))

        #x = input("pause")
        features.append(
            InputFeaturesGPT2(example_id=example.example_id,
                              choices_features=choices_features,
                              label=label))

    return features


def convert_examples_to_features_gpt2_v3(examples,
                                         label_list,
                                         max_seq_length,
                                         tokenizer,
                                         cls_token_at_end=True,
                                         cls_token_segment_id=1,
                                         sequence_a_segment_id=0,
                                         sequence_b_segment_id=1,
                                         sep_token_extra=False,
                                         pad_token_segment_id=0,
                                         pad_on_left=False,
                                         mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token_id
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples),
                                         desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(
                zip(example.contexts, example.endings)):
            tokens_a = tokenizer.tokenize(context)
            tokens_b = None
            if example.question.find("_") != -1:
                #this is for cloze question
                tokens_b = tokenizer.tokenize(
                    example.question.replace("_", ending))
            else:
                tokens_b = tokenizer.tokenize(example.question + " " + ending)
                # you can add seq token between quesiotn and ending. This does not make too much difference.
                # tokens_b = tokenizer.tokenize(example.question)
                # tokens_b += [sep_token]
                # if sep_token_extra:
                #     tokens_b += [sep_token]
                # tokens_b += tokenizer.tokenize(ending)

            special_tokens_count = 4
            _truncate_seq_pair(tokens_a, tokens_b,
                               max_seq_length - special_tokens_count)

            tmp = tokens_a
            tokens_a = tokens_b
            tokens_b = tmp
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = [sep_token] + tokens_a + [sep_token]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [sequence_a_segment_id] * len(tokens)
            a_length = len(tokens)
            b_length = len(tokens_b) + 1
            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            output_ids = input_ids[1:]
            output_ids = np.array(output_ids)
            #output_ids[0:(a_length - 1)] = -1
            output_ids[-1] = -1
            output_ids = list(output_ids)
            mc_output_ids = output_ids
            if ending_idx != label_map[example.label]:
                output_ids = [-1] * (len(input_ids) - 1)
            a_mask = [1] * (a_length - 1) + [0] * b_length + [0]
            b_mask = [0] * (a_length - 1) + [1] * b_length + [0]
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] *
                              padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] *
                               padding_length) + segment_ids
                output_ids = ([-1] * padding_length) + output_ids
                mc_output_ids = ([-1] * padding_length) + mc_output_ids
                a_mask = ([0] * padding_length) + a_mask
                b_mask = ([0] * padding_length) + b_mask

            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] *
                                             padding_length)
                output_ids = output_ids + ([-1] * padding_length)
                mc_output_ids = mc_output_ids + ([-1] * padding_length)
                a_mask = a_mask + ([0] * padding_length)
                b_mask = b_mask + ([0] * padding_length)
            #print(len(input_ids))
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(output_ids) == (max_seq_length - 1)
            assert len(mc_output_ids) == (max_seq_length - 1)
            assert len(a_mask) == (max_seq_length - 1)
            assert len(b_mask) == (max_seq_length - 1)

            choices_features.append(
                (tokens, input_ids, output_ids, mc_output_ids, input_mask,
                 segment_ids, a_mask, b_mask))
        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (tokens, input_ids, output_ids, mc_output_ids,
                             input_mask, segment_ids, a_mask,
                             b_mask) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(
                    str, input_ids))))
                #logger.info("input_mask: {}".format(' '.join(
                #    map(str, input_mask))))
                #logger.info("segment_ids: {}".format(' '.join(
                #    map(str, segment_ids))))
                logger.info("output_ids: {}".format(' '.join(
                    map(str, output_ids))))

        #x = input("pause")
        features.append(
            InputFeaturesGPT2V3(example_id=example.example_id,
                                choices_features=choices_features,
                                label=label))

    return features


def convert_examples_to_generative_features(examples,
                                            label_list,
                                            input_max_seq_length,
                                            output_max_seq_length,
                                            tokenizer,
                                            cls_token_at_end=False,
                                            cls_token='[CLS]',
                                            cls_token_segment_id=1,
                                            sep_token='[SEP]',
                                            sequence_a_segment_id=0,
                                            sequence_b_segment_id=1,
                                            sep_token_extra=False,
                                            pad_token_segment_id=0,
                                            pad_on_left=False,
                                            pad_token=0,
                                            mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples),
                                         desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(
                zip(example.contexts, example.endings)):
            #if ending_idx != label_map[example.label]:
            #    continue
            context = "Q: " +  context
            tokens_a = tokenizer.tokenize(context)
            tokens_b = None
            if example.question.find("_") != -1:
                #this is for cloze question
                tokens_b = tokenizer.tokenize(
                    example.question.replace("_", ending))
            else:
                tokens_b = tokenizer.tokenize(example.question + " " + ending)
                # you can add seq token between quesiotn and ending. This does not make too much difference.
                # tokens_b = tokenizer.tokenize(example.question)
                # tokens_b += [sep_token]
                # if sep_token_extra:
                #     tokens_b += [sep_token]
                # tokens_b += tokenizer.tokenize(ending)

            special_tokens_count = 4 if sep_token_extra else 1
            _truncate_seq_pair(tokens_a, [],
                               input_max_seq_length - special_tokens_count)
            _truncate_seq_pair(tokens_b, [],
                               output_max_seq_length - special_tokens_count)
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens_a =   tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens_a)
            tokens_b =   tokens_b + [sep_token]

            input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
            output_ids = tokenizer.convert_tokens_to_ids(tokens_b)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            input_padding_length = input_max_seq_length - len(input_ids)
            output_padding_length = output_max_seq_length - len(output_ids)

            input_ids = input_ids + ([pad_token] * input_padding_length)
            output_ids = output_ids + ([pad_token] * output_padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] *
                                       input_padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] *
                                         input_padding_length)

            assert len(input_ids) == input_max_seq_length
            assert len(input_mask) == input_max_seq_length
            assert len(segment_ids) == input_max_seq_length
            assert len(output_ids) == output_max_seq_length
            choices_features.append((tokens_a, tokens_b, input_ids, output_ids,
                                     input_mask, segment_ids))
        label = label_map[example.label]
        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (tokens, _, input_ids, output_ids, _,
                             _) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(
                    str, input_ids))))
                #logger.info("input_mask: {}".format(' '.join(
                #    map(str, input_mask))))
                #logger.info("segment_ids: {}".format(' '.join(
                #    map(str, segment_ids))))
                #logger.info("output_ids: {}".format(' '.join(
                #    map(str, output_ids))))
        features.append(
            InputGenerativeFeatures(example_id=example.example_id,
                                    choices_features=choices_features,
                                    label=label))

    return features


def convert_examples_to_generative_features_gpt(examples,
                                                label_list,
                                                input_max_seq_length,
                                                output_max_seq_length,
                                                tokenizer,
                                                cls_token_at_end=False,
                                                cls_token='[CLS]',
                                                cls_token_segment_id=1,
                                                sep_token='[SEP]',
                                                sequence_a_segment_id=0,
                                                sequence_b_segment_id=1,
                                                sep_token_extra=False,
                                                pad_token_segment_id=0,
                                                pad_on_left=False,
                                                pad_token=0,
                                                mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples),
                                         desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(
                zip(example.contexts, example.endings)):
            if ending_idx != label_map[example.label]:
                continue
            tokens_a = tokenizer.tokenize(context)
            tokens_b = None
            if example.question.find("_") != -1:
                #this is for cloze question
                tokens_b = tokenizer.tokenize(
                    example.question.replace("_", ending))
            else:
                tokens_b = tokenizer.tokenize(example.question + " " + ending)
                # you can add seq token between quesiotn and ending. This does not make too much difference.
                # tokens_b = tokenizer.tokenize(example.question)
                # tokens_b += [sep_token]
                # if sep_token_extra:
                #     tokens_b += [sep_token]
                # tokens_b += tokenizer.tokenize(ending)

            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, [],
                               input_max_seq_length - special_tokens_count)
            _truncate_seq_pair(tokens_b, [],
                               output_max_seq_length - special_tokens_count)
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens_a = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens_a)
            tokens_b = [cls_token] + tokens_b + [sep_token]

            input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
            output_ids = tokenizer.convert_tokens_to_ids(tokens_b)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            input_padding_length = input_max_seq_length - len(input_ids)
            output_padding_length = output_max_seq_length - len(output_ids)

            input_ids = input_ids + ([pad_token] * input_padding_length)
            output_ids = output_ids + ([pad_token] * output_padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] *
                                       input_padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] *
                                         input_padding_length)

            assert len(input_ids) == input_max_seq_length
            assert len(input_mask) == input_max_seq_length
            assert len(segment_ids) == input_max_seq_length
            assert len(output_ids) == output_max_seq_length
            choices_features.append((tokens_a, tokens_b, input_ids, output_ids,
                                     input_mask, segment_ids))
        label = label_map[example.label]

        features.append(
            InputGenerativeFeatures(example_id=example.example_id,
                                    choices_features=choices_features,
                                    label=label))

    return features


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
            logger.info(
                'Attention! you are removing from token_b (swag task is ok). '
                'If you are training ARC and RACE (you are poping question + options), '
                'you need to try to use a bigger max seq length!')
            tokens_b.pop()


processors = {
    "race": RaceProcessor,
    "swag": SwagProcessor,
    "arc": ARCProcessor,
    "winogrande": WinograndeProcessor,
    "commonsenseqa": CommonsenseqaProcessor,
    "commonsenseqafake": CommonsenseqafakeProcessor,
    "rank": RankProcessor
}

GLUE_TASKS_NUM_LABELS = {
    "race",
    4,
    "swag",
    4,
    "arc",
    4,
    "commonsenseqa",
    5,
    "rank",
    13542,
    "winogrande",
    2
}