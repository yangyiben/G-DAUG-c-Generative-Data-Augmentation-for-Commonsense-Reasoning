from transformers import BertModel
from transformers.modeling_bert import BertOnlyMLMHead
from transformers.modeling_roberta import RobertaLMHead, RobertaEmbeddings, RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaModel
from transformers import BertPreTrainedModel
from transformers import BertPreTrainedModel, GPT2PreTrainedModel, GPT2Model

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import GPT2Model
from transformers.modeling_utils import SequenceSummary




def top_k_top_p_filtering(logits,
                          top_k=0,
                          top_p=0.0,
                          filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch, vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """

    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits,
                                                top_k)[0][..., -1, None]
        for i in range(logits.size(0)):
            logits[i, indices_to_remove[i, :]] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = [
            sorted_indices[i, sorted_indices_to_remove[i, :]]
            for i in range(sorted_indices.size(0))
        ]
        for i in range(logits.size(0)):
            logits[i, indices_to_remove[i]] = filter_value
    return logits



class GenerativeGPT2QG(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GenerativeGPT2QG, self).__init__(config)
        self.transformer = GPT2ModelQG(config)
        self.loss = nn.CrossEntropyLoss(weight=None,
                                        size_average=None,
                                        ignore_index=50257,
                                        reduce=None,
                                        reduction='none')
        self.init_weights()
        self.transformer.tie_weights()

    def add_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        decoded = self.transformer(input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)
        target = input_ids[:, 1:]
        data_loss = self.loss(decoded.view(-1, self.config.vocab_size),
                              target.contiguous().view(-1)).view(
                                  input_ids.size(0), -1)
        return data_loss

    def generate(self, batch_size, max_len, sample=True, tmp=1, top_p=0.9):
        starting_word = "Q"
        stopping_word = self.tokenizer.eos_token
        res = []

        starting_word = self.tokenizer.convert_tokens_to_ids([starting_word
                                                              ])[0]
        stopping_word = self.tokenizer.convert_tokens_to_ids([stopping_word
                                                              ])[0]
        model_device = next(self.transformer.parameters()).device
        word = next(self.transformer.parameters()).new_zeros(
            batch_size, 1).fill_(starting_word).long()
        sentence_log_prob = torch.zeros_like(word, dtype=torch.float).view(-1)
        while word.size(0) != 0 and word.size(1) < max_len:
            #print(input_ids.size())
            #print(word.size())
            decoded = self.transformer(word,
                                       token_type_ids=None,
                                       attention_mask=torch.ones_like(word),
                                       generation=True)
            logits = decoded / tmp
            if sample:
                filtered_logits = top_k_top_p_filtering(logits, top_p=top_p)
                tmp_dist = F.softmax(filtered_logits, -1)
                topi = torch.multinomial(tmp_dist, 1)
                word_log_prob = F.nll_loss(
                    torch.log(tmp_dist), topi.view(-1), reduction='none') * -1
            else:
                tmp_dist = F.softmax(logits, -1)
                p, topi = torch.topk(tmp_dist, 1)
                word_log_prob = torch.log(p.view(-1))
            sentence_log_prob += word_log_prob
            word = torch.cat([word, topi], 1)
            stop_idx = (topi.view(-1) == stopping_word).nonzero().view(-1)
            if stop_idx.size(0) != 0:
                word_continue = word.index_select(
                    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                sentence_log_prob_s = sentence_log_prob.index_select(
                    0, stop_idx)
                sentence_log_prob_c = sentence_log_prob.index_select(
                    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                sentence_log_prob = sentence_log_prob_c
                #token_type_ids = token_type_ids.index_select(
                #    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                word_stop = word.index_select(0, stop_idx)
                #word_stop_pad = word_stop.new_zeros(word_stop.size(0), max_len).fill_(self.tokenizer.pad_token_id)
                #word_stop_pad[:, :word_stop.size(1)] = word_stop
                word = word_continue

                res.append((word_stop, sentence_log_prob_s))
        if word.size(0) > 0:
            res.append((word, sentence_log_prob))
        input_ids = [it[0] for it in res]
        sentences = []
        for it in input_ids:
            sentences += list(torch.unbind(it))

        sentence_log_prob = torch.cat([it[1] for it in res], dim=0)

        res = []

        for i, sent in enumerate(sentences):

            sent = self.tokenizer.decode(
                list(sent.cpu().numpy()),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True).replace("Q:", "").replace(
                    "[PAD]", "")
            conf = sentence_log_prob[i].item()
            #truth = " ".join(truth)
            #truth = truth.replace("[PAD]", "")
            #truth = truth.replace(" ##", "")
            #truth = truth.replace("##", "")
            #print(sent[0])
            res.append(sent + "\t" + str(conf))
        random.shuffle(res)
        return res


class GenerativeGPT2QGWinogrande(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GenerativeGPT2QGWinogrande, self).__init__(config)
        self.transformer = GPT2ModelQG(config)
        self.loss = nn.CrossEntropyLoss(weight=None,
                                        size_average=None,
                                        ignore_index=50257,
                                        reduce=None,
                                        reduction='none')
        self.init_weights()
        self.transformer.tie_weights()

    def add_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        decoded = self.transformer(input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)
        target = input_ids[:, 1:]
        data_loss = self.loss(decoded.view(-1, self.config.vocab_size),
                              target.contiguous().view(-1)).view(
                                  input_ids.size(0), -1)
        return data_loss

    def generate(self, batch_size, max_len, sample=True, tmp=1, top_p=0.9):
        starting_word = "\n"
        stopping_word = "\n"
        res = []

        starting_word = self.tokenizer.convert_tokens_to_ids([starting_word
                                                              ])[0]
        stopping_word = self.tokenizer.convert_tokens_to_ids([stopping_word
                                                              ])[0]
        model_device = next(self.transformer.parameters()).device
        word = next(self.transformer.parameters()).new_zeros(
            batch_size, 1).fill_(starting_word).long()
        sentence_log_prob = torch.zeros_like(word, dtype=torch.float).view(-1)
        while word.size(0) != 0 and word.size(1) < max_len:
            #print(input_ids.size())
            #print(word.size())
            decoded = self.transformer(word,
                                       token_type_ids=None,
                                       attention_mask=torch.ones_like(word),
                                       generation=True)
            logits = decoded / tmp
            if sample:
                filtered_logits = top_k_top_p_filtering(logits, top_p=top_p)
                tmp_dist = F.softmax(filtered_logits, -1)
                topi = torch.multinomial(tmp_dist, 1)
                word_log_prob = F.nll_loss(
                    torch.log(tmp_dist), topi.view(-1), reduction='none') * -1
            else:
                tmp_dist = F.softmax(logits, -1)
                p, topi = torch.topk(tmp_dist, 1)
                word_log_prob = torch.log(p.view(-1))
            sentence_log_prob += word_log_prob
            word = torch.cat([word, topi], 1)
            stop_idx = (topi.view(-1) == stopping_word).nonzero().view(-1)
            if stop_idx.size(0) != 0:
                word_continue = word.index_select(
                    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                sentence_log_prob_s = sentence_log_prob.index_select(
                    0, stop_idx)
                sentence_log_prob_c = sentence_log_prob.index_select(
                    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                sentence_log_prob = sentence_log_prob_c
                #token_type_ids = token_type_ids.index_select(
                #    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                word_stop = word.index_select(0, stop_idx)
                #word_stop_pad = word_stop.new_zeros(word_stop.size(0), max_len).fill_(self.tokenizer.pad_token_id)
                #word_stop_pad[:, :word_stop.size(1)] = word_stop
                word = word_continue

                res.append((word_stop, sentence_log_prob_s))
        if word.size(0) > 0:
            res.append((word, sentence_log_prob))
        input_ids = [it[0] for it in res]
        sentences = []
        for it in input_ids:
            sentences += list(torch.unbind(it))

        sentence_log_prob = torch.cat([it[1] for it in res], dim=0)

        res = []

        for i, sent in enumerate(sentences):

            sent = self.tokenizer.decode(
                list(sent.cpu().numpy()),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True).replace("\n ", "").replace(" \n", "").replace(
                    "[PAD]", "")
            conf = sentence_log_prob[i].item()
            #truth = " ".join(truth)
            #truth = truth.replace("[PAD]", "")
            #truth = truth.replace(" ##", "")
            #truth = truth.replace("##", "")
            #print(sent[0])
            res.append(sent + "\t" + str(conf))
        random.shuffle(res)
        return res

    def generate_context(self, batch_size, max_len, sample=True, tmp=1, top_p=0.9):
        starting_word = "\n"
        stopping_word = self.tokenizer.tokenize("a _ b")[1]
        res = []

        starting_word = self.tokenizer.convert_tokens_to_ids([starting_word
                                                              ])[0]
        stopping_word = self.tokenizer.convert_tokens_to_ids([stopping_word
                                                              ])[0]
        model_device = next(self.transformer.parameters()).device
        word = next(self.transformer.parameters()).new_zeros(
            batch_size, 1).fill_(starting_word).long()
        sentence_log_prob = torch.zeros_like(word, dtype=torch.float).view(-1)
        while word.size(0) != 0 and word.size(1) < max_len:
            #print(input_ids.size())
            #print(word.size())
            decoded = self.transformer(word,
                                       token_type_ids=None,
                                       attention_mask=torch.ones_like(word),
                                       generation=True)
            logits = decoded / tmp
            if sample:
                filtered_logits = top_k_top_p_filtering(logits, top_p=top_p)
                tmp_dist = F.softmax(filtered_logits, -1)
                topi = torch.multinomial(tmp_dist, 1)
                word_log_prob = F.nll_loss(
                    torch.log(tmp_dist), topi.view(-1), reduction='none') * -1
            else:
                tmp_dist = F.softmax(logits, -1)
                p, topi = torch.topk(tmp_dist, 1)
                word_log_prob = torch.log(p.view(-1))
            sentence_log_prob += word_log_prob
            word = torch.cat([word, topi], 1)
            stop_idx = (topi.view(-1) == stopping_word).nonzero().view(-1)
            if stop_idx.size(0) != 0:
                word_continue = word.index_select(
                    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                sentence_log_prob_s = sentence_log_prob.index_select(
                    0, stop_idx)
                sentence_log_prob_c = sentence_log_prob.index_select(
                    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                sentence_log_prob = sentence_log_prob_c
                #token_type_ids = token_type_ids.index_select(
                #    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                word_stop = word.index_select(0, stop_idx)
                #word_stop_pad = word_stop.new_zeros(word_stop.size(0), max_len).fill_(self.tokenizer.pad_token_id)
                #word_stop_pad[:, :word_stop.size(1)] = word_stop
                word = word_continue

                res.append((word_stop, sentence_log_prob_s))
        if word.size(0) > 0:
            res.append((word, sentence_log_prob))
        input_ids = [it[0] for it in res]
        sentences = []
        for it in input_ids:
            sentences += list(torch.unbind(it))

        sentence_log_prob = torch.cat([it[1] for it in res], dim=0)

        res = []

        for i, sent in enumerate(sentences):

            sent = self.tokenizer.decode(
                list(sent.cpu().numpy()),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True).replace("\n ", "").replace(" \n", "").replace(
                    "[PAD]", "")
            conf = sentence_log_prob[i].item()
            #truth = " ".join(truth)
            #truth = truth.replace("[PAD]", "")
            #truth = truth.replace(" ##", "")
            #truth = truth.replace("##", "")
            #print(sent[0])
            res.append(sent + "\t" + str(conf))
        random.shuffle(res)
        return res


    def continue_generate(self,word,  max_len, sample=True, tmp=1, top_p=0.9):
        starting_word = "\n"
        stopping_word = "\n"
        res = []

        #starting_word = self.tokenizer.convert_tokens_to_ids([starting_word
        #                                                      ])[0]
        stopping_word = self.tokenizer.convert_tokens_to_ids([stopping_word
                                                              ])[0]
        model_device = next(self.transformer.parameters()).device
        #word = next(self.transformer.parameters()).new_zeros(
        #    batch_size, 1).fill_(starting_word).long()
        sentence_log_prob = torch.zeros(word.size(0),1, dtype=torch.float, device = model_device).view(-1)
        while word.size(0) != 0 and word.size(1) < max_len:
            #print(input_ids.size())
            #print(word.size())
            decoded = self.transformer(word,
                                       token_type_ids=None,
                                       attention_mask=torch.ones_like(word),
                                       generation=True)
            logits = decoded / tmp
            if sample:
                filtered_logits = top_k_top_p_filtering(logits, top_p=top_p)
                tmp_dist = F.softmax(filtered_logits, -1)
                topi = torch.multinomial(tmp_dist, 1)
                word_log_prob = F.nll_loss(
                    torch.log(tmp_dist), topi.view(-1), reduction='none') * -1
            else:
                tmp_dist = F.softmax(logits, -1)
                p, topi = torch.topk(tmp_dist, 1)
                word_log_prob = torch.log(p.view(-1))
            sentence_log_prob += word_log_prob
            word = torch.cat([word, topi], 1)
            stop_idx = (topi.view(-1) == stopping_word).nonzero().view(-1)
            if stop_idx.size(0) != 0:
                word_continue = word.index_select(
                    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                sentence_log_prob_s = sentence_log_prob.index_select(
                    0, stop_idx)
                sentence_log_prob_c = sentence_log_prob.index_select(
                    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                sentence_log_prob = sentence_log_prob_c
                #token_type_ids = token_type_ids.index_select(
                #    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                word_stop = word.index_select(0, stop_idx)
                #word_stop_pad = word_stop.new_zeros(word_stop.size(0), max_len).fill_(self.tokenizer.pad_token_id)
                #word_stop_pad[:, :word_stop.size(1)] = word_stop
                word = word_continue

                res.append((word_stop, sentence_log_prob_s))
        if word.size(0) > 0:
            res.append((word, sentence_log_prob))
        input_ids = [it[0] for it in res]
        sentences = []
        for it in input_ids:
            sentences += list(torch.unbind(it))

        sentence_log_prob = torch.cat([it[1] for it in res], dim=0)

        res = []

        for i, sent in enumerate(sentences):

            sent = self.tokenizer.decode(
                list(sent.cpu().numpy()),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True).replace("\n ", "").replace(" \n", "").replace(
                    "[PAD]", "")
            conf = sentence_log_prob[i].item()
            #truth = " ".join(truth)
            #truth = truth.replace("[PAD]", "")
            #truth = truth.replace(" ##", "")
            #truth = truth.replace("##", "")
            #print(sent[0])
            res.append(sent + "\t" + str(conf))
        random.shuffle(res)
        return res



class GenerativeGPT2QGSwag(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GenerativeGPT2QGSwag, self).__init__(config)
        self.transformer = GPT2ModelQG(config)
        self.loss = nn.CrossEntropyLoss(weight=None,
                                        size_average=None,
                                        ignore_index=50257,
                                        reduce=None,
                                        reduction='none')
        self.init_weights()
        self.transformer.tie_weights()

    def add_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        decoded = self.transformer(input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)
        target = input_ids[:, 1:]
        data_loss = self.loss(decoded.view(-1, self.config.vocab_size),
                              target.contiguous().view(-1)).view(
                                  input_ids.size(0), -1)
        return data_loss

    def generate(self, batch_size, max_len, sample=True, tmp=1, top_p=0.9):
        starting_word = "\n"
        stopping_word = "\n"
        res = []

        starting_word = self.tokenizer.convert_tokens_to_ids([starting_word
                                                              ])[0]
        stopping_word = self.tokenizer.convert_tokens_to_ids([stopping_word
                                                              ])[0]
        model_device = next(self.transformer.parameters()).device
        word = next(self.transformer.parameters()).new_zeros(
            batch_size, 1).fill_(starting_word).long()
        sentence_log_prob = torch.zeros_like(word, dtype=torch.float).view(-1)
        while word.size(0) != 0 and word.size(1) < max_len:
            #print(input_ids.size())
            #print(word.size())
            decoded = self.transformer(word,
                                       token_type_ids=None,
                                       attention_mask=torch.ones_like(word),
                                       generation=True)
            logits = decoded / tmp
            if sample:
                filtered_logits = top_k_top_p_filtering(logits, top_p=top_p)
                tmp_dist = F.softmax(filtered_logits, -1)
                topi = torch.multinomial(tmp_dist, 1)
                word_log_prob = F.nll_loss(
                    torch.log(tmp_dist), topi.view(-1), reduction='none') * -1
            else:
                tmp_dist = F.softmax(logits, -1)
                p, topi = torch.topk(tmp_dist, 1)
                word_log_prob = torch.log(p.view(-1))
            sentence_log_prob += word_log_prob
            word = torch.cat([word, topi], 1)
            stop_idx = (topi.view(-1) == stopping_word).nonzero().view(-1)
            if stop_idx.size(0) != 0:
                word_continue = word.index_select(
                    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                sentence_log_prob_s = sentence_log_prob.index_select(
                    0, stop_idx)
                sentence_log_prob_c = sentence_log_prob.index_select(
                    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                sentence_log_prob = sentence_log_prob_c
                #token_type_ids = token_type_ids.index_select(
                #    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                word_stop = word.index_select(0, stop_idx)
                #word_stop_pad = word_stop.new_zeros(word_stop.size(0), max_len).fill_(self.tokenizer.pad_token_id)
                #word_stop_pad[:, :word_stop.size(1)] = word_stop
                word = word_continue

                res.append((word_stop, sentence_log_prob_s))
        if word.size(0) > 0:
            res.append((word, sentence_log_prob))
        input_ids = [it[0] for it in res]
        sentences = []
        for it in input_ids:
            sentences += list(torch.unbind(it))

        sentence_log_prob = torch.cat([it[1] for it in res], dim=0)

        res = []

        for i, sent in enumerate(sentences):

            sent = self.tokenizer.decode(
                list(sent.cpu().numpy()),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True).replace("\n ", "").replace(" \n", "").replace(
                    "[PAD]", "")
            conf = sentence_log_prob[i].item()
            #truth = " ".join(truth)
            #truth = truth.replace("[PAD]", "")
            #truth = truth.replace(" ##", "")
            #truth = truth.replace("##", "")
            #print(sent[0])
            res.append(sent + "\t" + str(conf))
        random.shuffle(res)
        return res




class GenerativeGPT2QGWrapper(nn.Module):
    def __init__(self, GenerativeGPT2QG):
        super(GenerativeGPT2QGWrapper, self).__init__()
        self.model = GenerativeGPT2QG

    def forward(self, batch_size, max_len, sample=True, tmp=1, top_p=0.9):
        return self.model.generate(batch_size,
                                   max_len,
                                   sample=sample,
                                   tmp=tmp,
                                   top_p=top_p)



class GenerativeGPT2QA2(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GenerativeGPT2QA2, self).__init__(config)
        self.transformer = GPT2ModelQA2(config)
        self.loss = nn.CrossEntropyLoss(weight=None,
                                        size_average=None,
                                        ignore_index=-1,
                                        reduce=None,
                                        reduction='none')
        self.mc_loss = nn.CrossEntropyLoss(weight=None, reduction='mean')
        self.init_weights()
        self.transformer.tie_weights()

    def add_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(self,
                input_ids,
                output_ids,
                token_type_ids=None,
                attention_mask=None,
                lm=True,
                mc=False,
                num_labels=5,
                labels=None):
        out = self.transformer(input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               lm=lm,
                               num_labels=num_labels,
                               mc=mc)

        #target = input_ids[:, 1:]
        output = []

        if lm:
            decoded = out.pop(0)
            data_loss = self.loss(decoded.view(-1, self.config.vocab_size),
                                  output_ids.contiguous().view(-1)).view(
                                      input_ids.size(0), -1)
            output.append(data_loss)

        if mc:
            logits = out.pop(0)
            if labels is not None:
                data_loss = self.mc_loss(logits, labels.view(-1))
                output.append(data_loss)
            else:
                output.append(logits)
        return output

    def generate(self,
                 input_ids,
                 max_len,
                 sample=True,
                 tmp=1,
                 label=None,
                 top_p=0.9):
        stopping_word = self.tokenizer.eos_token
        batch_size = 1
        res = []

        stopping_word = self.tokenizer.convert_tokens_to_ids([stopping_word
                                                              ])[0]

        word = input_ids.new_zeros(batch_size, 0)
        sentence_log_prob = 0.0
        while word.size(1) < max_len:
            #print(input_ids.size())
            #print(word.size())
            feed_input_ids = torch.cat([input_ids, word], 1)
            attention_mask = torch.ones_like(feed_input_ids)
            decoded = self.transformer(feed_input_ids,
                                       token_type_ids=None,
                                       attention_mask=attention_mask,
                                       lm=True,
                                       mc=False,
                                       generation=True)
            logits = decoded / tmp
            if sample:
                filtered_logits = top_k_top_p_filtering(logits, top_p=top_p)
                tmp_dist = F.softmax(filtered_logits, -1)
                topi = torch.multinomial(tmp_dist, 1)
                word_log_prob = F.nll_loss(
                    torch.log(tmp_dist), topi.view(-1), reduction='none') * -1
            else:
                tmp_dist = F.softmax(logits, -1)
                p, topi = torch.topk(tmp_dist, 1)
                word_log_prob = torch.log(p.view(-1))
            sentence_log_prob += word_log_prob
            word = torch.cat([word, topi], 1)
            if topi.item() == stopping_word:
                break

        sent = input_ids[0, :]

        sent = self.tokenizer.decode(
            list(sent.cpu().numpy()),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True).replace("Q: ",
                                                       "").replace("A:", "")

        ans = word[0, :]
        #print(ans)
        ans = self.tokenizer.decode(list(ans.cpu().numpy()),
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True).replace(
                                        "[PAD]", "").replace("[CLS]", "")

        conf = sentence_log_prob.item()
        if label is not None:
            truth = label[0, :]
            truth = self.tokenizer.decode(
                list(truth.cpu().numpy()),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True).replace("[PAD]",
                                                           "").replace(
                                                               "[CLS]", "")

            res = sent + "\t" + ans + "\t" + truth + "\t" + str(conf)
        else:
            res = sent + "\t" + ans + "\t" + str(conf)

        return res

    def get_score(
            self,
            input_ids,
            max_len,
    ):
        stopping_word = self.tokenizer.eos_token
        batch_size = 1
        res = []

        stopping_word = self.tokenizer.convert_tokens_to_ids([stopping_word
                                                              ])[0]

        word = input_ids.new_zeros(batch_size, 0)
        sentence_log_prob = 0.0
        while word.size(1) < max_len:
            #print(input_ids.size())
            #print(word.size())

            feed_input_ids = torch.cat([input_ids, word], 1)
            #print(stopping_word)
            attention_mask = torch.ones_like(feed_input_ids)
            decoded = self.transformer(feed_input_ids,
                                       token_type_ids=None,
                                       attention_mask=attention_mask,
                                       lm=True,
                                       mc=False,
                                       generation=True)
            logits = decoded

            tmp_dist = F.softmax(logits, -1)
            p, topi = torch.topk(tmp_dist, 1)
            word_log_prob = torch.log(p.view(-1))
            sentence_log_prob += word_log_prob.item()
            word = torch.cat([word, topi], 1)
            if topi.item() == stopping_word:
                break
        return sentence_log_prob


class GenerativeGPT2QA2Wrapper(nn.Module):
    def __init__(self, GenerativeGPT2QA2):
        super(GenerativeGPT2QA2Wrapper, self).__init__()
        self.model = GenerativeGPT2QA2

    def forward(self,
                input_ids,
                max_len,
                sample=True,
                tmp=1,
                label=None,
                top_p=0.9):
        return self.model.generate(input_ids,
                                   max_len,
                                   sample=sample,
                                   tmp=tmp,
                                   label=label,
                                   top_p=top_p)
class GenerativeGPT2WinograndeChoice(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GenerativeGPT2WinograndeChoice, self).__init__(config)
        self.transformer = GPT2ModelQAWinogrande(config)
        self.loss = nn.CrossEntropyLoss(weight=None,
                                        size_average=None,
                                        ignore_index=-1,
                                        reduce=None,
                                        reduction='none')
        self.mc_loss = nn.CrossEntropyLoss(weight=None, reduction='mean')
        self.init_weights()
        self.transformer.tie_weights()

    def add_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(self,
                input_ids,
                output_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None):
        out = self.transformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        #target = input_ids[:, 1:]
        #output = []

        decoded = out
        data_loss = self.loss(decoded.view(-1, self.config.vocab_size),
                              output_ids.contiguous().view(-1)).view(
                                  input_ids.size(0), -1)
        #output.append(data_loss)

        return data_loss

    def generate(
            self,
            input_ids,
            max_len,
            tmp=1,
    ):
        stopping_word = self.tokenizer.sep_token
        batch_size = 2
        res = []

        stopping_word = self.tokenizer.convert_tokens_to_ids([stopping_word
                                                              ])[0]

        word = input_ids.new_zeros(batch_size, 0)
        input_ids_tmp = input_ids
        input_ids = input_ids.expand(batch_size,  input_ids.size(1))

        while word.size(0) != 0 and word.size(1) < max_len:
            #print(input_ids.size())
            #print(word.size())

            feed_input_ids = torch.cat([input_ids, word], 1)
            attention_mask = torch.ones_like(feed_input_ids)
            decoded = self.transformer(feed_input_ids,
                                       token_type_ids=None,
                                       attention_mask=attention_mask,
                                       generation=True)
            logits = decoded / tmp
            tmp_dist = logits
            if word.size(1) == 0:

                _, topi = torch.topk(tmp_dist, 2)

                topi = torch.stack([topi[0, 0], topi[1, 1]], 0).view(2, 1)

            else:
                _, topi = torch.topk(tmp_dist, 1)

            word = torch.cat([word, topi], 1)
            #print(word)
            #x = input("stop")
            stop_idx = (topi.view(-1) == stopping_word).nonzero().view(-1)
            if stop_idx.size(0) != 0:
                word_continue = word.index_select(
                    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                input_ids = input_ids.index_select(
                    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                word_stop = word.index_select(0, stop_idx)
                word = word_continue
                res.append(word_stop)
        if word.size(0) > 0:
            res.append(word)

        word = []

        for it in res:
            word += list(torch.unbind(it))

        sent = input_ids_tmp[0, :]

        sent = self.tokenizer.convert_ids_to_tokens(list(sent.cpu().numpy()))
        sent = self.tokenizer.convert_tokens_to_string(sent).replace(
            self.tokenizer.sep_token, "").strip()

        ans = word[0]
        #print(ans)
        ans = self.tokenizer.convert_ids_to_tokens(list(ans.cpu().numpy()))
        ans1 = self.tokenizer.convert_tokens_to_string(ans).replace(
            self.tokenizer.sep_token, "").replace("[PAD]", "").strip()

        ans = word[1]
        #print(ans)
        ans = self.tokenizer.convert_ids_to_tokens(list(ans.cpu().numpy()))
        ans2 = self.tokenizer.convert_tokens_to_string(ans).replace(
            self.tokenizer.sep_token, "").replace("[PAD]", "").strip()

        res = sent + "\t" + ans1 + "\t" + ans2

        return res

    def get_score(
            self,
            input_ids,
            max_len,
    ):
        stopping_word = self.tokenizer.eos_token
        batch_size = 1
        res = []

        stopping_word = self.tokenizer.convert_tokens_to_ids([stopping_word
                                                              ])[0]

        word = input_ids.new_zeros(batch_size, 0)
        sentence_log_prob = 0.0
        while word.size(1) < max_len:
            #print(input_ids.size())
            #print(word.size())

            feed_input_ids = torch.cat([input_ids, word], 1)
            #print(stopping_word)
            attention_mask = torch.ones_like(feed_input_ids)
            decoded = self.transformer(feed_input_ids,
                                       token_type_ids=None,
                                       attention_mask=attention_mask,
                                       generation=True)
            logits = decoded

            tmp_dist = F.softmax(logits, -1)
            p, topi = torch.topk(tmp_dist, 1)
            word_log_prob = torch.log(p.view(-1))
            sentence_log_prob += word_log_prob.item()
            word = torch.cat([word, topi], 1)
            if topi.item() == stopping_word:
                break
        return sentence_log_prob

class GenerativeGPT2QASwag(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GenerativeGPT2QASwag, self).__init__(config)
        self.transformer = GPT2ModelQAWinogrande(config)
        self.loss = nn.CrossEntropyLoss(weight=None,
                                        size_average=None,
                                        ignore_index=-1,
                                        reduce=None,
                                        reduction='none')
        self.mc_loss = nn.CrossEntropyLoss(weight=None, reduction='mean')
        self.init_weights()
        self.transformer.tie_weights()

    def add_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(self,
                input_ids,
                output_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None):
        out = self.transformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        #target = input_ids[:, 1:]
        #output = []

        decoded = out
        data_loss = self.loss(decoded.view(-1, self.config.vocab_size),
                              output_ids.contiguous().view(-1)).view(
                                  input_ids.size(0), -1)
        #output.append(data_loss)

        return data_loss

    def generate(self,
                 input_ids,
                 max_len,
                 sample=True,
                 tmp=1,
                 label=None,
                 top_p=0.9):
        stopping_word = "\n"
        batch_size = 1
        res = []

        stopping_word = self.tokenizer.convert_tokens_to_ids([stopping_word
                                                              ])[0]

        word = input_ids.new_zeros(batch_size, 0)
        #sentence_log_prob = 0.0
        while word.size(1) < max_len:
            #print(input_ids.size())
            #print(word.size())
            feed_input_ids = torch.cat([input_ids, word], 1)
            attention_mask = torch.ones_like(feed_input_ids)
            decoded = self.transformer(feed_input_ids,
                                       token_type_ids=None,
                                       attention_mask=attention_mask,
                                       generation=True)
            logits = decoded / tmp
            if sample:
                filtered_logits = top_k_top_p_filtering(logits, top_p=top_p)
                tmp_dist = F.softmax(filtered_logits, -1)
                topi = torch.multinomial(tmp_dist, 1)
            #    word_log_prob = F.nll_loss(
            #        torch.log(tmp_dist), topi.view(-1), reduction='none') * -1
            else:
                tmp_dist = F.softmax(logits, -1)
                p, topi = torch.topk(tmp_dist, 1)
                #word_log_prob = torch.log(p.view(-1))
            #sentence_log_prob += word_log_prob
            word = torch.cat([word, topi], 1)
            if topi.item() == stopping_word:
                #print("stopped")
                break

        sent = input_ids[0, :]

        sent = self.tokenizer.convert_ids_to_tokens(list(sent.cpu().numpy()))
        sent = self.tokenizer.convert_tokens_to_string(sent).replace(
            "\n", "").replace("<|endoftext|>", "").strip()

        ans = word[0]
        #print(ans)
        ans = self.tokenizer.convert_ids_to_tokens(list(ans.cpu().numpy()))
        ans = self.tokenizer.convert_tokens_to_string(ans).replace(
            "\n", "").replace("<|endoftext|>", "").replace("[PAD]", "").replace("[SEP]", "").strip()



        if label is not None:
            truth = label[0, :]
            truth = self.tokenizer.convert_ids_to_tokens(list(truth.cpu().numpy()))
            truth = self.tokenizer.convert_tokens_to_string(truth).replace(
                "\n", "").replace("<|endoftext|>", "").replace("[PAD]", "").replace("[SEP]", "").strip()

            res = sent + "\t" + ans + "\t" + truth
        else:
            res = sent + "\t" + ans

        return res


class GenerativeGPT2QDSwag(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GenerativeGPT2QDSwag, self).__init__(config)
        self.transformer = GPT2ModelQAWinogrande(config)
        self.loss = nn.CrossEntropyLoss(weight=None,
                                        size_average=None,
                                        ignore_index=-1,
                                        reduce=None,
                                        reduction='none')
        self.mc_loss = nn.CrossEntropyLoss(weight=None, reduction='mean')
        self.init_weights()
        self.transformer.tie_weights()

    def add_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(
            self,
            input_ids,
            output_ids,
            token_type_ids=None,
            attention_mask=None,
    ):
        out = self.transformer(input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)

        #target = input_ids[:, 1:]
        #output = []

        decoded = out
        data_loss = self.loss(decoded.view(-1, self.config.vocab_size),
                              output_ids.contiguous().view(-1)).view(
                                  input_ids.size(0), -1)
        #output.append(data_loss)

        return data_loss

    def generate(self,
                 input_ids,
                 max_len,
                 sample=True,
                 tmp=1,
                 top_p=0.9):
        stopping_word = self.tokenizer.eos_token
        batch_size = 1
        res = []

        stopping_word = self.tokenizer.convert_tokens_to_ids([stopping_word
                                                              ])[0]

        word = input_ids.new_zeros(batch_size, 0)

        while word.size(1) < max_len:
            #print(input_ids.size())
            #print(word.size())
            feed_input_ids = torch.cat([input_ids, word], 1)
            attention_mask = torch.ones_like(feed_input_ids)
            decoded = self.transformer(feed_input_ids,
                                       token_type_ids=None,
                                       attention_mask=attention_mask,
                                       generation=True)
            logits = decoded / tmp
            if sample:
                filtered_logits = top_k_top_p_filtering(logits, top_p=top_p)
                tmp_dist = F.softmax(filtered_logits, -1)
                topi = torch.multinomial(tmp_dist, 1)

            else:
                tmp_dist = F.softmax(logits, -1)
                p, topi = torch.topk(tmp_dist, 1)


            word = torch.cat([word, topi], 1)
            if topi.item() == stopping_word:
                break

        sent = input_ids[0, :]


        sent = self.tokenizer.convert_ids_to_tokens(list(sent.cpu().numpy()))
        sent = self.tokenizer.convert_tokens_to_string(sent).replace(
            "\n", "").replace("<|endoftext|>", "").strip()

        ans = word[0]
        #print(ans)
        ans = self.tokenizer.convert_ids_to_tokens(list(ans.cpu().numpy()))
        ans = self.tokenizer.convert_tokens_to_string(ans).replace(
            "\n", "").replace("<|endoftext|>", "").replace("[PAD]", "").replace("[SEP]", "").strip()



        res = (sent, ans)

        return res

    def batch_generate(self,
                       input_ids,
                       max_len,
                       num_distractors=3,
                       sample=True,
                       tmp=1,
                       top_p=0.9):
        input_ids = input_ids.expand(num_distractors, input_ids.size(1))
        batch_size = num_distractors

        stopping_word = self.tokenizer.eos_token
        res = []

        stopping_word = self.tokenizer.convert_tokens_to_ids([stopping_word
                                                              ])[0]
        model_device = next(self.transformer.parameters()).device
        word = input_ids.new_zeros(batch_size, 0)

        while word.size(0) != 0 and word.size(1) < max_len:
            #print(input_ids.size())
            #print(word.size())
            feed_input_ids = torch.cat([input_ids, word], 1)
            attention_mask = torch.ones_like(feed_input_ids)

            decoded = self.transformer(feed_input_ids,
                                       token_type_ids=None,
                                       attention_mask=attention_mask,
                                       generation=True)
            logits = decoded / tmp
            if sample:
                filtered_logits = top_k_top_p_filtering(logits, top_p=top_p)
                tmp_dist = F.softmax(filtered_logits, -1)
                topi = torch.multinomial(tmp_dist, 1)

            else:
                tmp_dist = F.softmax(logits, -1)
                p, topi = torch.topk(tmp_dist, 1)

            word = torch.cat([word, topi], 1)
            stop_idx = (topi.view(-1) == stopping_word).nonzero().view(-1)
            if stop_idx.size(0) != 0:
                word_continue = word.index_select(
                    0, (topi.view(-1) != stopping_word).nonzero().view(-1))

                #token_type_ids = token_type_ids.index_select(
                #    0, (topi.view(-1) != stopping_word).nonzero().view(-1))
                word_stop = word.index_select(0, stop_idx)
                #word_stop_pad = word_stop.new_zeros(word_stop.size(0), max_len).fill_(self.tokenizer.pad_token_id)
                #word_stop_pad[:, :word_stop.size(1)] = word_stop
                word = word_continue
                input_ids = input_ids[:word.size(0), :]

                res.append(word_stop)
        if word.size(0) > 0:
            res.append(word)

        sentences = []
        for it in res:
            sentences += list(torch.unbind(it))

        res = []

        for ans in sentences:

            #print(ans)
            ans = self.tokenizer.convert_ids_to_tokens(list(ans.cpu().numpy()))
            ans = self.tokenizer.convert_tokens_to_string(ans).replace(
                "\n", "").replace("<|endoftext|>", "").replace("[PAD]", "").replace("[SEP]", "").strip()



            res.append(ans)

        return res
