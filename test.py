#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, io, sys, math
import numpy as np
import torch
import collections

PADDING_IDX=1

def get_data(input_sents, tokenizer, issanwen, dopmpad):
    sent_str = []
    positions = []
    sentences = []
    sentences_len = []
    sent_str_abs = []
    positions_abs = []
    sentences_abs = []
    sentences_abs_len = []
    length_in_count = np.zeros(int(MAX_SENT_LEN/10)+1)
    unk_words = {}
    line_count=0
    too_long_sent_count = 0
    long_sent_count = 0
    # for ind in range(len(input_sents)):
    for ind in range(len(input_sents)):
        sent=input_sents[ind]
        if issanwen:
            realmax_len = np.random.normal(loc=69.0, scale=10.0, size=None) #TODO: 99
        else:
            realmax_len=MAX_SENT_LEN
        # realmax_len=MAX_SENT_LEN
        
        if realmax_len > MAX_SENT_LEN:
            realmax_len = MAX_SENT_LEN 
        realmax_len = int(realmax_len)

        if len(sent) > realmax_len:
            # print("Long sentence with len %i in line %i." % (len(sent),line_count))
            sent=sent[0:realmax_len]
            sent = list(zng(sent)) # ends with punc
            if len(sent) == 0:
                sent=''
                too_long_sent_count+=1
            else:
                assert len(sent)==1
                sent = sent[0]
                long_sent_count+=1
        token_s = tokenizer.tokenize(sent)
        # if len(token_s) == 0:
        #     print("Empty sentence in line %i." % line_count)
        if len(token_s) > 21: #TODO: 31
            # index sentence words
            indexed = tokenizer.convert_tokens_to_ids(token_s)
            unk_idxs = [i for i, e in enumerate(indexed) if e == 100]
            for unk_idx in unk_idxs:
                w = sent[unk_idx] 
                unk_words[w] = unk_words.get(w, 0) + 1
            if dopmpad:
                ind_len = len(indexed)
                indexed = np.array(indexed)
                sliced = list(range(2,ind_len+1,2))+list(range(2,ind_len+1,2))
                # logger.info(ind)
                # logger.info(indexed.shape)
                indexed = np.insert(indexed, sliced, [PADDING_IDX]*ind_len)
                # logger.info(indexed.shape)
                # logger.info(indexed)

            # add sentence
            sent_str.append(sent)
            positions.append([len(sentences), len(sentences) + len(indexed)])
            sentences_len.append(len(indexed))
            sentences.extend(indexed)
            sentences.append(-1)

            if issanwen:
                summary = shorten_sents(sent, min_len=31, max_len=45)
                token_s_abs = tokenizer.tokenize(summary)
                indexed_abs = tokenizer.convert_tokens_to_ids(token_s_abs)
                sent_str_abs.append(summary)
                positions_abs.append([len(sentences_abs), len(sentences_abs) + len(indexed_abs)])
                sentences_abs_len.append(len(indexed_abs))
                sentences_abs.extend(indexed_abs)
                sentences_abs.append(-1)

            line_count+=1
            if len(token_s) > MAX_SENT_LEN:
                length_in_count[-1] += 1
            else:
                length_in_count[int(len(token_s)/10)] += 1
        # else:
        #     print("Short sentence in line %i. <=10" % line_count)


    # tensorize data
    positions = torch.LongTensor(positions)
    sentences = torch.LongTensor(sentences)
    positions_abs = torch.LongTensor(positions_abs)
    sentences_abs = torch.LongTensor(sentences_abs)
    data = {
        'dico': tokenizer,
        'positions': positions,
        'sentences': sentences,
        'positions_abs': positions_abs,
        'sentences_abs': sentences_abs,
        'unk_words': unk_words,
    }
    logger.info('long sentence count:')
    logger.info(long_sent_count)
    logger.info('long sentence that can not convert count:')
    logger.info(too_long_sent_count)
    length_in_count = length_in_count/np.sum(length_in_count)
    logger.info('sentence length bin count:')
    logger.info(length_in_count)
    logger.info('sentence length mean and std:')
    logger.info(np.mean(sentences_len))
    logger.info(np.std(sentences_len))
    if issanwen:
        logger.info('abstract sentence length mean and std:')
        logger.info(np.mean(sentences_abs_len))
        logger.info(np.std(sentences_abs_len))
    return data, sent_str, sent_str_abs

def batch_sentences_pm(sentences):
    """
    Take as input a list of n sentences (torch.LongTensor vectors) and return
    a tensor of size (s_len, n) where s_len is the length of the longest
    sentence, and a vector lengths containing the length of each sentence.
    """
    lengths = torch.LongTensor([len(s) + 2 for s in sentences])
    sent = torch.LongTensor(lengths.max(), lengths.size(0)).fill_(0)
    sent[0] = 4
    for i, s in enumerate(sentences):
        sent[1:lengths[i] - 1, i].copy_(s)
        sent[lengths[i] - 1, i] = 2
    return sent, lengths


# data = torch.load('./data/data_pad/jueju7_out.vl.pth')
dopmpad = False
epoch = 0
lang1 = 'sw'
lang1 = 'pm'
data_type = 'test1'
txt_path = '...'
tokenizer = ...

valid_input = []
with io.open(txt_path, "r", encoding='utf8') as f:
    for line in f:
        s = line.rstrip()
        valid_input.append(txt_path)

data, sent, sent_abs = get_data(valid_input, tokenizer, False, dopmpad)

data['positions'] = data['positions'].numpy()
n_sentences = len(data['positions'])
indices = np.arange(n_sentences)
batches = np.array_split(indices, math.ceil(len(indices) * 1. / 32))

txt_tone_enh = []
for sentence_ids in batches:
  pos = data['positions'][sentence_ids]
  sents = [data['sentences'][a:b] for a, b in pos]
  sent2_, len2_ = batch_sentences_pm(sents)
  lang2_id = 0
  sent2_enh, len2_enh = double_para(sent2_, len2_, lang2_id, do_pad=False, do_bos=self.params.do_bos, do_sep=self.params.do_sep)
  txt_tone_enh.extend(convert_to_text(sent2_enh, len2_enh, self.dico[lang2], lang2_id, self.params, do_pad=False, do_bos=self.params.do_bos, do_sep=self.params.do_sep))

hyp_name_enh = 'hyp{0}.{1}-{2}.{3}.tone_ehance.txt'.format(epoch, lang1, lang2, data_type)
hyp_path_enh = os.path.join(params.dump_path, hyp_name_enh)
# export sentences to hypothesis file / restore BPE segmentation
with open(hyp_path_enh, 'w', encoding='utf-8') as f:
    f.write('\n'.join(txt_tone_enh) + '\n')


# import jieba
# txt_path = './data/data_pad/jueju7_out.vl.txt'
# valid_input = []
# with io.open(txt_path, "r", encoding='utf8') as f:
#     for line in f:
#         s = line.rstrip()
#         seg_list = jieba.cut(s)
#         valid_input.append(seg_list)
#         print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

