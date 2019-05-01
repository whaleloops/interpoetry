#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, io, sys, math
import numpy as np
import torch
import collections


def load_vocab_rev(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with io.open(vocab_file, "r", encoding="utf8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[index] = token
            index += 1
    return vocab

def batch_sentences(sentences, lang_id):
    """
    Take as input a list of n sentences (torch.LongTensor vectors) and return
    a tensor of size (s_len, n) where s_len is the length of the longest
    sentence, and a vector lengths containing the length of each sentence.
    """
    assert type(lang_id) is int
    lengths = torch.LongTensor([len(s) + 2 for s in sentences])
    sent = torch.LongTensor(lengths.max(), lengths.size(0)).fill_(0)
    sent[0] = 4
    for i, s in enumerate(sentences):
        sent[1:lengths[i] - 1, i].copy_(s)
        sent[lengths[i] - 1, i] = 2
    return sent, lengths


# data = torch.load('./data/data_pad/jueju7_out.vl.pth')
tone_file = './data/vocab_tone_pure.txt'
data = torch.load('./data/data_acc/poem_jueju7_para.pm.pth')
data['sentences'] = data['sentences'].numpy()
data['positions'] = data['positions'].numpy()
dictionary = data['dico']
n_sentences = len(data['positions'])
indices = np.arange(n_sentences)
batches = np.array_split(indices, math.ceil(len(indices) * 1. / 32))

dictionary.ids_to_tones = load_vocab_rev(tone_file)
print(dictionary.ids_to_tones)

for sentence_ids in batches:
  pos1 = data['positions'][sentence_ids]
  sents = [data['sentences'][a:b] for a, b in pos1]
  for sent in sents:
    print (dictionary.convert_ids_to_tones(sent))


# ['1', '2', '1', '1', '1', '2', '2', '0', '1', '1', '2', '2', '1', '1', '2', '0',
#  '1', '1', '1', '2', '2', '1', '1', '0', '1', '2', '2', '1', '1', '2', '2', '0']
# ['1', '1', '3', '2', '1', '1', '2', '0', '2', '2', '2', '1', '1', '2', '2', '0',
#  '1', '2', '1', '1', '2', '2', '1', '0', '1', '1', '2', '2', '1', '1', '2', '0']
# ['1', '1', '2', '2', '1', '1', '1', '0', '2', '2', '2', '1', '1', '1', '2', '0',
#  '2', '2', '1', '1', '3', '2', '1', '0', '2', '1', '2', '2', '1', '1', '2', '0']
# ['2', '1', '2', '2', '1', '0', '2', '0', '2', '2', '2', '1', '1', '1', '2', '0',
#  '2', '1', '1', '1', '2', '2', '1', '0', '1', '1', '2', '2', '1', '1', '2', '0']
# ['1', '1', '2', '2', '1', '1', '2', '0', '2', '2', '1', '1', '1', '2', '2', '0',
#  '1', '3', '1', '1', '2', '3', '1', '0', '1', '1', '2', '2', '1', '1', '2', '0']

['1', '2', '1', '1', '1', '2', '2', '0', '1', '1', '2', '2', '1', '1', '2', '0',
 '1', '1', '1', '2', '2', '1', '3', '0', '1', '2', '2', '1', '1', '2', '2', '0']
['3', '1', '3', '2', '1', '1', '2', '0', '2', '2', '2', '1', '1', '2', '2', '0',
 '1', '2', '1', '1', '2', '2', '1', '0', '1', '1', '2', '2', '3', '1', '2', '0']
['1', '1', '2', '2', '1', '3', '1', '0', '2', '2', '2', '1', '1', '1', '2', '0',
 '2', '2', '1', '1', '3', '2', '1', '0', '2', '1', '2', '2', '1', '1', '2', '0']
['2', '1', '2', '2', '1', '0', '2', '0', '2', '2', '2', '1', '1', '1', '2', '0',
 '2', '1', '1', '1', '2', '2', '1', '0', '3', '1', '2', '2', '1', '1', '2', '0']
['1', '1', '2', '2', '1', '1', '2', '0', '2', '2', '3', '1', '1', '2', '2', '0',
 '1', '3', '1', '1', '2', '3', '1', '0', '1', '1', '2', '2', '1', '1', '2', '0']

# 近寒食雨草萋萋，著麦苗风柳映堤。等是有家归未得，杜鹃休向耳边啼。
# 绝域从军计惘然，东南幽恨满词笺。一箫一剑平生意，负尽狂名十五年。
# 莫道秋江离别难，舟船明日是长安。吴姬缓舞留君醉，随意青枫白露寒。
# 鹦鹉洲头浪飐沙，青楼春望日将斜。衔泥燕子争归舍，独自狂夫不忆家。
# 一自萧关起战尘，河湟隔断异乡春。汉儿尽作胡儿语，却向城头骂汉人。
# 溪水将桥不复回，小舟犹倚短篙开。交情得似山溪渡，不管风波去又来。