import numpy as np
import os
import torch
from torch.nn import functional as F

# read word2id
id2word = {}
print(os.getcwd())
with open('./data/vocab.txt') as f_in:
    con = f_in.readlines()
for id in range(len(con)):
    id2word[id] = con[id].strip()
word2id = dict(zip(id2word.values(), id2word.keys()))
pad_id = word2id['[PAD]']
bos_id = [4,5]
eos_id = word2id['[EOS]']

# Given a distribution, sample a list of tokens from it
def get_samples(scores):
    scores = scores.cpu()
    L, B, V = scores.shape
    samples = []
    with torch.no_grad():
        for l in range(L):
            probs = F.softmax(scores[l], -1)
            sample_l = torch.multinomial(probs, 1).view(-1)
            samples.append(sample_l.numpy())
    return np.array(samples, dtype=int)

# Given a distribution, get the argmax tokens from it
def get_bases(scores):
    return torch.argmax(scores, dim=-1).cpu().numpy()

def find_ngrams(input_list, n):
  return [ngram for ngram in zip(*[input_list[i:] for i in range(n)])]

def filter_sent(sent_in):
    sent_out = []
    for token in sent_in:
        if token != pad_id:
            sent_out.append(token)
    return np.array(sent_out)

# This reward function calculates the intersection ratio of 5-grams of sw and generated pm
def reward_func_ap(sw_pred, pm_gold, n = 5):
    sw_pred = filter_sent(sw_pred)
    ngrams_sw = find_ngrams(sw_pred, n)
    ngrams_pm = find_ngrams(pm_gold, n)
    intersection_ratio = len(set(ngrams_pm).intersection(ngrams_sw))/float(len(ngrams_pm))
    return intersection_ratio

# This reward function calculates the unique ratio of one sentence
def reward_func_ar(sw_pred, thresh, reward_type = 'punish'):
    sw_pred = filter_sent(sw_pred)
    unique_ratio = len(set(sw_pred))/float(len(sw_pred))
    assert reward_type in ['encourage', 'punish']
    if reward_type == 'encourage':
        if unique_ratio > thresh:
            return 1.0
        else:
            return unique_ratio
    else:
        if unique_ratio > thresh:
            return 1.0
        else:
            return unique_ratio - thresh

# def reward_func_ar(sw_pred, thresh, reward_type = 'punish'):
#     sw_pred = filter_sent(sw_pred)
#     sent_length = len(sw_pred)
#     vocab_size = len(set(sw_pred))
#     rep_ratio = (sent_length-vocab_size) / float(sent_length)
#     assert reward_type in ['encourage', 'punish']
#     if reward_type == 'encourage':
#         if rep_ratio < thresh:
#             return 1.0
#         else:
#             return -1 * rep_ratio
#     else:
#         if rep_ratio < thresh:
#             return 1.0
#         else:
#             return thresh - rep_ratio

# This reward function calculates the intersection ratio of bleu score of generated pm and abstract sw
def reward_func_cv(pm_pred, sw, len_sw, n=1):
    sw1, sw2 = sw[:int(len_sw/2)], sw[int(len_sw/2) :len_sw]
    pm_pred = filter_sent(pm_pred)
    ngrams_pm = find_ngrams(pm_pred, n)
    ngrams_sw1 = find_ngrams(sw1, n)
    ngrams_sw2 = find_ngrams(sw2, n)
    intersection_ratio1 = len(set(ngrams_pm).intersection(ngrams_sw1)) / float(len(ngrams_pm))
    intersection_ratio2 = len(set(ngrams_pm).intersection(ngrams_sw2)) / float(len(ngrams_pm))
    return np.min([0, intersection_ratio2 - intersection_ratio1])


def get_weights_ap(bases_in, samples_in, pm_golds):
    pm_golds = pm_golds.cpu().numpy()
    bases = bases_in.cpu().numpy()
    samples = samples_in.cpu().numpy()
    weights = []
    assert samples.shape == bases.shape
    seq_len, bsz = bases.shape
    for i in range(bsz):
        weight = reward_func_ap(bases[:,i], pm_golds[:,i]) - reward_func_ap(samples[:,i], pm_golds[:,i])
        weights.append([weight for j in range(seq_len)])
    weights = np.array(weights).transpose()
    return torch.Tensor(weights).view(-1)

def get_weights_ar(bases_in, thresh, reward_type = 'punish'):
    bases = bases_in.cpu().numpy()
    weights = []
    seq_len, bsz = bases.shape
    for i in range(bsz):
        weight = reward_func_ar(bases[:,i], thresh, reward_type)
        weights += [weight for j in range(seq_len)]
    weights = np.array(weights).transpose()
    return torch.Tensor(weights).view(-1)

def get_weights_cv(bases_in, samples_in, sws, len_sws):
    sws = sws.cpu().numpy()
    len_sws = len_sws.cpu().numpy()
    bases = bases_in.cpu().numpy()
    samples = samples_in.cpu().numpy()
    weights = []
    assert samples.shape == bases.shape
    seq_len, bsz = bases.shape
    for i in range(bsz):
        weight = reward_func_cv(samples[:, i], sws[:, i], len_sws[i]) - reward_func_cv(bases[:, i], sws[:, i], len_sws[i])
        weights.append([weight for j in range(seq_len)])
    weights = np.array(weights).transpose()
    return torch.Tensor(weights).view(-1)

def init_loss_rl_fn(loss_weight):
    loss_fn_no_mean = torch.nn.CrossEntropyLoss(weight=loss_weight, reduction='none')
    return loss_fn_no_mean





