import os
import torch
from logging import getLogger


logger = getLogger()


BOS_WORD = '[SOS]'
EOS_WORD = '[EOS]'
PAD_WORD = '[PAD]'
UNK_WORD = '[UNK]'
CLS_WORD = '[CLS]'
SEP_WORD = '[SEP]'
MASK_WORD = '[MASK]'

SPECIAL_WORD = '[special%i]'
SPECIAL_WORDS = 10
