# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

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
