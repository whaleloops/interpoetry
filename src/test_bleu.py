import os, re
import subprocess
from collections import OrderedDict
from logging import getLogger
import numpy as np
import torch
from torch import nn

TOOLS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
BLEU_SCRIPT_PATH = os.path.join(TOOLS_PATH, 'mosesdecoder/scripts/generic/multi-bleu-ch.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH), "Moses not found. Please be sure you downloaded Moses in %s" % TOOLS_PATH


def eval_moses_bleu(ref, hyp, return_all=False):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(ref) and os.path.isfile(hyp)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    bleu_score = -1
    if result.startswith('BLEU'):
        bleu_score = np.float32(result[7:result.index(',')])
        if return_all:
            bleus = re.search(r", (.+?)/(.+?)/(.+?)/(.+?) ", result)
            bleus = [bleus.group(group_idx) for group_idx in range(1, 5)]
            bleus = [np.float32(b) for b in bleus]
            bleu_score = [bleu_score] + bleus
    else:
        print('Impossible to parse BLEU score! "%s"' % result)
    return bleu_score

ref_path='/home/pengshancai/poem-prose_transfer/data/data/poem7_out.vl.txt'
hyp_path3='/home/pengshancai/poem-prose_transfer/dumped/test/4696894/hyp8.pm-sw-pm.valid.txt'
# bleu = eval_moses_bleu(ref_path, hyp_path3)
# print(bleu)
bleu = eval_moses_bleu(ref_path, hyp_path3, return_all=True)
print(bleu)