#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, io
import sys
import re

txt_path = 'data/poem/poem7.txt'
ton_path = 'data/poem/intonation7.txt'
PUNC = [u'？',u'，',u'！']

with io.open(txt_path, "r", encoding='utf8') as f1:
  with io.open(ton_path, "r", encoding='utf8') as f2:
    l1=f1.readline()
    l2=f2.readline()
    count=0
    while l1:
      s1 = l1.rstrip()
      s2 = l2.rstrip()
      # if len(s1)!=0:
      #   if not (s1[7] in PUNC):
      #     print(count)
      if len(s1) != len(s2):
        print(count)
        exit()
      l1=f1.readline()
      l2=f2.readline()
      count+=1



