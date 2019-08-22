import re
import numpy as np
import os
from src.rl_tools import *

# This function is used to extract test results from train.log
def extract_test_bleu(id):
    pat_bleu = re.compile('- BLEU [^ ]+ [^ ]+ : ([^ \n]+)')
    pat_bleu1 = re.compile('- BLEU-1 : ([^ \n]+)')
    pat_bleu2 = re.compile('- BLEU-2 : ([^ \n]+)')
    pat_bleu3 = re.compile('- BLEU-3 : ([^ \n]+)')
    pat_bleu4 = re.compile('- BLEU-4 : ([^ \n]+)')
    pat_epoch = re.compile('Starting epoch ([^ \n]+) \.\.\.')
    pat_ppl = re.compile('- ppl_sw_pm_test -> ([^ \n]+)')

    path_train_log = id + '/train.log'
    bleu_dic = {}
    with open(path_train_log) as f_in:
        con = f_in.read()
    end = -10
    start = con.find('====================== Starting epoch', end+10)
    while start != -1:
        end = start
        start = con.find('====================== Starting epoch', end+10)
        sub_con = con[end:start]
        try:
            epoch = pat_epoch.search(sub_con).group(1)
            bleu = pat_bleu.search(sub_con).group(1)
            bleu1 = pat_bleu1.search(sub_con).group(1)
            bleu2 = pat_bleu2.search(sub_con).group(1)
            bleu3 = pat_bleu3.search(sub_con).group(1)
            bleu4 = pat_bleu4.search(sub_con).group(1)
            ppl = pat_ppl.search(sub_con).group(1)
            # print(' ' + id + '-' + epoch + ' & ' + bleu1 + ' & ' + bleu2 + ' & ' + bleu3  + ' & ' +   bleu4 + ' & ' +  bleu  + '\\\\')
            # print(' \\hline')
            bleu_dic[int(epoch)] = [float(bleu1), float(bleu2), float(bleu3), float(bleu4), float(bleu), float(ppl)]
        except:
            break
    return bleu_dic

def show_bleu_score(model_name, epoches):
    bleu_dic = extract_test_bleu(model_name)
    bleu_lis = []
    bleu_lis1 = []
    bleu_lis2 = []
    bleu_lis3 = []
    bleu_lis4 = []
    ppl = []
    for i in epoches:
        bleu_lis1.append(bleu_dic[i][0])
        bleu_lis2.append(bleu_dic[i][1])
        bleu_lis3.append(bleu_dic[i][2])
        bleu_lis4.append(bleu_dic[i][3])
        bleu_lis.append(bleu_dic[i][4])
        ppl.append(bleu_dic[i][5])
    print(model_name + ' 1-gram bleu\t' + str(np.mean(bleu_lis1)))
    print(model_name + ' 2-gram bleu\t' + str(np.mean(bleu_lis2)))
    print(model_name + ' 3-gram bleu\t' + str(np.mean(bleu_lis3)))
    print(model_name + ' 4-gram bleu\t' + str(np.mean(bleu_lis4)))
    print(model_name + '\t\t bleu' + str(np.mean(bleu_lis)))
    print(model_name + '\t\t ppl' + str(np.mean(ppl)))
    print(model_name + '\t\t bleus' + str(bleu_lis))
    print(model_name + '\t\t ppls' + str(ppl))
    print('')

def get_inter_ratio(path):
    inter_ratio_lis = []
    with open(path) as f_in_sw:
        sw = f_in_sw.readlines()
    with open('./data/data_acc/jueju7_out.vl.txt') as f_in_pm:
        pm = f_in_pm.readlines()
    for i in range(len(sw)):
        sent_sw = sw[i].strip()
        sent_pm = pm[i].strip()
        inter_ratio = reward_func_ap(sent_sw, sent_pm, n=5)
        inter_ratio_lis.append(inter_ratio)
    return np.mean(inter_ratio_lis)

def show_inter_ratio(model_name, epoches):
    inter_ratio_lis = []
    for j in epoches:
        path = '../poem-prose_dump/' + model_name + '/hyp' + str(j) + '.pm-sw.valid.txt'
        inter_ratio_lis.append(get_inter_ratio(path))
    print('intersection_ratio for ' + model_name + ':\t' + str(np.mean(inter_ratio_lis)))

def get_repetition_ratio(path):
    with open(path) as f_in_:
        lines = f_in_.readlines()
    repetition_ratio_lis = []
    for i in range(960):
        line = lines[i].strip()
        repetition_ratio = 1.0 - len(set(line)) / float(len(line))
        repetition_ratio_lis.append(repetition_ratio)
    return np.mean(repetition_ratio_lis)

def show_repe_ratio(model_name, epoches):
    repe_ratio_lis = []
    for j in epoches:
        path = model_name + '/hyp' + str(j) + '.pm-sw.valid.txt'
        repe_ratio_lis.append(get_repetition_ratio(path))
    print('repetition_ratio for ' + model_name + ':\t' + str(np.mean(repe_ratio_lis)))

def main():
    # base 0-49
    # rl10 0-41 + anti copy loss
    # rl18 9-37 + anti repetition loss
    # rl22 9-54 + anti copy loss & anti repetition loss
    epoches = np.array(range(30, 35))
    # epoches = np.array(range(25, 30))

    model1 = '/mnt/nfs/work1/hongyu/pengshancai/exp/caibase'
    model2 = '/mnt/nfs/work1/hongyu/pengshancai/exp/4902580'
    model3 = './dumped/test/4948046'
    # model3 = 'testlstm/4976233'
    model4 = '/mnt/nfs/work1/hongyu/pengshancai/exp/4949781'
    # model4 = 'testlstm/4976257'

    show_bleu_score(model1, epoches)
    show_bleu_score(model2, epoches-8)
    show_bleu_score(model3, epoches)
    show_bleu_score(model4, epoches)
    show_repe_ratio(model1, epoches)
    show_repe_ratio(model2, epoches-8)
    show_repe_ratio(model3, epoches)
    show_repe_ratio(model4, epoches)




main()
