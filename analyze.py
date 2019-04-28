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
    path_train_log = '../poem-prose_dump/' + id + '/train.log'
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
            # print(' ' + id + '-' + epoch + ' & ' + bleu1 + ' & ' + bleu2 + ' & ' + bleu3  + ' & ' +   bleu4 + ' & ' +  bleu  + '\\\\')
            # print(' \\hline')
            bleu_dic[int(epoch)] = [float(bleu1), float(bleu2), float(bleu3), float(bleu4), float(bleu)]
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
    for i in epoches:
        bleu_lis1.append(bleu_dic[i][0])
        bleu_lis2.append(bleu_dic[i][1])
        bleu_lis3.append(bleu_dic[i][2])
        bleu_lis4.append(bleu_dic[i][3])
        bleu_lis.append(bleu_dic[i][-1])
    print(model_name + ' 1-gram bleu\t' + str(np.mean(bleu_lis1)))
    print(model_name + ' 2-gram bleu\t' + str(np.mean(bleu_lis2)))
    print(model_name + ' 3-gram bleu\t' + str(np.mean(bleu_lis3)))
    print(model_name + ' 4-gram bleu\t' + str(np.mean(bleu_lis4)))
    print(model_name + '\t\t bleu' + str(np.mean(bleu_lis)))
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
    for i in range(19200):
        line = lines[i].strip()
        repetition_ratio = 1.0 - len(set(line)) / float(len(line))
        repetition_ratio_lis.append(repetition_ratio)
    return np.mean(repetition_ratio_lis)

def show_repe_ratio(model_name, epoches):
    repe_ratio_lis = []
    for j in epoches:
        path = '../poem-prose_dump/' + model_name + '/hyp' + str(j) + '.pm-sw.valid.txt'
        repe_ratio_lis.append(get_repetition_ratio(path))
    print('repetition_ratio for ' + model_name + ':\t' + str(np.mean(repe_ratio_lis)))

def main():
    # base 0-49
    # rl10 0-41 + anti copy loss
    # rl18 9-37 + anti repetition loss
    # rl22 9-54 + anti copy loss & anti repetition loss
    epoches = np.array(range(30, 36))
    model1 = 'caibase'
    model2 = 'cairl10'
    model3 = 'cairl18'
    model4 = 'cairl22'
    show_bleu_score(model1, epoches)
    show_bleu_score(model2, epoches)
    show_bleu_score(model3, epoches-8)
    show_bleu_score(model4, epoches-8)

    show_repe_ratio(model1, epoches)
    show_repe_ratio(model2, epoches)
    show_repe_ratio(model3, epoches - 8)
    show_repe_ratio(model4, epoches - 8)
    print('repetition_ratio for real sanwen:\t' + str(get_repetition_ratio('./data/data_acc/sanwen.vl.txt')))

    show_inter_ratio(model1, epoches)
    show_inter_ratio(model2, epoches)
    show_inter_ratio(model3, epoches-8)
    show_inter_ratio(model4, epoches-8)


















# # for id in ['cairlbase', 'cairl10', 'cairl18']:
# model_name1 = 'caibase'
# bleu_dic1 = extract_test_bleu(model_name1)
# bleu_lis1 = []
# bleu_lis11 = []
# bleu_lis12 = []
# bleu_lis13 = []
# bleu_lis14 = []
# for i in epoches:
#     bleu_lis11.append(bleu_dic1[i][0])
#     bleu_lis12.append(bleu_dic1[i][1])
#     bleu_lis13.append(bleu_dic1[i][2])
#     bleu_lis14.append(bleu_dic1[i][3])
#     bleu_lis1.append(bleu_dic1[i][-1])
#
# model_name2 = 'cairl10'
# bleu_dic2 = extract_test_bleu(model_name2)
# bleu_lis2 = []
# bleu_lis21 = []
# bleu_lis22 = []
# bleu_lis23 = []
# bleu_lis24 = []
# for i in epoches:
#     bleu_lis21.append(bleu_dic2[i][0])
#     bleu_lis22.append(bleu_dic2[i][1])
#     bleu_lis23.append(bleu_dic2[i][2])
#     bleu_lis24.append(bleu_dic2[i][3])
#     bleu_lis2.append(bleu_dic2[i][-1])
#
# model_name3 = 'cairl18'
# bleu_dic3 = extract_test_bleu(model_name3)
# bleu_lis3 = []
# bleu_lis31 = []
# bleu_lis32 = []
# bleu_lis33 = []
# bleu_lis34 = []
# for i in epoches:
#     bleu_lis31.append(bleu_dic3[i-8][0])
#     bleu_lis32.append(bleu_dic3[i-8][1])
#     bleu_lis33.append(bleu_dic3[i-8][2])
#     bleu_lis34.append(bleu_dic3[i-8][3])
#     bleu_lis3.append(bleu_dic3[i-8][-1])
#
# model_name4 = 'cairl22'
# bleu_dic4 = extract_test_bleu(model_name4)
# bleu_lis4 = []
# bleu_lis41 = []
# bleu_lis42 = []
# bleu_lis43 = []
# bleu_lis44 = []
# for i in epoches:
#     bleu_lis41.append(bleu_dic3[i-8][0])
#     bleu_lis42.append(bleu_dic3[i-8][1])
#     bleu_lis43.append(bleu_dic3[i-8][2])
#     bleu_lis44.append(bleu_dic3[i-8][3])
#     bleu_lis4.append(bleu_dic4[i-8][-1])
#
# print(model_name1 + ' 1-gram\t' + str(np.mean(bleu_lis11)))
# print(model_name2 + ' 1-gram\t' + str(np.mean(bleu_lis21)))
# print(model_name3 + ' 1-gram\t' + str(np.mean(bleu_lis31)))
# print(model_name4 + ' 1-gram\t' + str(np.mean(bleu_lis41)))
# print('')
# print(model_name1 + ' 2-gram\t' + str(np.mean(bleu_lis12)))
# print(model_name2 + ' 2-gram\t' + str(np.mean(bleu_lis22)))
# print(model_name3 + ' 2-gram\t' + str(np.mean(bleu_lis32)))
# print(model_name4 + ' 2-gram\t' + str(np.mean(bleu_lis42)))
# print('')
# print(model_name1 + ' 3-gram\t' + str(np.mean(bleu_lis13)))
# print(model_name2 + ' 3-gram\t' + str(np.mean(bleu_lis23)))
# print(model_name3 + ' 3-gram\t' + str(np.mean(bleu_lis33)))
# print(model_name4 + ' 3-gram\t' + str(np.mean(bleu_lis43)))
# print('')
# print(model_name1 + ' 4-gram\t' + str(np.mean(bleu_lis14)))
# print(model_name2 + ' 4-gram\t' + str(np.mean(bleu_lis24)))
# print(model_name3 + ' 4-gram\t' + str(np.mean(bleu_lis34)))
# print(model_name4 + ' 4-gram\t' + str(np.mean(bleu_lis44)))
# print('')
# print(model_name1 + '\t' + str(np.mean(bleu_lis1)))
# print(model_name2 + '\t' + str(np.mean(bleu_lis2)))
# print(model_name3 + '\t' + str(np.mean(bleu_lis3)))
# print(model_name4 + '\t' + str(np.mean(bleu_lis4)))


# # This function is only used to extract test results from share2 evaluation res
#
# def extract_test_bleu_share2(id, model_name = ''):
#     pat_bleu = re.compile('- BLEU [^ ]+ [^ ]+ : ([^ \n]+)')
#     pat_bleu1 = re.compile('- BLEU-1 : ([^ \n]+)')
#     pat_bleu2 = re.compile('- BLEU-2 : ([^ \n]+)')
#     pat_bleu3 = re.compile('- BLEU-3 : ([^ \n]+)')
#     pat_bleu4 = re.compile('- BLEU-4 : ([^ \n]+)')
#
#     path_train_log = './dump/share_2/' + id + '/train.log'
#     with open(path_train_log) as f_in:
#         con = f_in.read()
#     start = con.find('Evaluating Para sw -> pm (test) ...')
#     end = con.find('Evaluating Mono pm  (valid)', start)
#     sub_con = con[start:end]
#     bleu = pat_bleu.search(sub_con).group(1)
#     bleu1 = pat_bleu1.search(sub_con).group(1)
#     bleu2 = pat_bleu2.search(sub_con).group(1)
#     bleu3 = pat_bleu3.search(sub_con).group(1)
#     bleu4 = pat_bleu4.search(sub_con).group(1)
#     print(' ' + model_name + '_' + id + ' & ' + bleu1 + ' & ' + bleu2 + ' & ' + bleu3  + ' & ' +   bleu4 + ' & ' +  bleu  + '\\\\')
#     print(' \\hline')


# This is to test if rl really helps reduce copy

# path_rl_sw = './dump/cairl10/hyp20.pm-sw.valid.txt'
# path_bl_sw = './dump/caibase/hyp20.pm-sw.valid.txt'
# path_pm = './dump/share_2/hyp20.pm.valid.txt'
#
# with open(path_rl_sw) as f_in:
#     rl_sw = f_in.readlines()
# with open(path_bl_sw) as f_in:
#     bl_sw = f_in.readlines()
# with open(path_pm) as f_in:
#     pm = f_in.readlines()
#
# assert len(pm) == len(bl_sw) and len(pm) == len(rl_sw)
#
# from src.rl_tools import *
#
#
#
# mlrl12_model = 'cairl22'
# mlrl1_model = 'cairl10'
# mlrl2_model = 'cairl18'
# ml_model = 'caibase'
# mlrl12_lis = []
# mlrl1_lis = []
# mlrl2_lis = []
# ml_lis = []
#
# # for j in range(25, 35):
# #     path = '../poem-prose_dump/' + mlrl12_model + '/hyp' + str(j) + '.pm-sw.valid.txt'
# #     mlrl12_lis.append(get_inter_ratio(path))
#
# for j in range(25, 35):
#     path = '../poem-prose_dump/' + mlrl2_model + '/hyp' + str(j-8) + '.pm-sw.valid.txt'
#     mlrl2_lis.append(get_inter_ratio(path))
#
# for j in range(25, 35):
#     path = '../poem-prose_dump/' + mlrl1_model + '/hyp' + str(j) + '.pm-sw.valid.txt'
#     mlrl1_lis.append(get_inter_ratio(path))
#
# for j in range(25, 35):
#     path = '../poem-prose_dump/' + ml_model + '/hyp' + str(j) + '.pm-sw.valid.txt'
#     ml_lis.append(get_inter_ratio(path))
#
# print('inter_ratio for ML only:\t' + str(np.mean(ml_lis)))
# print('inter_ratio for ML + RL1:\t' + str(np.mean(mlrl1_lis)))
# print('inter_ratio for ML + RL2:\t' + str(np.mean(mlrl2_lis)))
# print('inter_ratio for ML + RL1 + RL2:\t' + str(np.mean(mlrl12_lis)))

# inter_rl_lis = []
# inter_bl_lis = []
# for i in range(len(pm)):
#     sent_rl = rl_sw[i].strip()
#     sent_bl = bl_sw[i].strip()
#     sent_pm = pm[i].split('\t###\t')[1].strip()
#     inter_bl = reward_func_ap(sent_bl, sent_pm, n=5)
#     inter_rl = reward_func_ap(sent_rl, sent_pm, n=5)
#     inter_bl_lis.append(inter_bl)
#     inter_rl_lis.append(inter_rl)

# print('5-gram similar rate for ML only:\t' + str(np.mean(inter_bl_lis)))
# print('5-gram similar rate for ML + RL:\t' + str(np.mean(inter_rl_lis)))

# # This is to test if rl really helps reduce repetition
#
# from src.rl_tools import *
#
# path_rl_sw = './dump/cairl18/hyp5.pm-sw.valid.txt'
# path_bl_sw = './dump/caibase/hyp11.pm-sw.valid.txt'
# path_pm = './dump/share_2/hyp11.pm.valid.txt'
#
# with open(path_rl_sw) as f_in:
#     rl_sw = f_in.readlines()
# with open(path_bl_sw) as f_in:
#     bl_sw = f_in.readlines()
# with open(path_pm) as f_in:
#     pm = f_in.readlines()
#
# assert len(pm) == len(bl_sw) and len(pm) == len(rl_sw)
#
# unique_ratio_bl_lis = []
# unique_ratio_rl_lis = []
# for i in range(len(pm)):
#     sent_rl = rl_sw[i].strip()
#     sent_bl = bl_sw[i].strip()
#     sent_pm = pm[i].split('\t###\t')[1].strip()
#     unique_ratio_bl = len(set(sent_bl)) / float(len(sent_bl))
#     unique_ratio_rl = len(set(sent_rl)) / float(len(sent_rl))
#     unique_ratio_bl_lis.append(unique_ratio_bl)
#     unique_ratio_rl_lis.append(unique_ratio_rl)
#
# print('unique_ratio for ML only:\t' + str(np.mean(unique_ratio_bl_lis)))
# print('unique_ratio for ML + RL:\t' + str(np.mean(unique_ratio_rl_lis)))


# This is to get the unique ratio of ordinary text and generated text

# path_rl_sw = './dump/cairl18/hyp5.pm-sw.valid.txt'
# path_bl_sw = './dump/caibase/hyp20.pm-sw.valid.txt'
# path_gold = './data/data_acc/jueju7_out.vl.txt'

# with open(path_rl_sw) as f_in:
#     rl_sw = f_in.readlines()
# with open(path_bl_sw) as f_in:
#     bl_sw = f_in.readlines()
# with open(path_gold) as f_in:
#     gd_sw = f_in.readlines()
#
# assert len(pm) == len(bl_sw) and len(pm) == len(rl_sw)

# unique_ratio_bl_lis = []
# unique_ratio_rl_lis = []
# unique_ratio_gd_lis = []
# for i in range(len(pm)):
#     sent_rl = rl_sw[i].strip()
#     sent_bl = bl_sw[i].strip()
#     sent_gd = pm[i].split('\t###\t')[1].strip()
#     unique_ratio_bl = len(set(sent_bl)) / float(len(sent_bl))
#     unique_ratio_rl = len(set(sent_rl)) / float(len(sent_rl))
#     unique_ratio_gd = len(set(sent_gd)) / float(len(sent_gd))
#     unique_ratio_bl_lis.append(unique_ratio_bl)
#     unique_ratio_rl_lis.append(unique_ratio_rl)
#     unique_ratio_gd_lis.append(unique_ratio_gd)


#
#
# mlrl12_model = 'cairl22'
# mlrl1_model = 'cairl10'
# mlrl2b_model = 'cairl29'
# mlrl2a_model = 'cairl18'
# ml_model = 'caibase'
# mlrl12_lis = []
# mlrl1_lis = []
# mlrl2a_lis = []
# mlrl2b_lis = []
# ml_lis = []
#
# epoches = range(10, 15)
#
# # for j in epoches:
# #     path = './dump/' + mlrl12_model + '/hyp' + str(j) + '.pm-sw.valid.txt'
# #     mlrl12_lis.append(get_unique_ratio(path))
#
# # for j in epoches:
# #     path = '../poem-prose_dump/' + mlrl2_model + '/hyp' + str(j-8) + '.pm-sw.valid.txt'
# #     mlrl2_lis.append(get_repetition_ratio(path))
# for j in epoches:
#     path = '../poem-prose_dump/' + mlrl2a_model + '/hyp' + str(j-8) + '.pm-sw.valid.txt'
#     mlrl2a_lis.append(get_repetition_ratio(path))
#
# for j in epoches:
#     path = '../poem-prose_dump/' + mlrl2b_model + '/hyp' + str(j-20) + '.pm-sw.valid.txt'
#     mlrl2b_lis.append(get_repetition_ratio(path))
#
#
# for j in epoches:
#     path = '../poem-prose_dump/' + mlrl1_model + '/hyp' + str(j) + '.pm-sw.valid.txt'
#     mlrl1_lis.append(get_repetition_ratio(path))
#
# for j in epoches:
#     path = '../poem-prose_dump/' + ml_model + '/hyp' + str(j) + '.pm-sw.valid.txt'
#     ml_lis.append(get_repetition_ratio(path))
#
# for j in epoches:
#     path = '../poem-prose_dump/' + mlrl12_model + '/hyp' + str(j-8) + '.pm-sw.valid.txt'
#     mlrl12_lis.append(get_repetition_ratio(path))
#
#
#
# print('repetition_ratio for ML only:\t' + str(np.mean(ml_lis)))
# print('repetition_ratio for ML + RL1:\t' + str(np.mean(mlrl1_lis)))
# print('repetition_ratio for ML + RL2a:\t' + str(np.mean(mlrl2a_lis)))
# print('repetition_ratio for ML + RL2b:\t' + str(np.mean(mlrl2b_lis)))
# print('repetition_ratio for ML + RL1 + RL2:\t' + str(np.mean(mlrl12_lis)))
#
# # print('repetition_ratio for ML + RL1 + RL2:\t' + str(np.mean(mlrl12_lis)))
# print('repetition_ratio for gold:\t' + str(get_repetition_ratio('./data/data_acc/sanwen.vl.txt')))
#
# pat = re.compile('hyp([0-9]+)\.(.+)')
# file_names = os.listdir('./dump/cairl10_')
# for file_name in file_names:
#     if file_name.startswith('hyp'):
#         se = pat.search(file_name)
#         id = int(se.group(1))
#         if id >= 29:
#             continue
#         new_file_name = file_name.replace(str(id), str(id+28))
#         os.rename('./dump/cairl10_/' + file_name, './dump/cairl10_/' + new_file_name)
#
#
#
#
#
