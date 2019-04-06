import re

def extract_test_bleu(id, epochs = []):
    pat_bleu = re.compile('- BLEU [^ ]+ [^ ]+ : ([^ \n]+)')
    pat_bleu1 = re.compile('- BLEU-1 : ([^ \n]+)')
    pat_bleu2 = re.compile('- BLEU-2 : ([^ \n]+)')
    pat_bleu3 = re.compile('- BLEU-3 : ([^ \n]+)')
    pat_bleu4 = re.compile('- BLEU-4 : ([^ \n]+)')
    pat_epoch = re.compile('Starting epoch ([^ \n]+) \.\.\.')
    path_train_log = './dump/' + id + '/train.log'
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
            print(' ' + id + '-' + epoch + ' & ' + bleu1 + ' & ' + bleu2 + ' & ' + bleu3  + ' & ' +   bleu4 + ' & ' +  bleu  + '\\\\')
            print(' \\hline')
        except:
            break



# This function is only used to extract test results from share2 evaluation res
def extract_test_bleu_share2(id, model_name = ''):
    pat_bleu = re.compile('- BLEU [^ ]+ [^ ]+ : ([^ \n]+)')
    pat_bleu1 = re.compile('- BLEU-1 : ([^ \n]+)')
    pat_bleu2 = re.compile('- BLEU-2 : ([^ \n]+)')
    pat_bleu3 = re.compile('- BLEU-3 : ([^ \n]+)')
    pat_bleu4 = re.compile('- BLEU-4 : ([^ \n]+)')

    path_train_log = './dump/share_2/' + id + '/train.log'
    with open(path_train_log) as f_in:
        con = f_in.read()
    start = con.find('Evaluating Para sw -> pm (test) ...')
    end = con.find('Evaluating Mono pm  (valid)', start)
    sub_con = con[start:end]
    bleu = pat_bleu.search(sub_con).group(1)
    bleu1 = pat_bleu1.search(sub_con).group(1)
    bleu2 = pat_bleu2.search(sub_con).group(1)
    bleu3 = pat_bleu3.search(sub_con).group(1)
    bleu4 = pat_bleu4.search(sub_con).group(1)
    print(' ' + model_name + '_' + id + ' & ' + bleu1 + ' & ' + bleu2 + ' & ' + bleu3  + ' & ' +   bleu4 + ' & ' +  bleu  + '\\\\')
    print(' \\hline')

