from snownlp import SnowNLP
import time

max_iter = 50

def shorten_sents(sents_in, min_len = 30, max_len = 50):
    if len(sents_in) <= min_len:
        return sents_in.strip()
    cnt = 5
    sents = SnowNLP(sents_in)
    summary = select_sents(sents, cnt)
    iter = 1
    # if len(summary) >= min_len and len(summary) <= max_len:
    while len(summary) > max_len:
        cnt -= 1
        summary = select_sents(sents, cnt)
        iter += 1
        if iter > max_iter:
            return sents_in
    while len(summary) < min_len:
        cnt += 1
        summary = select_sents(sents, cnt)
        iter += 1
        if iter > max_iter:
            return sents_in
    return summary

def select_sents(sents, cnt):
    summary = ''
    selected_sents = sents.summary(cnt)
    for sent in sents.sentences:
        if sent in selected_sents:
            summary += sent + '，'
    return summary[:-1] + '。'

# def test():
#     start = time.time()
#     lens = []
#     with open('./data/data_acc/sanwen.tr.txt') as f_in:
#         con = f_in.readlines()
#     cnt = 0
#     print(len(con))
#     for line in con:
#         sum = shorten_sents(line)
#         cnt += 1
#         lens.append(len(sum))
#         # if cnt % 50 == 0:
#         print(cnt)
#         if cnt == 2000:
#             break
#     end = time.time()
#     print(end-start)